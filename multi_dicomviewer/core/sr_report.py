"""Radiation-dose Structured Report (SR) parsing — GUI-free.

Cath-lab systems archive an "Exam Protocol SR" / dose report per procedure
(SOP Class X-Ray Radiation Dose SR Storage). It carries NO pixel data — just
a content tree of per-irradiation-event dose records (TID 10001/10003). This
module turns that tree into plain dataclasses the SR viewer renders as a
table, plus a generic indent-tree text fallback for any other SR kind.

Matching is by ConceptNameCodeSequence CodeMeaning (case-insensitive). The
meanings are written into the file by the modality from the DICOM standard's
context groups, so they are stable per vendor; unknown meanings simply leave
fields None rather than failing the parse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: SOP Class UID of X-Ray Radiation Dose SR Storage.
DOSE_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.67"


def is_sr(ds) -> bool:
    """True for any Structured Report dataset (no pixel data expected)."""
    return str(getattr(ds, "Modality", "")).upper() == "SR" \
        or "ContentSequence" in ds


def is_dose_sr(ds) -> bool:
    """True when *ds* is an X-Ray Radiation Dose SR."""
    return str(getattr(ds, "SOPClassUID", "")) == DOSE_SR_SOP_CLASS


# --------------------------------------------------------------- SR helpers
def _meaning(item) -> str:
    """The content item's concept-name CodeMeaning, lower-cased ('' if none)."""
    seq = getattr(item, "ConceptNameCodeSequence", None)
    if seq:
        return str(getattr(seq[0], "CodeMeaning", "") or "").lower()
    return ""


def _children(item):
    return getattr(item, "ContentSequence", None) or []


def _num(item) -> Optional[float]:
    """A NUM content item's numeric value (None when absent/malformed)."""
    mv = getattr(item, "MeasuredValueSequence", None)
    if not mv:
        return None
    try:
        v = mv[0].NumericValue
        return float(v) if v is not None else None
    except Exception:                                    # noqa: BLE001
        return None


def _unit(item) -> str:
    mv = getattr(item, "MeasuredValueSequence", None)
    if not mv:
        return ""
    u = getattr(mv[0], "MeasurementUnitsCodeSequence", None)
    return str(getattr(u[0], "CodeValue", "") or "") if u else ""


def _value_of(item) -> str:
    """Best-effort display value of one content item, '' when valueless."""
    vt = str(getattr(item, "ValueType", "") or "")
    if vt == "NUM":
        v = _num(item)
        if v is None:
            return ""
        u = _unit(item)
        return f"{v:g} {u}".strip()
    if vt == "TEXT":
        return str(getattr(item, "TextValue", "") or "")
    if vt == "CODE":
        seq = getattr(item, "ConceptCodeSequence", None)
        return str(getattr(seq[0], "CodeMeaning", "") or "") if seq else ""
    if vt == "DATETIME":
        return str(getattr(item, "DateTime", "") or "")
    if vt == "DATE":
        return str(getattr(item, "Date", "") or "")
    if vt == "TIME":
        return str(getattr(item, "Time", "") or "")
    if vt == "UIDREF":
        return str(getattr(item, "UID", "") or "")
    if vt == "PNAME":
        return str(getattr(item, "PersonName", "") or "")
    return ""


def _find(items, meaning: str):
    """First direct child whose concept meaning equals *meaning* (ci)."""
    m = meaning.lower()
    for it in items:
        if _meaning(it) == m:
            return it
    return None


def _find_value(items, meaning: str) -> str:
    it = _find(items, meaning)
    return _value_of(it) if it is not None else ""


def _find_num(items, meaning: str) -> Optional[float]:
    it = _find(items, meaning)
    return _num(it) if it is not None else None


def _fmt_dt(raw: str) -> str:
    """'20260708132615[.frac]' → '2026-07-08 13:26:15' (best-effort)."""
    s = str(raw).split(".")[0]
    if len(s) >= 14 and s[:14].isdigit():
        return (f"{s[0:4]}-{s[4:6]}-{s[6:8]} "
                f"{s[8:10]}:{s[10:12]}:{s[12:14]}")
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return str(raw)


# ------------------------------------------------------------- dose report
@dataclass
class DoseEvent:
    """One irradiation event (a fluoro run or an acquisition run)."""
    datetime: str = ""            # formatted "YYYY-MM-DD HH:MM:SS"
    event_type: str = ""          # "Fluoroscopy" / "Stationary Acquisition" …
    protocol: str = ""
    primary_angle: Optional[float] = None      # >0 LAO, <0 RAO (deg)
    secondary_angle: Optional[float] = None    # >0 CRA, <0 CAU (deg)
    pulses: Optional[float] = None
    pulse_rate: Optional[float] = None         # pulses/s
    dap_gym2: Optional[float] = None           # Gy·m²
    dose_rp_gy: Optional[float] = None         # Gy at reference point

    @property
    def angle_text(self) -> str:
        """C-arm angle in the cath-lab convention, e.g. 'LAO 30 / CRA 25'."""
        parts = []
        if self.primary_angle is not None:
            a = self.primary_angle
            parts.append(f"{'LAO' if a >= 0 else 'RAO'} {abs(a):.0f}")
        if self.secondary_angle is not None:
            a = self.secondary_angle
            parts.append(f"{'CRA' if a >= 0 else 'CAU'} {abs(a):.0f}")
        return " / ".join(parts)


@dataclass
class DoseReport:
    """Parsed X-Ray Radiation Dose SR."""
    device: str = ""              # "Siemens AXIOM-Artis"
    events: list = field(default_factory=list)      # [DoseEvent]
    #: study-level totals; read from the Accumulated container when present,
    #: otherwise summed from the events.
    total_dap_gym2: Optional[float] = None
    total_dose_rp_gy: Optional[float] = None
    total_fluoro_time_s: Optional[float] = None
    n_fluoro: int = 0
    n_acq: int = 0


def _parse_event(container) -> DoseEvent:
    kids = _children(container)
    ev = DoseEvent()
    ev.datetime = _fmt_dt(_find_value(kids, "DateTime Started"))
    ev.event_type = _find_value(kids, "Irradiation Event Type")
    ev.protocol = _find_value(kids, "Acquisition Protocol")
    ev.primary_angle = _find_num(kids, "Positioner Primary Angle")
    ev.secondary_angle = _find_num(kids, "Positioner Secondary Angle")
    ev.pulses = _find_num(kids, "Number of Pulses")
    ev.pulse_rate = _find_num(kids, "Pulse Rate")
    ev.dap_gym2 = _find_num(kids, "Dose Area Product")
    ev.dose_rp_gy = _find_num(kids, "Dose (RP)")
    return ev


def parse_dose_sr(ds) -> DoseReport:
    """Parse an X-Ray Radiation Dose SR dataset into a :class:`DoseReport`.
    Never raises on missing pieces — absent values stay None/''."""
    rep = DoseReport()
    root = _children(ds)

    manu = _find_value(root, "Device Observer Manufacturer")
    model = _find_value(root, "Device Observer Model Name")
    rep.device = " ".join(x for x in (manu, model) if x)

    for it in root:
        m = _meaning(it)
        if m == "irradiation event x-ray data":
            rep.events.append(_parse_event(it))
        elif m == "accumulated x-ray dose data":
            kids = _children(it)
            rep.total_dap_gym2 = _find_num(kids, "Dose Area Product Total")
            rep.total_dose_rp_gy = _find_num(kids, "Dose (RP) Total")
            rep.total_fluoro_time_s = _find_num(kids, "Total Fluoro Time")

    fl = [e for e in rep.events if "fluoro" in e.event_type.lower()]
    rep.n_fluoro = len(fl)
    rep.n_acq = len(rep.events) - len(fl)
    # Fall back to summing events when the totals container is absent.
    if rep.total_dap_gym2 is None:
        vals = [e.dap_gym2 for e in rep.events if e.dap_gym2 is not None]
        rep.total_dap_gym2 = sum(vals) if vals else None
    if rep.total_dose_rp_gy is None:
        vals = [e.dose_rp_gy for e in rep.events if e.dose_rp_gy is not None]
        rep.total_dose_rp_gy = sum(vals) if vals else None
    if rep.total_fluoro_time_s is None:
        # pulses / pulse-rate ≒ run seconds, summed over the fluoro runs
        secs = [e.pulses / e.pulse_rate for e in fl
                if e.pulses and e.pulse_rate]
        rep.total_fluoro_time_s = sum(secs) if secs else None
    return rep


# --------------------------------------------------------- generic SR text
def generic_sr_text(ds, max_nodes: int = 4000) -> str:
    """Indented plain-text rendering of ANY SR content tree — the fallback
    for SR kinds this app doesn't specially parse."""
    lines: list[str] = []

    def walk(items, depth: int) -> None:
        for it in items:
            if len(lines) >= max_nodes:
                return
            seq = getattr(it, "ConceptNameCodeSequence", None)
            name = str(getattr(seq[0], "CodeMeaning", "") or "") if seq \
                else str(getattr(it, "ValueType", "") or "?")
            val = _value_of(it)
            lines.append("  " * depth + (f"{name}: {val}" if val else name))
            walk(_children(it), depth + 1)

    walk(_children(ds), 0)
    if len(lines) >= max_nodes:
        lines.append("…")
    return "\n".join(lines)
