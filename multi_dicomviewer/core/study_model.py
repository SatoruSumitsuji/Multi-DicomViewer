"""Patient > Study > Series hierarchy shared by both viewer modules.

The whole point of the single-app design is that XA and CT series for the
same patient live under one Patient node, so the shell can put them
side by side and keep them in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Modality(str, Enum):
    XA = "XA"          # X-ray angiography (invasive cath cine)
    CT = "CT"          # cardiac CT (CCTA)
    IVUS = "IVUS"      # intravascular ultrasound (pull-back cine)
    OTHER = "OTHER"

    @classmethod
    def from_dicom(cls, value: Optional[str]) -> "Modality":
        v = (value or "").upper()
        if v == "XA":
            return cls.XA
        if v == "CT":
            return cls.CT
        if v == "IVUS":
            return cls.IVUS
        return cls.OTHER


@dataclass
class Series:
    series_uid: str
    modality: Modality
    description: str
    files: list[str] = field(default_factory=list)
    number: Optional[int] = None
    #: Raw DICOM Modality string (e.g. "XA"/"CT"/"IVUS"/"US"/"OT").
    #: Kept verbatim for display so kinds the app doesn't specially
    #: handle show their real code instead of a generic "OTHER".
    dicom_modality: str = ""
    #: Sortable acquisition timestamp ("YYYYMMDDHHMMSS.ffff" style, from
    #: AcquisitionDateTime / AcquisitionDate+Time). "" sorts last.
    acq_time: str = ""
    #: DICOM AcquisitionNumber of the first file seen in this series; None
    #: when the tag is absent. Series-level display only — individual
    #: instances within a series can in principle differ. Retained because
    #: the Export dialog can use it as a filename field; the tree no
    #: longer shows it (AcquisitionNumber was rarely populated in the
    #: user's data — InstanceNumber replaced it in the UI).
    acq_number: Optional[int] = None
    #: DICOM InstanceNumber of the first file seen in this series. Used
    #: as the tree's "Instance No" column and a sortable key. For
    #: multi-frame single-file series this is the file's InstanceNumber;
    #: for packed-XA split rows it equals the row's own file InstanceNumber.
    instance_number: Optional[int] = None
    #: Total images/frames in the series = sum of NumberOfFrames across its
    #: files, so a single-file multi-frame cine (NM/US/XA) reports e.g. 72 not
    #: 1. 0 until scan_folder fills it in; image_count falls back to the file
    #: count meanwhile.
    n_images: int = 0

    @property
    def kind(self) -> str:
        return self.dicom_modality or self.modality.value

    @property
    def image_count(self) -> int:
        """Tree 'N img' count — the summed frame count, falling back to the
        file count when not yet computed."""
        return self.n_images or len(self.files)

    @property
    def label(self) -> str:
        n = f"#{self.number} " if self.number is not None else ""
        desc = f" — {self.description}" if self.description else ""
        return f"{n}{self.kind}{desc}  [{self.image_count} img]"


@dataclass
class Study:
    study_uid: str
    description: str
    date: str
    series: dict[str, Series] = field(default_factory=dict)

    @property
    def label(self) -> str:
        d = self.date or "????-??-??"
        desc = self.description or "(no study description)"
        return f"{d}  {desc}"


@dataclass
class Patient:
    patient_id: str
    name: str
    studies: dict[str, Study] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.patient_id})"

    def modalities(self) -> set[Modality]:
        out: set[Modality] = set()
        for st in self.studies.values():
            for se in st.series.values():
                out.add(se.modality)
        return out

    def has_both_modalities(self) -> bool:
        m = self.modalities()
        return Modality.XA in m and Modality.CT in m
