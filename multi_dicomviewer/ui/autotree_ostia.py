"""全冠動脈描出: set the coronary ostia by hand, one after another
(user 2026-11-28).

「LM起始部もしくはLAD/LCX起始部とRCA起始部を順に設定してください」 —
the user moves the centreline crossing (the CT viewer's crosshair) onto
each ostium and presses OK; the next step follows. LMなし (LAD and LCX leave
the aorta separately) replaces the LM step by LAD + LCX. A step can be
skipped (e.g. no RCA wanted); 戻る goes one step back. When every step is
done, 抽出開始 runs the extraction. The window is modeless (stays on top)
so the CT can be paged / rotated while it is open.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton,
                             QRadioButton, QVBoxLayout)

from multi_dicomviewer.i18n import t


class OstiaDialog(QDialog):
    """*get_point*() → the crosshair (world mm, 3) or None; *on_run*(ostia)
    with ostia = [(role, xyz), …] when 抽出開始 is pressed."""

    def __init__(self, parent, get_point, on_run, on_cancel=None):
        super().__init__(parent)
        self.setWindowTitle(t("全冠動脈描出：起始部の設定"))
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(False)
        self._get_point = get_point
        self._on_run = on_run
        self._on_cancel = on_cancel
        self._pts = {}                    # role → xyz (None = skipped)
        self._step = 0
        lay = QVBoxLayout(self)
        head = QLabel(t("LM起始部もしくはLAD/LCX起始部とRCA起始部を順に設定して"
                        "ください。"))
        head.setWordWrap(True)
        f = head.font()
        f.setBold(True)
        head.setFont(f)
        lay.addWidget(head)
        row = QHBoxLayout()
        self._lm_rb = QRadioButton(t("LMあり"))
        self._nolm_rb = QRadioButton(t("LMなし（LAD・LCXが別々に起始）"))
        self._lm_rb.setChecked(True)
        self._lm_rb.toggled.connect(self._mode_changed)
        row.addWidget(self._lm_rb)
        row.addWidget(self._nolm_rb)
        row.addStretch(1)
        lay.addLayout(row)
        self._rows = QLabel("")
        self._rows.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._rows)
        self._ask = QLabel("")
        self._ask.setWordWrap(True)
        self._ask.setStyleSheet("color:#1a5fb4;")
        lay.addWidget(self._ask)
        btns = QHBoxLayout()
        self._ok = QPushButton(t("OK（この位置を設定）"))
        self._ok.setDefault(True)
        self._ok.clicked.connect(self._set_here)
        self._skip = QPushButton(t("スキップ"))
        self._skip.clicked.connect(self._skip_step)
        self._back = QPushButton(t("戻る"))
        self._back.clicked.connect(self._step_back)
        self._run = QPushButton(t("抽出開始"))
        self._run.clicked.connect(self._start)
        self._stop = QPushButton(t("中止"))
        self._stop.clicked.connect(self.reject)
        for b in (self._ok, self._skip, self._back, self._run, self._stop):
            btns.addWidget(b)
        lay.addLayout(btns)
        self._refresh()

    # ---- steps ------------------------------------------------------------
    def roles(self):
        return ["LM", "RCA"] if self._lm_rb.isChecked() else \
            ["LAD", "LCX", "RCA"]

    def _mode_changed(self, _on=None):
        self._pts = {k: v for k, v in self._pts.items() if k == "RCA"}
        self._step = 0
        while self._step < len(self.roles()) and \
                self.roles()[self._step] in self._pts:
            self._step += 1
        self._refresh()

    def _set_here(self):
        rs = self.roles()
        if self._step >= len(rs):
            return
        p = self._get_point() if self._get_point is not None else None
        if p is None:
            self._ask.setText(t("CT の十字（センターライン交点）が取れません。"))
            return
        self._pts[rs[self._step]] = np.asarray(p, float).reshape(3).copy()
        self._step += 1
        self._refresh()

    def _skip_step(self):
        rs = self.roles()
        if self._step < len(rs):
            self._pts[rs[self._step]] = None
            self._step += 1
            self._refresh()

    def _step_back(self):
        if self._step > 0:
            self._step -= 1
            self._pts.pop(self.roles()[self._step], None)
            self._refresh()

    def ostia(self):
        return [(r, self._pts[r]) for r in self.roles()
                if self._pts.get(r) is not None]

    def _refresh(self):
        rs = self.roles()
        lines = []
        for i, r in enumerate(rs):
            p = self._pts.get(r, "unset")
            if p is None:
                st = "<span style='color:#888;'>" + t("スキップ") + "</span>"
            elif isinstance(p, str):
                st = "<span style='color:#c05a00;'>" + t("未設定") + "</span>"
            else:
                st = ("<span style='color:#2e7d32;'>" + t("設定済み")
                      + f" ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})</span>")
            mark = "▶ " if i == self._step else "&nbsp;&nbsp;&nbsp;"
            lines.append(f"{mark}<b>{t('{r} 起始部', r=r)}</b>: {st}")
        self._rows.setText("<br>".join(lines))
        done = self._step >= len(rs)
        if done:
            self._ask.setText(t("すべて設定しました。「抽出開始」を押してください。")
                              if self.ostia()
                              else t("起始部が1つもありません。「戻る」で設定してください。"))
        else:
            self._ask.setText(t("センターライン交点を {r} の起始部に合わせて「OK」を"
                                "クリックしてください。", r=rs[self._step]))
        self._ok.setEnabled(not done)
        self._skip.setEnabled(not done)
        self._back.setEnabled(self._step > 0)
        self._run.setEnabled(done and bool(self.ostia()))

    def _start(self):
        ost = self.ostia()
        if not ost:
            return
        self.accept()
        if self._on_run is not None:
            self._on_run(ost)

    def reject(self):
        super().reject()
        if self._on_cancel is not None:
            self._on_cancel()
