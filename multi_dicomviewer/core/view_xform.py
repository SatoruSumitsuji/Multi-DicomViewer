"""Display-transform geometry shared by the IVUS cross-section canvas and the
long-axis builder.

A cross-section is shown as: RAW frame → orientation (horizontal flip first,
then ``rot`` × 90° clockwise — see ImageCanvas._oriented) → free rotation by
``deg`` degrees clockwise on screen (QPainter.rotate, y-down) about the
oriented image's centre. "View" coordinates below are the oriented image's
pixel coordinates AFTER that free rotation, i.e. where a raw pixel is actually
seen.

The long-axis strip samples the RAW frames, so the cut line the user sees on
screen must be carried back through these transforms — per frame, because the
Multi-Rot angle keyframes give every frame its own free rotation (user
2026-11-24: rotated ~180°, the strip showed the red and blue sides swapped).
"""
from __future__ import annotations

import math

import numpy as np


def oriented_size(w: float, h: float, rot: int) -> tuple[float, float]:
    """(w, h) of a raw w×h frame after the orientation."""
    return (h, w) if rot % 2 else (w, h)


def orient_point(x, y, w, h, flip, rot):
    """Raw (x, y) of a w×h frame → oriented-image coordinates."""
    if flip:
        x = w - x
    for _ in range(rot % 4):              # 90° clockwise: (x, y) → (h − y, x)
        x, y = h - y, x
        w, h = h, w
    return x, y


def unorient_point(x, y, w, h, flip, rot):
    """Oriented-image (x, y) → raw coordinates of the w×h raw frame."""
    ow, oh = oriented_size(w, h, rot)
    for _ in range(rot % 4):              # inverse of one CW turn
        # forward: (x, y) [size ow0×oh0] → (oh0 − y, x); here ow = oh0
        x, y = y, ow - x
        ow, oh = oh, ow
    if flip:
        x = w - x
    return x, y


def rotate_about(x, y, cx, cy, deg):
    """Rotate (x, y) about (cx, cy) by *deg* clockwise on a y-down screen
    (the QPainter.rotate sense)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def raw_to_view(x, y, w, h, flip, rot, deg):
    """Raw (x, y) → where it is seen (oriented + free-rotated)."""
    ox, oy = orient_point(x, y, w, h, flip, rot)
    ow, oh = oriented_size(w, h, rot)
    return rotate_about(ox, oy, ow / 2.0, oh / 2.0, deg)


def view_to_raw(x, y, w, h, flip, rot, deg):
    """Inverse of :func:`raw_to_view`."""
    ow, oh = oriented_size(w, h, rot)
    ox, oy = rotate_about(x, y, ow / 2.0, oh / 2.0, -deg)
    return unorient_point(ox, oy, w, h, flip, rot)


def view_dirs_to_raw(ux, uy, flip, rot, degs) -> np.ndarray:
    """A screen direction (ux, uy) → the raw-frame direction, one per entry of
    *degs* (each frame's free rotation). Returns an (n, 2) float array."""
    a = np.radians(-np.asarray(degs, dtype=np.float64))     # undo free rot
    c, s = np.cos(a), np.sin(a)
    dx = c * ux - s * uy
    dy = s * ux + c * uy
    for _ in range(rot % 4):              # undo each CW turn: (dx, dy) → (dy, −dx)
        dx, dy = dy, -dx
    if flip:
        dx = -dx
    return np.stack([dx, dy], axis=1)
