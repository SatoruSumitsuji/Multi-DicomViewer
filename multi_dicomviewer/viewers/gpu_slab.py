"""GPU slab MIP for the pygfx CT viewer (wgpu compute shader).

The THICK (slab-MIP) image used to be built on the CPU in NumPy — max over N
oblique planes of a trilinear sample — which cost ~340 ms per pane on an M4 at
the at-rest quality and forced the coarse-while-dragging LOD. This does the
SAME maths in a compute shader on pygfx's shared wgpu device and reads back the
float MIP (ih, iw), so the caller keeps its existing W/L / colour-band LUT →
QImage step and the overlay paints the result exactly as before.

Geometry mirrors ct_viewer_pygfx._compute_slab_qimage line for line: screen
pixel grid = linspace(0..pw, iw) × linspace(0..ph, ih), roll + pan in plane
mm, base point pc + WX·u + WY·v, planes at linspace(-th/2, th/2, N) along n,
voxel = world / spacing, out-of-volume samples read as _HU_LO.

The volume is uploaded once per series as an r16float 3-D texture (half the
memory of float32; HU error <= 1 below 2048 HU, 2 above — invisible at any
CT window) and sampled with the hardware trilinear filter on voxel centres,
which is what _trilinear_sample computes on the CPU.

Any failure (no adapter, texture too large, driver error …) raises; the viewer
then falls back to the CPU path for the rest of the session.
"""
from __future__ import annotations

import math

import numpy as np

_WGSL = """
struct Params {
    u: vec4<f32>,
    v: vec4<f32>,
    n: vec4<f32>,
    pc: vec4<f32>,
    inv_dims: vec4<f32>,   // 1/sx, 1/sy, 1/sz
    shape: vec4<f32>,      // nx, ny, nz
    pan_rot: vec4<f32>,    // px, py, cos(roll), sin(roll)
    view: vec4<f32>,       // pw, ph, scale, thick
    hu: vec4<f32>,         // hu_lo
    dims: vec4<u32>,       // iw, ih, nplanes
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> outb: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iw = P.dims.x;
    let ih = P.dims.y;
    if (gid.x >= iw || gid.y >= ih) {
        return;
    }
    let pw = P.view.x;
    let ph = P.view.y;
    let scale = P.view.z;
    let th = P.view.w;
    var sxp = 0.0;
    if (iw > 1u) { sxp = f32(gid.x) * pw / f32(iw - 1u); }
    var syp = 0.0;
    if (ih > 1u) { syp = f32(gid.y) * ph / f32(ih - 1u); }
    let aa = (sxp - pw * 0.5) / scale;
    let bb = (ph * 0.5 - syp) / scale;
    let ca = P.pan_rot.z;
    let sa = P.pan_rot.w;
    let wx = P.pan_rot.x + aa * ca - bb * sa;
    let wy = P.pan_rot.y + aa * sa + bb * ca;
    let base = P.pc.xyz + wx * P.u.xyz + wy * P.v.xyz;
    let np_ = P.dims.z;
    let shape = P.shape.xyz;
    let hu_lo = P.hu.x;
    var m = -3.0e38;
    for (var i = 0u; i < np_; i = i + 1u) {
        var t = 0.0;
        if (np_ > 1u) { t = -th * 0.5 + th * f32(i) / f32(np_ - 1u); }
        let vx = (base + t * P.n.xyz) * P.inv_dims.xyz;      // voxel coords
        var h = hu_lo;
        if (all(vx >= vec3<f32>(0.0)) && all(vx <= shape - vec3<f32>(1.0))) {
            h = textureSampleLevel(vol, samp, (vx + vec3<f32>(0.5)) / shape, 0.0).x;
        }
        m = max(m, h);
    }
    outb[gid.y * iw + gid.x] = m;
}
"""


class GpuSlab:
    """Slab-MIP compute on pygfx's shared wgpu device. GUI thread only."""

    #: z-slices per write_texture call (keeps each staging upload modest).
    _UPLOAD_CHUNK = 64

    def __init__(self, hu_lo: float):
        import pygfx as gfx
        import wgpu
        self._wgpu = wgpu
        self._dev = gfx.renderers.wgpu.get_shared().device
        self._hu_lo = float(hu_lo)
        self._pipe = self._dev.create_compute_pipeline(
            layout="auto",
            compute={"module": self._dev.create_shader_module(code=_WGSL),
                     "entry_point": "main"})
        self._samp = self._dev.create_sampler(
            mag_filter="linear", min_filter="linear",
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge")
        self._ubuf = self._dev.create_buffer(
            size=160, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._tex = None
        self._tex_view = None
        self._vol_key = None
        self._obuf = None
        self._obuf_size = 0
        self._bg = None

    # ------------------------------------------------------------ volume
    def set_volume(self, vol: np.ndarray) -> None:
        """Upload *vol* (z, y, x) if it is not the one already on the GPU."""
        key = (id(vol), vol.shape)
        if key == self._vol_key:
            return
        self.release()
        wgpu = self._wgpu
        nz, ny, nx = vol.shape
        lim = self._dev.limits.get("max-texture-dimension3d", 2048)
        if max(nx, ny, nz) > lim:
            raise RuntimeError(f"volume {vol.shape} exceeds 3-D texture limit {lim}")
        tex = self._dev.create_texture(
            size=(nx, ny, nz), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        c = self._UPLOAD_CHUNK
        for z0 in range(0, nz, c):
            chunk = np.ascontiguousarray(vol[z0:z0 + c], dtype=np.float16)
            self._dev.queue.write_texture(
                {"texture": tex, "origin": (0, 0, z0), "mip_level": 0},
                chunk,
                {"offset": 0, "bytes_per_row": nx * 2, "rows_per_image": ny},
                (nx, ny, chunk.shape[0]))
        self._tex = tex
        self._tex_view = tex.create_view()
        self._vol_key = key
        self._bg = None

    def release(self) -> None:
        if self._tex is not None:
            try:
                self._tex.destroy()
            except Exception:                            # noqa: BLE001
                pass
        self._tex = self._tex_view = self._vol_key = None
        self._bg = None

    # ------------------------------------------------------------ compute
    def mip(self, p: dict, nplanes: int) -> np.ndarray:
        """Float32 (ih, iw) slab MIP for the _slab_params snapshot *p*."""
        wgpu = self._wgpu
        self.set_volume(p["vol"])
        iw, ih = int(p["iw"]), int(p["ih"])
        nbytes = iw * ih * 4
        if self._obuf is None or self._obuf_size < nbytes:
            if self._obuf is not None:
                self._obuf.destroy()
            self._obuf = self._dev.create_buffer(
                size=nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            self._obuf_size = nbytes
            self._bg = None
        if self._bg is None:
            self._bg = self._dev.create_bind_group(
                layout=self._pipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self._ubuf,
                                                "offset": 0, "size": 160}},
                    {"binding": 1, "resource": self._tex_view},
                    {"binding": 2, "resource": self._samp},
                    {"binding": 3, "resource": {"buffer": self._obuf,
                                                "offset": 0,
                                                "size": self._obuf_size}},
                ])
        sx, sy, sz = p["dims"]
        nz, ny, nx = p["vol"].shape
        a = math.radians(p["roll"])
        u = np.zeros(40, np.float32)
        u[0:3], u[4:7], u[8:11], u[12:15] = p["u"], p["v"], p["n"], p["pc"]
        u[16:19] = (1.0 / sx, 1.0 / sy, 1.0 / sz)
        u[20:23] = (nx, ny, nz)
        u[24:28] = (p["pan"][0], p["pan"][1], math.cos(a), math.sin(a))
        u[28:32] = (p["pw"], p["ph"], p["ph"] / (2.0 * max(1e-3, p["ps"])),
                    p["thick"])
        u[32] = self._hu_lo
        u.view(np.uint32)[36:39] = (iw, ih, max(1, int(nplanes)))
        self._dev.queue.write_buffer(self._ubuf, 0, u)
        enc = self._dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._pipe)
        cp.set_bind_group(0, self._bg)
        cp.dispatch_workgroups((iw + 7) // 8, (ih + 7) // 8, 1)
        cp.end()
        self._dev.queue.submit([enc.finish()])
        raw = self._dev.queue.read_buffer(self._obuf, 0, nbytes)
        return np.frombuffer(raw, np.float32).reshape(ih, iw).copy()
