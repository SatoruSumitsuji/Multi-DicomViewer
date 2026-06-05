# Third-Party Notices

Multi-DICOMviewer is distributed under the **GNU General Public License v3.0**
(see `LICENSE`). That choice is required by **PyQt6**, which is GPL-licensed.

The distributed application bundles the following third-party components. Each
remains under its own license; this file is an informational summary, not a
substitute for the upstream license texts.

| Component | Role | License |
|-----------|------|---------|
| **PyQt6** (Riverbank) | GUI toolkit | **GPL v3** (or commercial) |
| **Qt** (via PyQt6) | underlying GUI framework | LGPL v3 |
| **VTK** | CT viewer rendering (Windows) | BSD 3-Clause |
| **pygfx / wgpu / rendercanvas** | CT viewer rendering (macOS, Metal) | BSD 2-Clause |
| **pydicom** | DICOM parsing | MIT |
| **NumPy** | array math | BSD 3-Clause |
| **pylibjpeg**, **-libjpeg**, **-openjpeg** | compressed DICOM decode | MIT / BSD / libjpeg |
| **imagecodecs** | fast cine decode (optional) | BSD 3-Clause (bundles codecs with their own permissive licenses) |
| **imageio** | MP4 export writer API | BSD 2-Clause |
| **imageio-ffmpeg** | bundled FFmpeg binary for MP4 export | FFmpeg is **LGPL/GPL**; the bundled build's terms apply |
| **PyInstaller** | build/packaging only | GPL with a bootloader exception permitting distribution of the frozen app |

## Notes

- **PyQt6 is the copyleft trigger.** Because the app links PyQt6 under the GPL,
  the combined work is distributed under GPL v3. The complete corresponding
  source is the public repository this build was produced from.
- **FFmpeg** ships as a static binary inside `imageio-ffmpeg`. Its own license
  (LGPL, or GPL depending on the build) governs that component; consult the
  upstream `imageio-ffmpeg` distribution for the exact build and its notices.
- For the authoritative, full license texts, see each project's upstream
  repository / distribution.

## Not a medical device

Multi-DICOMviewer is **research / educational software only. It is not a medical
device and is not for clinical diagnosis.** See `README.md`.
