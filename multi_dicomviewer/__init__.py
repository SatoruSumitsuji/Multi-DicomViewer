"""Multi-DICOMviewer — research DICOM viewer for multiple modalities.

Handles XA angiography and cardiac CT today, with IVUS, OCT/OFDI, NM and
MRI on the roadmap. Architecture: a common shell (DICOM core + study
browser + linked layout) with pluggable modality viewer modules sharing
one study model so a patient's studies can be correlated side by side.
"""

__version__ = "2.0.5"
