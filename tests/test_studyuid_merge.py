"""Patient nodes that share a StudyInstanceUID must fuse into one.

A DICOM StudyInstanceUID is unique to one study of one patient, so two Patient
nodes carrying the same study UID are the same person. This happens when a
single file's PatientID / PatientName bytes are truncated by the modality (an
Iwaki XA unit wrote a 9-digit ID + a cut-off cp932 name on its first cine while
the rest of the study had the correct 10-digit ID). The merge must be loss-free
and keep the intact identity.
"""
from multi_dicomviewer.core.dicom_io import _merge_studyuid_duplicate_patients
from multi_dicomviewer.core.study_model import Modality, Patient, Series, Study


def _series(uid, *files):
    return Series(series_uid=uid, modality=Modality.XA, description="", files=list(files))


def _patient(pid, name, study_uid, series):
    st = Study(study_uid=study_uid, description="", date="20260501")
    for se in series:
        st.series[se.series_uid] = se
    return Patient(patient_id=pid, name=name, studies={study_uid: st})


def test_corrupt_header_clip_merges_into_intact_patient():
    good = _patient("0005396189", "マエダ　カンガク", "STUDY1",
                    [_series("SE2", "f2", "f3"), _series("SE3", "f4")])
    # XA000001: same study UID, truncated 9-digit ID and a mojibake name.
    bad = _patient("000539618", "マエダ　カンガ�", "STUDY1",
                   [_series("SE1", "f1")])
    patients = {bad.patient_id: bad, good.patient_id: good}  # bad scanned first

    _merge_studyuid_duplicate_patients(patients)

    assert list(patients) == ["0005396189"]          # one node, intact identity wins
    surviving = patients["0005396189"]
    assert surviving.name == "マエダ　カンガク"  # clean name kept
    series = surviving.studies["STUDY1"].series
    assert set(series) == {"SE1", "SE2", "SE3"}        # the orphan clip is preserved
    # No file dropped anywhere.
    all_files = [f for se in series.values() for f in se.files]
    assert sorted(all_files) == ["f1", "f2", "f3", "f4"]


def test_clean_name_wins_even_with_fewer_files():
    # Mojibake node has MORE files; the clean-name node must still be the
    # surviving identity (name-cleanliness outranks file count).
    clean = _patient("IDclean", "ヤマダ　タロウ", "STUDY1", [_series("SE1", "f1")])
    dirty = _patient("IDdirty", "ヤマダ　タロ�", "STUDY1",
                     [_series("SE2", "f2"), _series("SE3", "f3")])
    patients = {dirty.patient_id: dirty, clean.patient_id: clean}

    _merge_studyuid_duplicate_patients(patients)

    assert list(patients) == ["IDclean"]
    assert patients["IDclean"].name == "ヤマダ　タロウ"


def test_same_series_split_across_nodes_merges_files_without_dupes():
    a = _patient("ID1", "A", "STUDY1", [_series("SE1", "f1")])
    b = _patient("ID1longer", "A", "STUDY1", [_series("SE1", "f1", "f2")])
    patients = {a.patient_id: a, b.patient_id: b}

    _merge_studyuid_duplicate_patients(patients)

    assert len(patients) == 1
    se = next(iter(patients.values())).studies["STUDY1"].series["SE1"]
    assert sorted(se.files) == ["f1", "f2"]            # f1 not duplicated


def test_distinct_studies_are_left_separate():
    p1 = _patient("ID1", "Alice", "STUDY1", [_series("SE1", "f1")])
    p2 = _patient("ID2", "Bob", "STUDY2", [_series("SE2", "f2")])
    patients = {p1.patient_id: p1, p2.patient_id: p2}

    _merge_studyuid_duplicate_patients(patients)

    assert set(patients) == {"ID1", "ID2"}             # genuinely different patients


def test_patients_chained_through_shared_studies_all_merge():
    # P1 shares STUDY_A with P2; P2 shares STUDY_B with P3 -> all one person.
    p1 = _patient("ID1", "X", "STUDY_A", [_series("SE1", "f1")])
    p2 = Patient("ID2", "X", studies={
        "STUDY_A": Study("STUDY_A", "", "", {"SE2": _series("SE2", "f2")}),
        "STUDY_B": Study("STUDY_B", "", "", {"SE3": _series("SE3", "f3")}),
    })
    p3 = _patient("ID3", "X", "STUDY_B", [_series("SE4", "f4")])
    patients = {"ID1": p1, "ID2": p2, "ID3": p3}

    _merge_studyuid_duplicate_patients(patients)

    assert len(patients) == 1
    surviving = next(iter(patients.values()))
    assert set(surviving.studies) == {"STUDY_A", "STUDY_B"}
    assert set(surviving.studies["STUDY_A"].series) == {"SE1", "SE2"}
