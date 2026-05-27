"""Display-only anonymization policy.

Multi-DICOMviewer never rewrites DICOM files. This module only decides *which*
tags count as case-identifying (PHI) and what placeholder text replaces
their value when the global Anonymize toggle is on. Every place that
shows DICOM-derived text on screen (study tree, viewer titles, tag list
dialog, image overlay) routes patient/case fields through here so a single
toggle blanks them everywhere consistently.
"""
from __future__ import annotations

# Shown in place of a masked value. Kept short so it fits a corner overlay.
ANON_PLACEHOLDER = "(anonymized)"

# DICOM keywords treated as case-identifying. This is the
# patient/visit/operator-identifying subset of the DICOM basic
# confidentiality profile — enough to de-identify the screen without
# hiding clinically relevant acquisition parameters.
PHI_KEYWORDS: frozenset[str] = frozenset(
    {
        # Patient identity
        "PatientName",
        "PatientID",
        "IssuerOfPatientID",
        "OtherPatientIDs",
        "OtherPatientIDsSequence",
        "OtherPatientNames",
        "PatientBirthDate",
        "PatientBirthTime",
        "PatientBirthName",
        "PatientMotherBirthName",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "PatientTelecomInformation",
        "CountryOfResidence",
        "RegionOfResidence",
        "MilitaryRank",
        "BranchOfService",
        "EthnicGroup",
        "PatientReligiousPreference",
        "PatientComments",
        "ResponsiblePerson",
        "ResponsibleOrganization",
        "MedicalRecordLocator",
        "InsurancePlanIdentification",
        # Visit / study identifiers and dates
        "AccessionNumber",
        "StudyID",
        "StudyDate",
        "StudyTime",
        "SeriesDate",
        "SeriesTime",
        "AcquisitionDate",
        "AcquisitionTime",
        "AcquisitionDateTime",
        "ContentDate",
        "ContentTime",
        "InstanceCreationDate",
        "InstanceCreationTime",
        "AdmissionID",
        "ScheduledProcedureStepID",
        "RequestedProcedureID",
        "PerformedProcedureStepID",
        # People / facility
        "ReferringPhysicianName",
        "ReferringPhysicianTelephoneNumbers",
        "ReferringPhysicianAddress",
        "PerformingPhysicianName",
        "NameOfPhysiciansReadingStudy",
        "OperatorsName",
        "PhysiciansOfRecord",
        "RequestingPhysician",
        "RequestingService",
        "InstitutionName",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "StationName",
        "DeviceSerialNumber",
    }
)


def is_phi(keyword: str) -> bool:
    """True if *keyword* (a pydicom tag keyword) is case-identifying."""
    return keyword in PHI_KEYWORDS


def mask_text(keyword: str, value: str, anonymized: bool) -> str:
    """Return *value*, or the placeholder when anonymizing a PHI field.

    *value* is the already-stringified display value; non-PHI fields and
    the un-anonymized state pass straight through so callers can use this
    unconditionally.
    """
    if anonymized and is_phi(keyword):
        return ANON_PLACEHOLDER
    return value
