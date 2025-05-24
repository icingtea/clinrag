from typing import List, Optional, Any
from pydantic import BaseModel
from enum import Enum

class ChunkType(str, Enum):
    OVERVIEW = "overview"
    DESIGN = "design"
    ELIGIBILITY = "eligibility"
    CONDITIONS = "conditions"
    ARMS_INTERVENTIONS = "armsInterventions"
    OUTCOMES_PRIMARY = "primaryOutcomes"
    OUTCOMES_SECONDARY = "secondaryOutcomes"

class DesignMetaData(BaseModel):
    studyType: Optional[str] = None
    phases: Optional[List[str]] = None
    allocation: Optional[str] = None
    interventionModel: Optional[str] = None
    primaryPurpose: Optional[str] = None
    maskingType: Optional[str] = None
    whoMasked: Optional[List[str]] = None
    enrollmentCount: Optional[int] = None

class EligibilityMetaData(BaseModel):
    healthyVolunteers: Optional[bool] = None
    sex: Optional[str] = None
    minimumAge: Optional[str] = None
    maximumAge: Optional[str] = None
    stdAges: Optional[List[str]] = None

class TrialMetaData(BaseModel):
    nctId: str
    acronym: Optional[str] = None
    status: Optional[str] = None
    statusVerifiedDate: Optional[str] = None
    whyStopped: Optional[str] = None
    startDate: Optional[str] = None
    completionDate: Optional[str] = None
    designInfo: Optional[DesignMetaData] = None
    eligibility: Optional[EligibilityMetaData] = None

class Chunk(BaseModel):
    source_id: str
    metadata: TrialMetaData
    section: ChunkType
    text: str
    embeddings: List[float]
