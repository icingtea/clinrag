from typing import List, Optional
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class ChunkType(str, Enum):
    OVERVIEW = "overview"
    DESIGN = "design"
    ELIGIBILITY = "eligibility"
    CONDITIONS = "conditions"
    ARMS_INTERVENTIONS = "armsInterventions"
    OUTCOMES_PRIMARY = "primaryOutcomes"
    OUTCOMES_SECONDARY = "secondaryOutcomes"

class TrialMetaData(BaseModel):
    nctId: str
    status: Optional[str] = None
    startDate: Optional[str] = None
    completionDate: Optional[str] = None
    studyType: Optional[str] = None
    allocation: Optional[str] = None
    interventionModel: Optional[str] = None
    maskingType: Optional[str] = None
    enrollmentCount: Optional[int] = None
    healthyVolunteers: Optional[bool] = None
    sex: Optional[str] = None
    minimumAge: Optional[str] = None
    maximumAge: Optional[str] = None
    stdAges: Optional[List[str]] = None

class Chunk(BaseModel):
    source_id: str
    metadata: TrialMetaData
    section: ChunkType
    text: str
    embeddings: List[float]
