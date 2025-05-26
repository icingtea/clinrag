from typing import Annotated, List, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import TypedDict, Dict, Any
from datetime import datetime
from enum import Enum

class Status(Enum):
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    AVAILABLE = "AVAILABLE"
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"
    WITHHELD = "WITHHELD"
    UNKNOWN = "UNKNOWN"

class StudyType(Enum):
    EXPANDED_ACCESS = "EXPANDED_ACCESS"
    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"

class DesignAllocation(Enum):
    RANDOMIZED = "RANDOMIZED"
    NON_RANDOMIZED = "NON_RANDOMIZED"
    NA = "N/A"

class InterventionalAssignment(Enum):
    SINGLE_GROUP = "SINGLE_GROUP"
    PARALLEL = "PARALLEL"
    CROSSOVER = "CROSSOVER"
    FACTORIAL = "FACTORIAL"
    SEQUENTIAL = "SEQUENTIAL"

class MaskingType(Enum):
    NONE = "NONE"
    SINGLE = "SINGLE"
    DOUBLE = "DOUBLE"
    TRIPLE = "TRIPLE"
    QUADRUPLE = "QUADRUPLE"

class Sex(Enum):
    FEMALE = "FEMALE"
    MALE = "MALE"
    ALL = "ALL"

class StdAges(Enum):
    CHILD = "CHILD"
    ADULT = "ADULT"
    OLDER_ADULT = "OLDER_ADULT"

class PromptMetadata(BaseModel):
    nctId: List[str] = Field(default_factory=list)
    status: List[Status] = Field(default_factory=list)
    startDateBefore: Optional[datetime]
    startDateAfter: Optional[datetime]
    completionDateBefore: Optional[datetime]
    completionDateAfter: Optional[datetime]
    studyType: List[StudyType] = Field(default_factory=list)
    allocation: List[DesignAllocation] = Field(default_factory=list)
    interventionModel: List[InterventionalAssignment] = Field(default_factory=list)
    maskingType: List[MaskingType] = Field(default_factory=list)
    healthyVolunteers: List[bool] = Field(default_factory=list)
    sex: List[Sex] = Field(default_factory=list)
    stdAges: List[StdAges] = Field(default_factory=list)

class GraphState(BaseModel):
    question: str
    metadata: Dict[str, Any]
    filter: Dict[str, Any]
    context: List[str]
    memory: Annotated[List, add_messages]