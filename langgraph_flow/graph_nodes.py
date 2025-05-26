import openai
import textwrap
from enum import Enum
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any
from state_schema import PromptMetadata, GraphState, Status, StudyType, DesignAllocation, InterventionalAssignment, MaskingType, Sex, StdAges

load_dotenv()

def query_metadata_extraction(client: openai.OpenAI, state: GraphState) -> PromptMetadata:
    question: str = state["question"]
    system_prompt: str = textwrap.dedent(f"""
    You are a metadata extractor for clinical trial queries. Parse the user's question and return a JSON with the following fields:

    - nctId: list of strings
    - status: list of enum values from: {', '.join([e.name for e in Status])}
    - startDateBefore: ISO 8601 date (e.g., "2021-05-01") or null
    - startDateAfter: ISO 8601 date or null
    - completionDateBefore: ISO 8601 date or null
    - completionDateAfter: ISO 8601 date or null
    - studyType: list of enum values from: {', '.join([e.name for e in StudyType])}
    - allocation: list of enum values from: {', '.join([e.name for e in DesignAllocation])}
    - interventionModel: list of enum values from: {', '.join([e.name for e in InterventionalAssignment])}
    - maskingType: list of enum values from: {', '.join([e.name for e in MaskingType])}
    - healthyVolunteers: list of booleans
    - sex: list of enum values from: {', '.join([e.name for e in Sex])}
    - stdAges: list of enum values from: {', '.join([e.name for e in StdAges])}, infer this from any age-related info or explicitly mentioned categories.

    If no data is found for a field, use `null` for dates and empty lists for others. Return only valid JSON.
    """)

    user_prompt: str = f"User question: {question}\nReturn only valid JSON."

    response = client.responses.parse(
        model="chatgpt-4o-latest",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format= PromptMetadata
    )

    state["metadata"] = response.output_parsed.model_dump()
    return response.output_parsed
    
def db_filter_assembly(state: GraphState) -> Dict[str, Any]:
    metadata = state["metadata"]
    filter_dict: Dict[str, Any] = {}

    for field, value in metadata.items():
        if isinstance(value, list) and value:
            if isinstance(value[0], Enum):
                filter_dict[field] = {"$in": [v.value for v in value]}
            else:
                filter_dict[field] = {"$in": value}

        elif isinstance(value, datetime) and value:
            if field == "startDateBefore":
                filter_dict.setdefault("startDate", {})["$lt"] = value
            elif field == "startDateAfter":
                filter_dict.setdefault("startDate", {})["$gt"] = value
            elif field == "completionDateBefore":
                filter_dict.setdefault("completionDate", {})["$lt"] = value
            elif field == "completionDateAfter":
                filter_dict.setdefault("completionDate", {})["$gt"] = value

    return filter_dict