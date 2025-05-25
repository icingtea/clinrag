import os
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from graph_state import PromptMetadata, GraphState, Status, StudyType, DesignAllocation, InterventionalAssignment, MaskingType, Sex, StdAges
from openai import OpenAI

load_dotenv()
client = OpenAI()

def query_metadata_extraction(state: Dict[str, Any]) -> Dict[str, Any]:
    print("== METADATA EXTRACTION ==")

    question = state["question"]

    system_prompt = f"""
    You are a metadata extractor for clinical trial queries. Parse the user's question and return a JSON with the following fields, with strict adherence:
    - nctId: list of strings
    - status: list of enum values from: {', '.join([e.name for e in Status])}
    - startDate: list of ISO 8601 dates (e.g. "2021-05-01")
    - completionDate: list of ISO 8601 dates
    - studyType: list of enum values from: {', '.join([e.name for e in StudyType])}
    - allocation: list of enum values from: {', '.join([e.name for e in DesignAllocation])}
    - interventionModel: list of enum values from: {', '.join([e.name for e in InterventionalAssignment])}
    - maskingType: list of enum values from: {', '.join([e.name for e in MaskingType])}
    - enrollmentCount: list of integers
    - healthyVolunteers: list of booleans
    - sex: list of enum values from: {', '.join([e.name for e in Sex])}
    - minimumAge: list of strings (e.g., "18 Years")
    - maximumAge: list of strings
    - stdAges: list of enum values from: {', '.join([e.name for e in StdAges])}, infer this from any information you might get regarding minimum and maximum ages, or if explicity asked for certain standard age categories

    Return an empty list for any field where no info is found. Adhere strictly to enum values.
    """
    user_prompt = f"User question: {question}\nReturn only valid JSON."

    response = client.responses.parse(
        model="chatgpt-4o-latest",
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format= PromptMetadata
    )

    print(response.output_parsed)