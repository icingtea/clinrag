import os
import textwrap
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import openai
import torch
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from state_schema import (
    PromptMetadata,
    GraphState,
    Status,
    StudyType,
    DesignAllocation,
    InterventionalAssignment,
    MaskingType,
    Sex,
    StdAges,
)

def query_metadata_extraction(client: openai.OpenAI, state: GraphState) -> PromptMetadata:
    question = state["question"]

    system_prompt = textwrap.dedent(f"""
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

    user_prompt = f"User question: {question}\nReturn only valid JSON."

    response = client.responses.parse(
        model="chatgpt-4o-latest",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format=PromptMetadata
    )

    parsed: PromptMetadata = response.output_parsed
    state["metadata"] = parsed.model_dump()

    return parsed


def db_filter_assembly(state: GraphState) -> Dict[str, Any]:
    metadata = state["metadata"]
    filter_dict: Dict[str, Any] = {}

    for field, value in metadata.items():
        if isinstance(value, list) and value:
            if all(isinstance(v, Enum) for v in value):
                filter_dict[field] = {"$in": [v.value for v in value]}
            else:
                filter_dict[field] = {"$in": value}

        elif isinstance(value, datetime):
            if field == "startDateBefore":
                filter_dict.setdefault("startDate", {})["$lt"] = value
            elif field == "startDateAfter":
                filter_dict.setdefault("startDate", {})["$gt"] = value
            elif field == "completionDateBefore":
                filter_dict.setdefault("completionDate", {})["$lt"] = value
            elif field == "completionDateAfter":
                filter_dict.setdefault("completionDate", {})["$gt"] = value

    state["filter"] = filter_dict
    return filter_dict

def vector_search( state: GraphState, model: SentenceTransformer, index_name: str, collection: Collection ) -> List[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = state["question"]
    filter = state["filter"]

    prompt_embeddings = model.encode([text], device=device).tolist()

    pipeline = [
        {
            "$vector_search": {
                "index": index_name,
                "path": "embeddings",
                "queryVector": prompt_embeddings,
                "exact": True,
                "limit": 15,
                "filter": filter,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]

    results = collection.aggregate(pipeline)
    context_docs = [doc["text"] for doc in results if float(doc["score"]) > 0.7]
    state["context"] = "\n\n".join(context_docs)

    return context_docs
