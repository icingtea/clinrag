import os
import textwrap
import openai
import torch
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnableConfig

from state_schema import (
    PromptMetadata,
    Status,
    StudyType,
    DesignAllocation,
    InterventionalAssignment,
    MaskingType,
    Sex,
    StdAges,
    State
)
def query_metadata_extraction(state: State, config: RunnableConfig) -> Dict[str, Any]:
    client: openai.OpenAI = state["openai_client"]

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

    user_prompt = f"User question: {state['question']}\nReturn only valid JSON."

    try:
        response = client.responses.parse(
            model="chatgpt-4o-latest",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            text_format=PromptMetadata
        )
        parsed = response.output_parsed
        print(parsed)
        return {"metadata": parsed.model_dump(), "error": None}
    except Exception as e:
        print(e)
        return {"metadata": {}, "error": "[ERROR] Metadata extraction failed."}
    
def db_filter_assembly(state: State) -> Dict[str, Any]:
    filter_dict: Dict[str, Any] = {}

    for field, value in state["metadata"].items():

        if isinstance(value, list) and value:
            if value is None or (isinstance(value, list) and not value):
                continue

            field = f"metadata.{field}"
            flat_values = [v.value if isinstance(v, Enum) else v for v in value]
            if len(flat_values) == 1:
                filter_dict[field] = flat_values[0]
            else:
                filter_dict[field] = {"$in": flat_values}

        elif isinstance(value, datetime):
            if field == "startDateBefore":
                filter_dict.setdefault("metadata.startDate", {})["$lt"] = value
            elif field == "startDateAfter":
                filter_dict.setdefault("metadata.startDate", {})["$gt"] = value
            elif field == "completionDateBefore":
                filter_dict.setdefault("metadata.completionDate", {})["$lt"] = value
            elif field == "completionDateAfter":
                filter_dict.setdefault("metadata.completionDate", {})["$gt"] = value

    print(filter_dict)
    return {"filter": filter_dict}

def vector_search(state: State) -> Dict[str, Any]:
    model: SentenceTransformer = state["embedding_model"]
    index_name: str = state["index_name"]
    collection: Collection = state["mongo_collection"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        prompt_embeddings = model.encode([state["question"]], device=device).tolist()[0]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embeddings",
                    "queryVector": prompt_embeddings,
                    "exact": True,
                    "limit": 15,
                    "filter": state["filter"]
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
 
        results = collection.aggregate(pipeline)
        context_docs = [doc["text"] for doc in results if float(doc["score"]) > 0.7]

        print(context_docs)

        return {"context": context_docs, "error": None}

    except Exception as e:
        print(e)
        return {"context": [], "error": "[ERROR] Vector search failed."}

def chat_response(state: State) -> Dict[str, Any]:
    client: openai.OpenAI = state["openai_client"]
    prompt = state["question"]
    context = state["context"]
    memory = state["memory"]

    system_prompt = """
    Answer the user question based off of the context provided to you. Make sure you stay factual, and do not respond with info that could not be inferred from the given context.
    Respond along the lines of 'I'm sorry, I don't know', or 'I'm sorry, I don't have that information provided to me' if the context isn't enough for you to correctly answer.
    """
    try:
        response = client.responses.parse(
            model="chatgpt-4o-latest",
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"MESSAGE HISTORY: {memory}\nCONTEXT: {context}\nUSER PROMPT: {prompt}"
                }
            ]
        )
        return {"response": response.output_text, "error": None}

    except Exception as e:
        print(e)
        return {"response": None, "error": "[ERROR] Failed to get chat response."}

def error_response(state: State) -> Dict[str, Any]:
    return {"response": state["error"]}

def error_check(state: State) -> bool:
    if state["error"]:
        return True
    else:
        return False
