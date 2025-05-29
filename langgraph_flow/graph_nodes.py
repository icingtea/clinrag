import os
import textwrap
import openai
import torch
import logging
from dotenv import load_dotenv
from datetime import datetime
from enum import Enum
from typing import Any, Dict
from pymongo import MongoClient
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
torch.classes.__path__ = []

logging.basicConfig(filename="session.log", filemode="w")
logger = logging.getLogger("applog")
logger.setLevel(logging.DEBUG)

MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_SEARCH_INDEX = os.getenv("VECTOR_SEARCH_INDEX")

if not all([MONGODB_URI, OPENAI_API_KEY, DATABASE_NAME, COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SEARCH_INDEX]):
    raise ValueError("[ERROR] Missing one or more required environment variables")

mongo_client = MongoClient(MONGODB_URI)
mongo_collection = mongo_client[DATABASE_NAME][COLLECTION_NAME]
openai_client = openai.OpenAI()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
index_name = VECTOR_SEARCH_INDEX

from langgraph_flow.state_schema import (
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

def query_metadata_extraction(state: State) -> Dict[str, Any]:
    client: openai.OpenAI = openai_client

    system_prompt = textwrap.dedent(f"""
    You are a metadata extractor for clinical trial queries. Parse the user's question and return a JSON with the following fields. 
    You may be provided with some conversation context. If the user's question seems like a followup, collect metadata by inferring accordingly.
    IT IS OF UTMOST IMPORTANCE TO INCLUDE METADATA ASKED FOR IN THE CURRENT USER QUESTION THAT CAN BE INFERRED FROM YOUR CONTEXT, TO MAKE SURE THERE IS LONGEVITY OF CONVERSATION AND THINGS TALKED ABOUT EARLIER ARE REFERENCEABLE.
    DO NOT HALLUCINATE INFO. IF THE USER REFERS TO A PART OF CONVERSATION YOU CANNOT SEE OR PINPOINT, SAY YOU DON'T REMEMBER AND ASK FOR THEM TO ELABORATE ON WHAT EXACTLY THEY'RE ASKING FOR.
    DO NOT MAKE BLANKET STATEMENTS ABOUT WHAT IS CONTAINED IN THE DATASET. YOU DO NOT KNOW WHAT'S IN IT.
                                    
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
    THANKS :D
    """)

    recent_context = "\n".join([f"ROLE: {message.type.upper()} MESSAGE: {message.content}" for message in state["memory"][-6:]]),
    user_prompt = f"Context: {recent_context}\nUser question: {state['question']}\nReturn only valid JSON."

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

        state_change = {"metadata": parsed.model_dump(), 
                        "recent_context": "\n".join([f"ROLE: {message.type.upper()} MESSAGE: {message.content}" for message in state["memory"][-4:]]), 
                        "error": None}
        logger.info(f"[METADATA NODE] {state_change}")
        return state_change
    
    except Exception as e:
        state_change = {"metadata": {}, "error": "[ERROR] Metadata extraction failed."}
        logger.debug(f"[METADATA NODE][ERROR] {e}")
        return state_change
    
def db_filter_assembly(state: State) -> Dict[str, Any]:
    filter_dict: Dict[str, Any] = {}

    for field, value in state["metadata"].items():

        if isinstance(value, list) and value:
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

    state_change = {"filter": filter_dict}
    logger.info(f"[FILTERING NODE] {state_change}")
    return state_change

def vector_search(state: State) -> Dict[str, Any]:
    model: SentenceTransformer = embedding_model
    search_index: str = index_name
    collection: Collection = mongo_collection

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        prompt_embeddings = model.encode([state["question"]], device=device).tolist()[0]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": search_index,
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
        context_docs = [doc["text"] for doc in results if float(doc["score"]) > 0.5]

        state_change = {"context": context_docs, "error": None}
        logger.info(f"[VECTOR SEARCH NODE] {state_change}")
        return state_change

    except Exception as e:
        state_change = {"context": [], "error": "[ERROR] Vector search failed."}
        logger.debug(f"[VECTOR SEARCH NODE][ERROR] {e}")
        return state_change

def chat_response(state: State) -> Dict[str, Any]:
    client: openai.OpenAI = openai_client
    prompt = state["question"]
    context = state["context"]
    memory = state["memory"]

    system_prompt = textwrap.dedent("""
        You are an expert assistant answering questions about clinical trials. Use the provided context to answer the user's question accurately and precisely. You may infer reasonable conclusions if they logically follow from the context, but do not introduce information not supported by the documents. If a clear answer is not possible, say something like:

        > “I'm sorry, I don't have enough information to answer that.”

        Do not mention the existence of "context" or "filters" in your response.

        The trials you see have already been filtered based on details inferred from the user's query. For example, if a user asks about trials that started before a certain date, only trials matching that condition will be shown to you. The same applies to fields like:

        - `nctId`  
        - `status`  
        - `startDateBefore` / `startDateAfter`  
        - `completionDateBefore` / `completionDateAfter`  
        - `studyType`  
        - `allocation`  
        - `interventionModel`  
        - `maskingType`  
        - `healthyVolunteers`  
        - `sex`  
        - `stdAges`  

        Therefore, when questions relate to these properties, you can assume the trials shown already meet the implied criteria.

        Be factual, concise, and avoid speculation. Always include trial numbers (e.g., NCT IDs) when referencing studies.
    """)

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
        
        state_change = {"response": response.output_text,
                        "memory": [HumanMessage(content = prompt), AIMessage(content = response.output_text)],
                        "error": None}
        logger.info(f"[CHAT RESPONSE NODE] {state_change}")
        return state_change

    except Exception as e:
        state_change = {"response": None, "error": "[ERROR] Failed to get chat response."}
        logger.debug(f"[CHAT RESPONSE NODE][ERROR] {e}")
        return state_change

def error_response(state: State) -> Dict[str, Any]:
    state_change = {"response": state["error"]}
    logger.info(f"[ERROR RESPONSE NODE] {state_change}")
    return state_change

def error_check(state: State) -> bool:
    if state["error"]:
        return True
    else:
        return False
