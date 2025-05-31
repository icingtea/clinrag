import os
import json
from dotenv import load_dotenv
from datetime import datetime
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import Dict, Any, Optional

load_dotenv()

if not os.getenv("MONGODB_URI"):
    raise EnvironmentError("[ERROR] MONGODB_URI is not set in your .env file.")
if not os.getenv("DATABASE_NAME"):
    raise EnvironmentError("[ERROR] DATABASE_NAME is not set in your .env file.")
if not os.getenv("COLLECTION_NAME"):
    raise EnvironmentError("[ERROR] COLLECTION_NAME is not set in your .env file.")

URI: Optional[str] = os.getenv("MONGODB_URI")
DATA_PATH: str = os.path.join("preprocessing", "trial_data", "trials.jsonl")

client: MongoClient = MongoClient(URI, server_api=ServerApi("1"))
db: Database = client[os.getenv("DATABASE_NAME")]
collec: Collection = db[os.getenv("COLLECTION_NAME")]


def load_document(document: Dict[str, Any]) -> None:
    print("\n== DOCUMENT ==")
    print("-------------------------------------------------------\n")
    print(
        f"[INFO] Inserting document with NCTID {document['source_id']}, Section {document['section']}"
    )

    if not isinstance(document.get("metadata"), dict):
        print(
            f"[WARN] Skipping document {document.get('source_id')} due to malformed metadata"
        )
        return

    document["metadata"]["startDate"] = parse_date(
        document["metadata"].get("startDate")
    )
    document["metadata"]["completionDate"] = parse_date(
        document["metadata"].get("completionDate")
    )

    result = collec.insert_one(document)
    print(f"[INFO] Inserted with _id: {result.inserted_id}")


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    if len(date_str) == 7:
        date_str += "-01"
    elif len(date_str) == 4:
        date_str += "-01-01"
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("[INFO] Successfully connected to MongoDB")
        print("-------------------------------------------------------\n")
    except Exception as e:
        print("[FATAL] Something went wrong during ping:", e)

    try:
        with open(DATA_PATH, "r") as trial_data:
            for line in trial_data:
                try:
                    doc: Dict[str, Any] = json.loads(line)
                    load_document(doc)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping malformed JSON line: {e}")

            print("[INFO] ALL RECORDS INSERTED.")
    except FileNotFoundError:
        print("[ERROR] Output file path not found.")
    except Exception as e:
        print(f"[FATAL] Unexpected error during data fetch: {e}")
