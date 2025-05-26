import requests
import json
import os
from chunking_utils import parse_data, create_chunks
from typing import List
from schemas import TrialMetaData

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DATA_PATH = os.path.join("preprocessing", "trial_data", "trials.jsonl")

def get_NCT_ids(page_size=1000) -> List[str]:
    study_ids = []

    print("\n=== Fetching NCT IDs ===")
    print("-------------------------------------------------------\n")

    try:
        response = requests.get(f"{BASE_URL}?pageSize={page_size}")
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch NCT IDs: {e}")
        return []

    print(f"[INFO] Request status: {response.status_code}")

    if response.status_code != 200:
        print("[WARNING] Non-200 response; possible issue fetching data.")
        return study_ids

    studies = response.json().get("studies", [])
    for study in studies:
        nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        if nct_id:
            study_ids.append(nct_id)

    print(f"[INFO] Retrieved {len(study_ids)} NCT IDs.")
    return study_ids

def get_full_studies(study_ids: List[str]) -> None:
    print("\n=== Fetching Full Study Data ===")
    print("-------------------------------------------------------\n")

    counter = 1

    try:
        with open(DATA_PATH, "w") as outfile:
            for nct_id in study_ids:
                response = requests.get(f"{BASE_URL}/{nct_id}")
                print(f"[{counter}] Fetching {nct_id}... Status: {response.status_code}")

                if response.status_code != 200:
                    print(f"[ERROR] Skipping {nct_id} due to bad status code: {response.status_code}")
                    continue

                try:
                    full_study_data = response.json()
                    study_metadata: TrialMetaData = parse_data(full_study_data)

                    for chunk in create_chunks(full_study_data, study_metadata):
                        json.dump(chunk.model_dump(), outfile)
                        outfile.write("\n")

                except Exception as parse_err:
                    print(f"[ERROR] Failed to process {nct_id}: {parse_err}")
                    break

                counter += 1
                print("-------------------------------------------------------\n")
    except FileNotFoundError:
        print("[ERROR] Output file path not found.")
    except Exception as e:
        print(f"[FATAL] Unexpected error during study fetch: {e}")

if __name__ == "__main__":
    ids = get_NCT_ids()
    get_full_studies(ids)
