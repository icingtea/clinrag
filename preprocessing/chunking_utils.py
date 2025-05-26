import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from preprocessing.schemas import Chunk, TrialMetaData, ChunkType
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("intfloat/e5-large-v2")

def unpack_protocol_sections(study: Dict[str, Any]) -> Tuple[
    Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any],
    Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
]:
    protocol: Dict[str, Any] = study.get("protocolSection", {})

    return (
        protocol,
        protocol.get("designModule", {}),
        protocol.get("designModule", {}).get("designInfo", {}),
        protocol.get("designModule", {}).get("enrollmentInfo", {}),
        protocol.get("designModule", {}).get("designInfo", {}).get("maskingInfo", {}),
        protocol.get("eligibilityModule", {}),
        protocol.get("identificationModule", {}),
        protocol.get("statusModule", {}),
        protocol.get("conditionsModule", {}),
        protocol.get("armsInterventionsModule", {})
    )

def parse_data(full_study_data: Dict[str, Any]) -> TrialMetaData:
    (
        protocol,
        design_module,
        design_info,
        enrollment_info,
        masking_info,
        eligibility_module,
        identification_module,
        status_module,
        conditions_module,
        arms_interventions_module
    ) = unpack_protocol_sections(full_study_data)
    
    trial_metadata: TrialMetaData = TrialMetaData(
        nctId=identification_module.get("nctId", None),
        status=status_module.get("overallStatus", None),
        startDate=status_module.get("startDateStruct", {}).get("date", None),
        completionDate=status_module.get("completionDateStruct", {}).get("date", None),
        studyType=design_module.get("studyType", None),
        allocation=design_info.get("allocation", None),
        interventionModel=design_info.get("interventionModel", None),
        maskingType=masking_info.get("masking", None),
        enrollmentCount=enrollment_info.get("count", None),
        healthyVolunteers=eligibility_module.get("healthyVolunteers", None),
        sex=eligibility_module.get("sex", None),
        minimumAge=eligibility_module.get("minimumAge", None),
        maximumAge=eligibility_module.get("maximumAge", None),
        stdAges=eligibility_module.get("stdAges", None)
    )

    return trial_metadata

def create_chunks(full_study_data: Dict[str, Any], study_metadata: TrialMetaData) -> List[Chunk]:
    (
        protocol,
        design_module,
        design_info,
        enrollment_info,
        masking_info,
        eligibility_module,
        identification_module,
        status_module,
        conditions_module,
        arms_interventions_module
    ) = unpack_protocol_sections(full_study_data)

    nct_id: str = identification_module.get("nctId")

    overview_chunk_text: str = "\n".join([
        f"Study Overview ({nct_id})",
        f"Study Title: {identification_module.get('officialTitle', 'No info available')}",
        f"Overview Title: {identification_module.get('briefTitle', 'No info available')}",
        f"Description: {protocol.get('descriptionModule', {}).get('briefSummary', 'No info available')}",
        f"Status: {status_module.get('overallStatus', 'No info available')} (Verified: {status_module.get('statusVerifiedDate', 'No info available')})",
        f"Start Date: {status_module.get('startDateStruct', {}).get('date', 'No info available')}",
        f"Completion Date: {status_module.get('completionDateStruct', {}).get('date', 'No info available')}",
        f"Why Stopped (if applicable): {status_module.get('whyStopped', 'N/A')}"
    ])

    overview_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.OVERVIEW,
        text=overview_chunk_text,
        embeddings=embed_text(overview_chunk_text, ChunkType.OVERVIEW)
    )

    phases: List[str] = design_module.get("phases", ["No info available"])
    masked_entities: List[str] = masking_info.get("whoMasked", ["No info available"])

    design_chunk_text: str = "\n".join([
        f"Study Design Details ({nct_id})",
        f"Study Type: {design_module.get('studyType', 'No info available')}",
        f"Study Phases: {', '.join(phases)}",
        f"Study Design Info: {design_info.get('allocation', 'No info available')} allocation with {design_info.get('interventionModel', 'No info available')} intervention model. Primary purpose: {design_info.get('primaryPurpose', 'No info available')}",
        f"Masking Info: {masking_info.get('masking', 'No info available')} masking; Masked entities: {', '.join(masked_entities)}",
        f"Enrollment: {enrollment_info.get('count', 'No info available')} enrolled (type: {enrollment_info.get('type', 'No info available')})"
    ])

    design_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.DESIGN,
        text=design_chunk_text,
        embeddings=embed_text(design_chunk_text, ChunkType.DESIGN)
    )

    std_ages: List[str] = eligibility_module.get("stdAges", ["No info available"])

    eligibility_chunk_text = "\n".join([
        f"Eligibility Criteria ({nct_id})",
        f"Sex: {eligibility_module.get('sex', 'No info available')}",
        f"Minimum Age: {eligibility_module.get('minimumAge', 'No info available')}",
        f"Maximum Age: {eligibility_module.get('maximumAge', 'No info available')}",
        f"Healthy Volunteers: {eligibility_module.get('healthyVolunteers', 'No info available')}",
        f"Standard Ages: {', '.join(std_ages)}"
    ])

    eligibility_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.ELIGIBILITY,
        text=eligibility_chunk_text,
        embeddings=embed_text(eligibility_chunk_text, ChunkType.ELIGIBILITY)
    )

    conditions: List[str] = conditions_module.get("conditions", ["No info available"])
    keywords: List[str] = conditions_module.get("keywords", ["No info available"])

    conditions_chunk_text = "\n".join([
        f"Condition Details ({nct_id})",
        f"Study-Related Conditions: {', '.join(conditions)}",
        f"Study Keywords: {', '.join(keywords)}"
    ])

    conditions_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.CONDITIONS,
        text=conditions_chunk_text,
        embeddings=embed_text(conditions_chunk_text, ChunkType.CONDITIONS)
    )

    arms: List[Dict[str, Any]] = arms_interventions_module.get("armGroups", []) or []
    interventions: List[Dict[str, Any]] = arms_interventions_module.get("interventions", []) or []

    arms_text: str = "\n".join([
        f"- Arm: {arm.get('label', 'No info available')}, Type: {arm.get('type', 'No info available')}" for arm in arms
    ])

    interventions_text: str = "\n".join([
        f"- Intervention: {interv.get('name', 'No info available')} ({interv.get('type', 'No info available')}): {interv.get('description', 'No info available')}" for interv in interventions
    ])

    arms_interventions_chunk_text = "\n".join([
            f"Arms and Interventions ({nct_id})",
            "Study Arms:",
            arms_text if arms_text else "No arm group info available.",
            "",
            "Interventions:",
            interventions_text if interventions_text else "No intervention info available."
        ])

    arms_interventions_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.ARMS_INTERVENTIONS,
        text=arms_interventions_chunk_text,
        embeddings=embed_text(arms_interventions_chunk_text, ChunkType.ARMS_INTERVENTIONS)
    )
    
    primary_outcomes: List[Dict[str, Any]] = protocol.get("outcomesModule", {}).get("primaryOutcomes", [])
    
    if primary_outcomes:
        primary_text = "\n".join([
            "\n".join([
                f"PRIMARY OUTCOME {i + 1}",
                f"\tMeasure: {outcome.get('measure', 'No measure available')}",
                f"\tDescription: {outcome.get('description', 'No description available')}",
                f"\tTime Frame: {outcome.get('timeFrame', 'No time frame available')}"
            ])
            for i, outcome in enumerate(primary_outcomes)
        ])
    else:
        primary_text = "No primary outcomes info available."

    outcomes_primary_chunk_text = "\n".join([
        f"Primary Outcome Info ({nct_id}):",
        primary_text
    ])

    outcomes_primary_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.OUTCOMES_PRIMARY,
        text=outcomes_primary_chunk_text,
        embeddings=embed_text(outcomes_primary_chunk_text, ChunkType.OUTCOMES_PRIMARY)
    )

    secondary_outcomes: List[Dict[str, Any]] = protocol.get("outcomesModule", {}).get("secondaryOutcomes", [])

    if secondary_outcomes:
        secondary_text = "\n".join([
            "\n".join([
                f"SECONDARY OUTCOME {i + 1}",
                f"\tMeasure: {outcome.get('measure', 'No measure available')}",
                f"\tDescription: {outcome.get('description', 'No description available')}",
                f"\tTime Frame: {outcome.get('timeFrame', 'No time frame available')}"
            ])
            for i, outcome in enumerate(secondary_outcomes)
        ])
    else:
        secondary_text = "No secondary outcomes info available."

    outcomes_secondary_chunk_text = "\n".join([
        f"Secondary Outcome Info ({nct_id}):",
        secondary_text
    ])

    outcomes_secondary_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.OUTCOMES_SECONDARY,
        text=outcomes_secondary_chunk_text,
        embeddings=embed_text(outcomes_secondary_chunk_text, ChunkType.OUTCOMES_SECONDARY)
    )

    return [
        overview_chunk, 
        design_chunk, 
        eligibility_chunk, 
        conditions_chunk, 
        arms_interventions_chunk, 
        outcomes_primary_chunk, 
        outcomes_secondary_chunk
    ]

def embed_text(text: str, chunk_type: ChunkType) -> List[float]:
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\t[embedding:{chunk_type.value}] using device: {device}")
    embeddings: np.ndarray = MODEL.encode([text], device=device)
    print(f"\t[embedding:{chunk_type.value}] size: {embeddings.shape}\n")

    return embeddings.tolist()[0]