import os
import torch
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
from schemas import Chunk, TrialMetaData, ChunkType

load_dotenv()
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))


def embed_text(text: str, chunk_type: ChunkType) -> List[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\t[embedding:{chunk_type.value}] using device: {device}")
    embeddings = model.encode([text], device=device)
    print(f"\t[embedding:{chunk_type.value}] size: {embeddings.shape}\n")
    return embeddings[0].tolist()


def unpack_protocol_sections(study: dict) -> tuple:
    protocol = study.get("protocolSection", {})
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
        protocol.get("armsInterventionsModule", {}),
    )


def parse_data(full_study_data: dict) -> TrialMetaData:
    (
        _,
        design_module,
        design_info,
        enrollment_info,
        masking_info,
        eligibility_module,
        identification_module,
        status_module,
        _,
        _,
    ) = unpack_protocol_sections(full_study_data)

    return TrialMetaData(
        nctId=identification_module.get("nctId"),
        status=status_module.get("overallStatus"),
        startDate=status_module.get("startDateStruct", {}).get("date"),
        completionDate=status_module.get("completionDateStruct", {}).get("date"),
        studyType=design_module.get("studyType"),
        allocation=design_info.get("allocation"),
        interventionModel=design_info.get("interventionModel"),
        maskingType=masking_info.get("masking"),
        enrollmentCount=enrollment_info.get("count"),
        healthyVolunteers=eligibility_module.get("healthyVolunteers"),
        sex=eligibility_module.get("sex"),
        minimumAge=eligibility_module.get("minimumAge"),
        maximumAge=eligibility_module.get("maximumAge"),
        stdAges=eligibility_module.get("stdAges"),
    )


def create_chunks(full_study_data: dict, study_metadata: TrialMetaData) -> List[Chunk]:
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
        arms_interventions_module,
    ) = unpack_protocol_sections(full_study_data)

    nct_id = identification_module.get("nctId", "Unknown")
    url = f"https://clinicaltrials.gov/study/{nct_id}"

    # OVERVIEW
    overview_chunk_text = "\n".join(
        [
            f"Study Overview ({nct_id})",
            f"Study Title: {identification_module.get('officialTitle', 'No info available')}",
            f"Overview Title: {identification_module.get('briefTitle', 'No info available')}",
            f"Description: {protocol.get('descriptionModule', {}).get('briefSummary', 'No info available')}",
            f"Status: {status_module.get('overallStatus', 'No info available')} (Verified: {status_module.get('statusVerifiedDate', 'No info available')})",
            f"Start Date: {status_module.get('startDateStruct', {}).get('date', 'No info available')}",
            f"Completion Date: {status_module.get('completionDateStruct', {}).get('date', 'No info available')}",
            f"Why Stopped (if applicable): {status_module.get('whyStopped', 'N/A')}",
            f"Study Link: {url}",
        ]
    )
    overview_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.OVERVIEW,
        text=overview_chunk_text,
        embeddings=embed_text(overview_chunk_text, ChunkType.OVERVIEW),
    )

    # DESIGN
    phases = design_module.get("phases", ["No info available"])
    masked_entities = masking_info.get("whoMasked", ["No info available"])
    design_chunk_text = "\n".join(
        [
            f"Study Design Details ({nct_id})",
            f"Study Type: {design_module.get('studyType', 'No info available')}",
            f"Study Phases: {', '.join(phases)}",
            f"Study Design Info: {design_info.get('allocation', 'No info available')} allocation with {design_info.get('interventionModel', 'No info available')} intervention model. Primary purpose: {design_info.get('primaryPurpose', 'No info available')}",
            f"Masking Info: {masking_info.get('masking', 'No info available')} masking; Masked entities: {', '.join(masked_entities)}",
            f"Enrollment: {enrollment_info.get('count', 'No info available')} enrolled (type: {enrollment_info.get('type', 'No info available')})",
            f"Study Link: {url}",
        ]
    )
    design_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.DESIGN,
        text=design_chunk_text,
        embeddings=embed_text(design_chunk_text, ChunkType.DESIGN),
    )

    # ELIGIBILITY
    std_ages = eligibility_module.get("stdAges", ["No info available"])
    eligibility_chunk_text = "\n".join(
        [
            f"Eligibility Criteria ({nct_id})",
            f"Sex: {eligibility_module.get('sex', 'No info available')}",
            f"Minimum Age: {eligibility_module.get('minimumAge', 'No info available')}",
            f"Maximum Age: {eligibility_module.get('maximumAge', 'No info available')}",
            f"Healthy Volunteers: {eligibility_module.get('healthyVolunteers', 'No info available')}",
            f"Standard Ages: {', '.join(std_ages)}",
            f"Study Link: {url}",
        ]
    )
    eligibility_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.ELIGIBILITY,
        text=eligibility_chunk_text,
        embeddings=embed_text(eligibility_chunk_text, ChunkType.ELIGIBILITY),
    )

    # CONDITIONS
    conditions = conditions_module.get("conditions", ["No info available"])
    keywords = conditions_module.get("keywords", ["No info available"])
    conditions_chunk_text = "\n".join(
        [
            f"Condition Details ({nct_id})",
            f"Study-Related Conditions: {', '.join(conditions)}",
            f"Study Keywords: {', '.join(keywords)}",
            f"Study Link: {url}",
        ]
    )
    conditions_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.CONDITIONS,
        text=conditions_chunk_text,
        embeddings=embed_text(conditions_chunk_text, ChunkType.CONDITIONS),
    )

    # ARMS & INTERVENTIONS
    arms = arms_interventions_module.get("armGroups", [])
    interventions = arms_interventions_module.get("interventions", [])
    arms_text = "\n".join(
        [
            f"- Arm: {arm.get('label', 'No info available')}, Type: {arm.get('type', 'No info available')}"
            for arm in arms
        ]
    )
    interventions_text = "\n".join(
        [
            f"- Intervention: {interv.get('name', 'No info available')} ({interv.get('type', 'No info available')}): {interv.get('description', 'No info available')}"
            for interv in interventions
        ]
    )
    arms_interventions_chunk_text = "\n".join(
        [
            f"Arms and Interventions ({nct_id})",
            "Study Arms:",
            arms_text or "No arm group info available.",
            "",
            "Interventions:",
            interventions_text or "No intervention info available.",
        ]
    )
    arms_interventions_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.ARMS_INTERVENTIONS,
        text=arms_interventions_chunk_text,
        embeddings=embed_text(
            arms_interventions_chunk_text, ChunkType.ARMS_INTERVENTIONS
        ),
    )

    # OUTCOMES
    def format_outcomes(outcomes: list, label: str) -> str:
        if not outcomes:
            return f"No {label.lower()} outcomes info available."
        return "\n".join(
            [
                "\n".join(
                    [
                        f"{label.upper()} OUTCOME {i + 1}",
                        f"\tMeasure: {o.get('measure', 'No measure available')}",
                        f"\tDescription: {o.get('description', 'No description available')}",
                        f"\tTime Frame: {o.get('timeFrame', 'No time frame available')}",
                        f"Study Link: {url}",
                    ]
                )
                for i, o in enumerate(outcomes)
            ]
        )

    outcomes_primary = protocol.get("outcomesModule", {}).get("primaryOutcomes", [])
    outcomes_secondary = protocol.get("outcomesModule", {}).get("secondaryOutcomes", [])

    primary_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.OUTCOMES_PRIMARY,
        text="\n".join(
            [
                f"Primary Outcome Info ({nct_id}):",
                format_outcomes(outcomes_primary, "Primary"),
            ]
        ),
        embeddings=embed_text(
            format_outcomes(outcomes_primary, "Primary"), ChunkType.OUTCOMES_PRIMARY
        ),
    )

    secondary_chunk = Chunk(
        source_id=nct_id,
        metadata=study_metadata,
        section=ChunkType.OUTCOMES_SECONDARY,
        text="\n".join(
            [
                f"Secondary Outcome Info ({nct_id}):",
                format_outcomes(outcomes_secondary, "Secondary"),
            ]
        ),
        embeddings=embed_text(
            format_outcomes(outcomes_secondary, "Secondary"),
            ChunkType.OUTCOMES_SECONDARY,
        ),
    )

    return [
        overview_chunk,
        design_chunk,
        eligibility_chunk,
        conditions_chunk,
        arms_interventions_chunk,
        primary_chunk,
        secondary_chunk,
    ]
