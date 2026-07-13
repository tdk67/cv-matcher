"""Shared data structures for the agentic pipeline.

PipelineContext flows through all 4 agents: Planner -> Retriever -> Responder -> Validator.
"""

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the knowledge base."""
    text: str
    source: str         # filename
    section: str        # CV section
    similarity: float   # 0-1 score
    metadata: dict = field(default_factory=dict)


@dataclass
class MatchCandidate:
    """A matched person with score and evidence."""
    person_name: str
    score: float            # 0-100 percentage
    evidence: str           # supporting text snippet/justification
    source_document: str
    sections: list[str]



@dataclass
class PipelineContext:
    """Shared state flowing through the agent pipeline."""
    # Input
    query: str = ""

    # Planner output
    query_type: str = "unknown"     # "keyword" | "similarity" | "complex" | "out_of_scope"
    extracted_skills: list[str] = field(default_factory=list)
    extracted_requirements: str = ""
    is_in_scope: bool = True
    rejection_reason: str = ""

    # Retriever output
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)

    # Responder output
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    match_candidates: list[MatchCandidate] = field(default_factory=list)

    # Validator output
    validation_passed: bool = False
    validation_feedback: str = ""
    validation_failure_type: str = ""

    # Pipeline state
    retry_count: int = 0
    max_retries: int = 3
    total_latency_ms: float = 0.0
