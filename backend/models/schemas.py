"""
Core data models for the ReproAgent pipeline.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"

class Verdict(str, Enum):
    REPRODUCIBLE = "reproducible"
    PARTIALLY = "partially_reproducible"
    NOT_REPRODUCIBLE = "not_reproducible"
    INCONCLUSIVE = "inconclusive"

class FailureReason(str, Enum):
    MISSING_HYPERPARAMS = "missing_hyperparameters"
    DATASET_UNAVAILABLE = "dataset_unavailable"
    HARDWARE_GAP = "hardware_gap"
    AMBIGUOUS_METHOD = "ambiguous_methodology"
    CODE_GEN_FAILURE = "code_generation_failure"
    UNKNOWN = "unknown"


# ── Parser Output ──────────────────────────────────────

class PaperMetadata(BaseModel):
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    arxiv_id: str = ""
    url: str = ""

class ParsedPaper(BaseModel):
    metadata: PaperMetadata
    sections: dict[str, str] = Field(default_factory=dict)
    raw_text: str = ""
    num_pages: int = 0


# ── Extractor Output ──────────────────────────────────

class DatasetInfo(BaseModel):
    name: str
    splits: dict[str, Any] = Field(default_factory=dict)
    preprocessing: str = ""
    is_standard: bool = False

class ExtractedField(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = ""

class Methodology(BaseModel):
    model_architecture: ExtractedField
    architecture_details: dict[str, ExtractedField] = Field(default_factory=dict)
    dataset: DatasetInfo | None = None
    hyperparameters: dict[str, ExtractedField] = Field(default_factory=dict)
    training_procedure: str = ""
    loss_function: ExtractedField | None = None
    optimizer: ExtractedField | None = None
    evaluation_metrics: list[str] = Field(default_factory=list)
    claimed_results: dict[str, float] = Field(default_factory=dict)
    hardware: str = ""
    missing_details: list[str] = Field(default_factory=list)
    avg_confidence: float = 0.0


# ── CodeGen Output ─────────────────────────────────────

class GeneratedCode(BaseModel):
    script: str
    requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


# ── Report Output ──────────────────────────────────────

class MetricComparison(BaseModel):
    metric_name: str
    claimed: float
    achieved: float | None = None
    delta: float | None = None
    within_threshold: bool = False

class ReproducibilityReport(BaseModel):
    paper: PaperMetadata
    methodology: Methodology
    extraction_confidence: float = 0.0
    final_code: str = ""
    comparisons: list[MetricComparison] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    verdict: Verdict = Verdict.INCONCLUSIVE
    failure_reasons: list[FailureReason] = Field(default_factory=list)
    analysis: str = ""
    recommendations: list[str] = Field(default_factory=list)


# ── Pipeline State ─────────────────────────────────────

class PipelineState(BaseModel):
    run_id: str
    paper_url: str
    status: AgentStatus = AgentStatus.PENDING
    parsed_paper: ParsedPaper | None = None
    methodology: Methodology | None = None
    generated_code: GeneratedCode | None = None
    report: ReproducibilityReport | None = None
    current_agent: str = ""
    progress_messages: list[str] = Field(default_factory=list)
    error: str = ""


# ── WebSocket Messages ─────────────────────────────────

class ProgressUpdate(BaseModel):
    run_id: str
    agent: str
    status: AgentStatus
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
