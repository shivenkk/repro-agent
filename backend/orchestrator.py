"""
Orchestrator — Runs the reproducibility analysis pipeline.

Usage:
    python -m backend.orchestrator https://arxiv.org/abs/1512.03385
"""

import uuid
import asyncio
from typing import Callable

from backend.models.schemas import (
    PipelineState, AgentStatus, ProgressUpdate,
    ReproducibilityReport, MetricComparison, Verdict, FailureReason,
)
from backend.agents.parser import parse_paper
from backend.agents.extractor import extract_methodology
from backend.agents.codegen import generate_code


def _log(state: PipelineState, agent: str, message: str, callback=None):
    state.current_agent = agent
    state.progress_messages.append(f"[{agent}] {message}")
    print(f"  [{agent}] {message}")
    if callback:
        update = ProgressUpdate(
            run_id=state.run_id,
            agent=agent,
            status=AgentStatus.RUNNING,
            message=message,
        )
        callback(update)


def _build_report(state: PipelineState) -> ReproducibilityReport:
    """Build report based on extraction quality and code generation."""
    method = state.methodology
    score = 0.0

    # Hyperparameter completeness (40 pts)
    if method.hyperparameters:
        high_conf = sum(1 for f in method.hyperparameters.values() if f.confidence >= 0.7)
        total = len(method.hyperparameters)
        score += 40.0 * (high_conf / total) if total > 0 else 0.0

    # Dataset identified (15 pts)
    if method.dataset and method.dataset.name != "unknown":
        score += 15.0

    # Model identified (15 pts)
    if method.model_architecture.confidence >= 0.7:
        score += 15.0

    # Code was generated (20 pts)
    if state.generated_code and len(state.generated_code.script) > 100:
        score += 20.0

    # Has claimed results (10 pts)
    if method.claimed_results:
        score += 10.0

    score = round(min(score, 100.0), 1)

    if score >= 75:
        verdict = Verdict.REPRODUCIBLE
    elif score >= 45:
        verdict = Verdict.PARTIALLY
    else:
        verdict = Verdict.NOT_REPRODUCIBLE

    failure_reasons = []
    if method.missing_details:
        failure_reasons.append(FailureReason.MISSING_HYPERPARAMS)
    if method.dataset and not method.dataset.is_standard:
        failure_reasons.append(FailureReason.DATASET_UNAVAILABLE)

    comparisons = [
        MetricComparison(
            metric_name=name,
            claimed=val,
            achieved=None,
            delta=None,
            within_threshold=False,
        )
        for name, val in method.claimed_results.items()
    ]

    return ReproducibilityReport(
        paper=state.parsed_paper.metadata,
        methodology=method,
        extraction_confidence=method.avg_confidence,
        final_code=state.generated_code.script if state.generated_code else "",
        comparisons=comparisons,
        overall_score=score,
        verdict=verdict,
        failure_reasons=failure_reasons,
        analysis="",
        recommendations=[],
    )


async def run_pipeline(
    paper_url: str,
    progress_callback: Callable | None = None,
) -> PipelineState:
    state = PipelineState(
        run_id=str(uuid.uuid4())[:8],
        paper_url=paper_url,
        status=AgentStatus.RUNNING,
    )

    cb = progress_callback

    try:
        # ── Step 1: Parse ──────────────────────────────
        _log(state, "parser", "Downloading and parsing paper...")
        state.parsed_paper = await parse_paper(paper_url)
        _log(state, "parser",
             f"Parsed: {state.parsed_paper.metadata.title} "
             f"({state.parsed_paper.num_pages} pages, "
             f"{len(state.parsed_paper.raw_text):,} chars)")

        # ── Step 2: Extract ────────────────────────────
        _log(state, "extractor", "Extracting methodology...")
        state.methodology = await extract_methodology(state.parsed_paper)
        hp_count = len(state.methodology.hyperparameters)
        conf = state.methodology.avg_confidence
        _log(state, "extractor",
             f"Extracted {hp_count} hyperparameters "
             f"(avg confidence: {conf:.2f})")

        # ── Step 3: Generate Code ──────────────────────
        _log(state, "codegen", "Generating training script...")
        state.generated_code = await generate_code(state.methodology)
        _log(state, "codegen",
             f"Generated {len(state.generated_code.script)} chars, "
             f"{len(state.generated_code.requirements)} packages needed")
        if state.generated_code.assumptions:
            _log(state, "codegen",
                 f"Assumptions: {', '.join(state.generated_code.assumptions)}")

        # ── Step 4: Report ─────────────────────────────
        _log(state, "evaluator", "Generating report...")
        state.report = _build_report(state)
        _log(state, "evaluator",
             f"Score: {state.report.overall_score}/100 — "
             f"Verdict: {state.report.verdict.value}")

        state.status = AgentStatus.SUCCESS

    except Exception as e:
        state.status = AgentStatus.ERROR
        state.error = str(e)
        _log(state, state.current_agent or "orchestrator", f"FATAL: {e}")

    return state


def _print_report(state: PipelineState):
    r = state.report
    if not r:
        print("\nNo report generated.")
        return

    print(f"\n{'='*60}")
    print(f"  REPRODUCIBILITY REPORT")
    print(f"{'='*60}")
    print(f"  Paper:  {r.paper.title}")
    print(f"  URL:    {r.paper.url}")
    print(f"  Score:  {r.overall_score}/100")
    print(f"  Verdict: {r.verdict.value.upper()}")
    print(f"{'='*60}")

    print(f"\n  Extraction Confidence: {r.extraction_confidence:.2f}")
    print(f"  Code Generated: {'Yes' if r.final_code else 'No'}")

    if r.methodology.hyperparameters:
        print(f"\n  Hyperparameters Found:")
        for name, field in r.methodology.hyperparameters.items():
            conf_bar = "█" * int(field.confidence * 10) + "░" * (10 - int(field.confidence * 10))
            print(f"    {name}: {field.value}  [{conf_bar}] {field.confidence:.1f}")

    if r.comparisons:
        print(f"\n  Claimed Results:")
        for c in r.comparisons:
            print(f"    {c.metric_name}: {c.claimed:.4f}")

    if r.methodology.missing_details:
        print(f"\n  Missing from Paper:")
        for item in r.methodology.missing_details:
            print(f"    - {item}")

    if r.final_code:
        print(f"\n  Generated Code: {len(r.final_code)} chars")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://arxiv.org/abs/1512.03385"

    print(f"ReproAgent")
    print(f"Paper: {url}\n")

    state = asyncio.run(run_pipeline(url))
    _print_report(state)
