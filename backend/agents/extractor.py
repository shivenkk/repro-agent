"""
Extractor Agent — Reads parsed paper text and extracts structured methodology.
"""

from backend.models.schemas import (
    ParsedPaper, Methodology, ExtractedField, DatasetInfo
)
from backend.services.llm import ask_llm_json

STANDARD_DATASETS = {
    "mnist", "cifar-10", "cifar-100", "imagenet", "imagenet 2012",
    "imagenet2012", "coco", "voc", "svhn", "stl-10", "fashion-mnist",
    "celeba", "glue", "squad", "sst-2", "mnli", "imdb", "ag_news",
    "wikitext", "ptb", "c4", "openwebtext",
}

EXTRACTION_PROMPT = """Extract the methodology from this ML paper as JSON.

Return EXACTLY this structure:
{{
    "model_architecture": {{"value": "name", "confidence": 0.0-1.0, "source": "Section X"}},
    "architecture_details": {{"detail_name": {{"value": "...", "confidence": 0.9, "source": "Section X"}}}},
    "dataset": {{"name": "...", "splits": {{"train": null, "val": null, "test": null}}, "preprocessing": "..."}},
    "hyperparameters": {{
        "learning_rate": {{"value": 0.1, "confidence": 1.0, "source": "Section X"}},
        "batch_size": {{"value": 256, "confidence": 1.0, "source": "Section X"}},
        "epochs": {{"value": 100, "confidence": 0.5, "source": "inferred"}},
        "optimizer": {{"value": "SGD", "confidence": 1.0, "source": "Section X"}},
        "weight_decay": {{"value": 0.0001, "confidence": 1.0, "source": "Section X"}},
        "momentum": {{"value": 0.9, "confidence": 1.0, "source": "Section X"}},
        "lr_schedule": {{"value": "description", "confidence": 0.8, "source": "Section X"}}
    }},
    "loss_function": {{"value": "cross entropy", "confidence": 1.0, "source": "Section X"}},
    "optimizer_details": {{"value": "SGD with details", "confidence": 1.0, "source": "Section X"}},
    "training_procedure": "brief description",
    "evaluation_metrics": ["accuracy", "top-5 error"],
    "claimed_results": {{"metric_name": 0.92}},
    "hardware": "GPUs if mentioned, else empty string",
    "missing_details": ["only things genuinely NOT in the paper"]
}}

Rules:
- Use ACTUAL numbers from the paper, not null/None
- confidence 1.0 = explicitly stated, 0.7+ = clearly implied, <0.4 = not found
- "hardware" must be a plain string
- "claimed_results" values must be decimal numbers (0.0357 not 3.57%)
- Search ALL sections for numbers near "learning rate", "batch", "momentum", "weight decay"

PAPER:
{paper_text}"""


def _truncate_paper(paper: ParsedPaper, max_chars: int = 15000) -> str:
    """
    Build a focused version of the paper text.
    Instead of blindly taking the first N chars, we extract
    the most information-dense chunks by searching for regions
    that contain hyperparameter keywords.
    """
    text = paper.raw_text

    # Try to cut off references section
    ref_markers = ["References\n", "REFERENCES\n", "Bibliography\n"]
    for marker in ref_markers:
        idx = text.rfind(marker)
        if idx > len(text) * 0.5:
            text = text[:idx]
            break

    # If short enough, return as-is
    if len(text) <= max_chars:
        return text

    # Strategy: take the abstract/intro (first 4k chars) +
    # find the most keyword-rich chunks from the rest.
    # This ensures we capture the experiments/implementation section
    # even if it's deep in the paper.

    head = text[:3000]
    rest = text[3000:]

    # Split remaining text into overlapping chunks
    chunk_size = 2000
    step = 1500
    chunks = []
    for i in range(0, len(rest), step):
        chunk = rest[i:i + chunk_size]
        if chunk.strip():
            chunks.append((i, chunk))

    # Score each chunk by density of methodology keywords
    keywords = [
        "learning rate", "batch size", "epoch", "optimizer",
        "sgd", "adam", "momentum", "weight decay", "dropout",
        "train", "iteration", "schedule", "warm", "decay",
        "augment", "crop", "flip", "loss", "cross entropy",
        "accuracy", "error rate", "top-1", "top-5", "f1",
        "bleu", "implementation", "hyperparameter", "lr",
        "regulariz", "initialize", "layer", "conv", "resid",
        "hidden", "dimension", "attention", "head",
    ]

    scored = []
    for pos, chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(kw) for kw in keywords)
        scored.append((score, pos, chunk))

    # Sort by score descending, take best chunks
    scored.sort(key=lambda x: -x[0])

    # Take top chunks until we hit the budget
    budget = max_chars - len(head) - 200  # leave room for markers
    selected = []
    used = 0
    for score, pos, chunk in scored:
        if used + len(chunk) > budget:
            continue
        selected.append((pos, chunk))
        used += len(chunk)

    # Sort selected chunks by position to maintain reading order
    selected.sort(key=lambda x: x[0])

    parts = [head, "\n\n[...]\n\n"]
    for pos, chunk in selected:
        parts.append(chunk)
        parts.append("\n[...]\n")

    return "".join(parts)


def _parse_extracted_field(data) -> ExtractedField:
    """Safely parse an ExtractedField from LLM output."""
    if isinstance(data, dict) and "value" in data:
        return ExtractedField(
            value=data["value"],
            confidence=float(data.get("confidence", 0.5)),
            source=data.get("source", ""),
        )
    return ExtractedField(value=data, confidence=0.5, source="")


def _build_methodology(raw: dict) -> Methodology:
    """Convert raw LLM JSON output into a typed Methodology object."""

    # Model architecture
    arch_raw = raw.get("model_architecture", {"value": "unknown"})
    model_architecture = _parse_extracted_field(arch_raw)

    # Architecture details
    arch_details = {}
    for k, v in raw.get("architecture_details", {}).items():
        arch_details[k] = _parse_extracted_field(v)

    # Dataset
    ds_raw = raw.get("dataset", {})
    dataset_name = ds_raw.get("name", "unknown") if isinstance(ds_raw, dict) else str(ds_raw)
    dataset = DatasetInfo(
        name=dataset_name,
        splits=ds_raw.get("splits", {}) if isinstance(ds_raw, dict) else {},
        preprocessing=ds_raw.get("preprocessing", "") if isinstance(ds_raw, dict) else "",
        is_standard=dataset_name.lower().strip() in STANDARD_DATASETS,
    )

    # Hyperparameters
    hyperparams = {}
    for k, v in raw.get("hyperparameters", {}).items():
        hyperparams[k] = _parse_extracted_field(v)

    # Loss function
    loss_raw = raw.get("loss_function")
    loss_fn = _parse_extracted_field(loss_raw) if loss_raw else None

    # Optimizer
    opt_raw = raw.get("optimizer_details") or raw.get("optimizer")
    optimizer = _parse_extracted_field(opt_raw) if opt_raw else None

    # Evaluation metrics
    eval_metrics = raw.get("evaluation_metrics", [])
    if isinstance(eval_metrics, str):
        eval_metrics = [eval_metrics]

    # Claimed results
    claimed = raw.get("claimed_results", {})
    claimed_clean = {}
    for k, v in claimed.items():
        try:
            claimed_clean[k] = float(v)
        except (ValueError, TypeError):
            pass

    # Hardware (LLM sometimes returns dict instead of string)
    hardware_raw = raw.get("hardware", "")
    if isinstance(hardware_raw, dict):
        hardware = str(hardware_raw.get("value", ""))
    else:
        hardware = str(hardware_raw)

    # Missing details
    missing = raw.get("missing_details", [])
    if isinstance(missing, str):
        missing = [missing]

    # Calculate average confidence
    all_confidences = [model_architecture.confidence]
    for f in hyperparams.values():
        all_confidences.append(f.confidence)
    for f in arch_details.values():
        all_confidences.append(f.confidence)
    if loss_fn:
        all_confidences.append(loss_fn.confidence)
    if optimizer:
        all_confidences.append(optimizer.confidence)
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    return Methodology(
        model_architecture=model_architecture,
        architecture_details=arch_details,
        dataset=dataset,
        hyperparameters=hyperparams,
        training_procedure=raw.get("training_procedure", ""),
        loss_function=loss_fn,
        optimizer=optimizer,
        evaluation_metrics=eval_metrics,
        claimed_results=claimed_clean,
        hardware=hardware,
        missing_details=missing,
        avg_confidence=round(avg_conf, 3),
    )


async def extract_methodology(paper: ParsedPaper) -> Methodology:
    """
    Main entry point. Takes a parsed paper, returns structured methodology.
    """
    paper_text = _truncate_paper(paper)
    prompt = EXTRACTION_PROMPT.format(paper_text=paper_text)

    raw = await ask_llm_json(prompt)
    methodology = _build_methodology(raw)

    return methodology


# ── Quick test: parse ResNet then extract ──────────────

if __name__ == "__main__":
    import asyncio
    from backend.agents.parser import parse_paper

    async def main():
        url = "https://arxiv.org/abs/1512.03385"
        print(f"Parsing: {url}")
        paper = await parse_paper(url)
        print(f"Parsed: {paper.metadata.title}\n")

        print("Extracting methodology...")
        method = await extract_methodology(paper)

        print(f"\n{'='*60}")
        print(f"MODEL: {method.model_architecture.value}")
        print(f"  confidence: {method.model_architecture.confidence}")
        print(f"\nDATASET: {method.dataset.name if method.dataset else 'unknown'}")
        print(f"  standard: {method.dataset.is_standard if method.dataset else False}")
        print(f"  preprocessing: {method.dataset.preprocessing if method.dataset else 'none'}")

        print(f"\nHYPERPARAMETERS:")
        for name, field in method.hyperparameters.items():
            print(f"  {name}: {field.value} (conf: {field.confidence}, src: {field.source})")

        print(f"\nLOSS: {method.loss_function.value if method.loss_function else 'unknown'}")
        print(f"OPTIMIZER: {method.optimizer.value if method.optimizer else 'unknown'}")

        print(f"\nCLAIMED RESULTS:")
        for metric, val in method.claimed_results.items():
            print(f"  {metric}: {val}")

        print(f"\nEVAL METRICS: {method.evaluation_metrics}")
        print(f"HARDWARE: {method.hardware}")

        print(f"\nMISSING DETAILS:")
        for item in method.missing_details:
            print(f"  - {item}")

        print(f"\nOVERALL CONFIDENCE: {method.avg_confidence}")
        print(f"{'='*60}")

    asyncio.run(main())
