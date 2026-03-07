"""
CodeGen Agent — Takes extracted methodology and generates a reproducible training script.
"""

import json
from backend.models.schemas import Methodology, GeneratedCode
from backend.services.llm import ask_llm

CODEGEN_PROMPT = """Generate a complete Python training script from this methodology.

Rules:
- PyTorch only. ALL imports at top. One file. No markdown.
- Config dict at top for hyperparameters
- Seeding: torch.manual_seed(42), np.random.seed(42)
- device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
- NEVER use ImageNet. Use CIFAR-10 via torchvision.datasets.CIFAR10(download=True) with 10 classes
- Use torchvision.models (resnet18, etc) when possible instead of writing from scratch
- Only use torch, torchvision, numpy — no other packages
- Print per epoch: EPOCH: n/total | LOSS: val | ACCURACY: val
- Print final: RESULT: ACCURACY=val
- Cap at 3 epochs. Mark assumptions with # ASSUMPTION: ...
- Include train loop + validation loop

METHODOLOGY:
{methodology_json}

Output ONLY the Python script:"""


def _methodology_to_json(method: Methodology) -> str:
    """Convert methodology to a compact JSON string for the prompt."""
    data = {
        "model": method.model_architecture.value,
        "dataset": method.dataset.name if method.dataset else "unknown",
        "hyperparameters": {
            k: v.value for k, v in method.hyperparameters.items()
        },
        "loss": method.loss_function.value if method.loss_function else "cross_entropy",
        "optimizer": method.optimizer.value if method.optimizer else "SGD",
        "metrics": method.evaluation_metrics,
        "claimed_results": method.claimed_results,
    }
    return json.dumps(data, indent=2)


def _force_cifar10_substitution(script: str) -> str:
    """
    If the script tries to use ImageNet, force-replace with CIFAR-10.
    Safety net for when the LLM ignores prompt instructions.
    """
    imagenet_indicators = [
        "datasets.ImageNet",
        "ImageNet(",
        "imagenet",
        "ILSVRC",
    ]

    needs_fix = any(indicator in script for indicator in imagenet_indicators)
    if not needs_fix:
        return script

    replacements = [
        ("datasets.ImageNet(", "datasets.CIFAR10("),
        ("ImageNet(", "CIFAR10("),
        ("num_classes=1000", "num_classes=10"),
        ("num_classes = 1000", "num_classes = 10"),
        ("fc = nn.Linear(512, 1000)", "fc = nn.Linear(512, 10)"),
        ("fc = nn.Linear(2048, 1000)", "fc = nn.Linear(2048, 10)"),
    ]
    for old, new in replacements:
        script = script.replace(old, new)

    if "ASSUMPTION: Using CIFAR-10" not in script:
        script = "# ASSUMPTION: Using CIFAR-10 instead of ImageNet (auto-substituted)\n" + script

    return script


def _extract_requirements(script: str) -> list[str]:
    """Parse imports to determine pip packages needed."""
    import_map = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "transformers": "transformers",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
        "scipy": "scipy",
    }

    reqs = set()
    for line in script.split("\n"):
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            for module, package in import_map.items():
                if module in line:
                    reqs.add(package)

    return sorted(reqs)


def _extract_assumptions(script: str) -> list[str]:
    """Extract lines marked with # ASSUMPTION:"""
    assumptions = []
    for line in script.split("\n"):
        if "# ASSUMPTION:" in line:
            assumption = line.split("# ASSUMPTION:")[-1].strip()
            assumptions.append(assumption)
    return assumptions


async def generate_code(methodology: Methodology) -> GeneratedCode:
    """
    Generate a training script from extracted methodology.
    """
    method_json = _methodology_to_json(methodology)
    prompt = CODEGEN_PROMPT.format(methodology_json=method_json)

    response = await ask_llm(prompt)

    # Clean up markdown fences if LLM added them
    script = response.strip()
    if script.startswith("```python"):
        script = script[len("```python"):].strip()
    if script.startswith("```"):
        script = script[3:].strip()
    if script.endswith("```"):
        script = script[:-3].strip()

    # Force-remove ImageNet references
    script = _force_cifar10_substitution(script)

    requirements = _extract_requirements(script)
    assumptions = _extract_assumptions(script)

    return GeneratedCode(
        script=script,
        requirements=requirements,
        assumptions=assumptions,
    )


# ── Quick test ─────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from backend.agents.parser import parse_paper
    from backend.agents.extractor import extract_methodology

    async def main():
        url = "https://arxiv.org/abs/1512.03385"
        print(f"Parsing: {url}")
        paper = await parse_paper(url)
        print(f"Parsed: {paper.metadata.title}\n")

        print("Extracting methodology...")
        method = await extract_methodology(paper)
        print(f"Extraction confidence: {method.avg_confidence}\n")

        print("Generating code...")
        code = await generate_code(method)

        print(f"\n{'='*60}")
        print(f"REQUIREMENTS: {code.requirements}")
        print(f"ASSUMPTIONS: {code.assumptions}")
        print(f"SCRIPT LENGTH: {len(code.script)} chars")
        print(f"{'='*60}")
        print(f"\n{code.script}")

    asyncio.run(main())
