"""
Parser Agent — Downloads arXiv papers and extracts structured text.
"""

import re
import tempfile
from pathlib import Path

import arxiv
import fitz  # PyMuPDF
import httpx

from backend.models.schemas import ParsedPaper, PaperMetadata


SECTION_PATTERNS = [
    r"^\d+\.?\s+[A-Z]",
    r"^[A-Z][a-z]+(?:\s+[A-Z&][a-z]*)*\s*$",
    r"^(?:Abstract|Introduction|Related Work|Background|Method|"
    r"Methodology|Approach|Experiments|Results|Discussion|"
    r"Conclusion|Acknowledgements|References|Appendix)",
]


def _is_section_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


def _extract_arxiv_id(url: str) -> str:
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract arXiv ID from: {url}")


async def _download_pdf(arxiv_id: str) -> Path:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    tmp = Path(tempfile.mkdtemp()) / f"{arxiv_id}.pdf"

    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(pdf_url)
        resp.raise_for_status()
        tmp.write_bytes(resp.content)

    return tmp


def _fetch_metadata(arxiv_id: str) -> PaperMetadata:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))

    if not results:
        return PaperMetadata(arxiv_id=arxiv_id, url=f"https://arxiv.org/abs/{arxiv_id}")

    paper = results[0]
    return PaperMetadata(
        title=paper.title,
        authors=[a.name for a in paper.authors],
        abstract=paper.summary,
        arxiv_id=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _extract_text_and_sections(pdf_path: Path) -> tuple[str, dict[str, str], int]:
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)

    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()

    lines = full_text.split("\n")
    sections: dict[str, str] = {}
    current_section = "preamble"
    current_text: list[str] = []

    for line in lines:
        if _is_section_heading(line):
            if current_text:
                sections[current_section] = "\n".join(current_text).strip()
            current_section = line.strip()
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        sections[current_section] = "\n".join(current_text).strip()

    return full_text, sections, num_pages


async def parse_paper(url: str) -> ParsedPaper:
    arxiv_id = _extract_arxiv_id(url)
    metadata = _fetch_metadata(arxiv_id)
    pdf_path = await _download_pdf(arxiv_id)

    try:
        raw_text, sections, num_pages = _extract_text_and_sections(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)

    return ParsedPaper(
        metadata=metadata,
        sections=sections,
        raw_text=raw_text,
        num_pages=num_pages,
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        url = "https://arxiv.org/abs/1512.03385"  # ResNet paper
        print(f"Parsing: {url}\n")

        paper = await parse_paper(url)

        print(f"Title: {paper.metadata.title}")
        print(f"Authors: {', '.join(paper.metadata.authors[:3])}...")
        print(f"Pages: {paper.num_pages}")
        print(f"Sections found: {len(paper.sections)}")
        print(f"\nSection names:")
        for name in paper.sections:
            preview = paper.sections[name][:80].replace("\n", " ")
            print(f"  [{name}] -> {preview}...")
        print(f"\nTotal chars: {len(paper.raw_text):,}")

    asyncio.run(main())
