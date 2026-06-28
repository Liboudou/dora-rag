import re
from typing import List, Tuple

HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+)?(ARTICLE|CHAPTER|SECTION|ANNEX|TITLE|PART)\s+([\wIVXLCDM\d]+)",
    re.IGNORECASE | re.MULTILINE,
)

PAGE_RE = re.compile(r"(?:\[Page\s+(\d+)\]|Page\s+(\d+)(?:\s+of|\b))", re.IGNORECASE)


def _extract_page(text_before: str) -> int | None:
    m = PAGE_RE.search(text_before)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def chunk_text(text: str, base_metadata: dict) -> List[Tuple[str, dict]]:
    headings = [
        (m.start(), m.group(1).upper(), m.group(2).upper())
        for m in HEADING_RE.finditer(text)
    ]

    if not headings:
        return [(text.strip(), {**base_metadata})]

    chunks: List[Tuple[str, dict]] = []
    for i, (pos, htype, hnum) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        chunk_text_content = text[pos:end].strip()

        metadata = {**base_metadata, f"{htype.lower()}_number": hnum}
        page = _extract_page(text[max(0, pos - 200) : pos])
        if page is not None:
            metadata["page"] = page

        chunks.append((chunk_text_content, metadata))

    return chunks
