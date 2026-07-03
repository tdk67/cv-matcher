"""Section-aware document chunker for CV documents.

Splits text on CV section headers first, then applies recursive
character splitting within sections. Each chunk is tagged with its
section name for citation.
"""

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


from src.config import settings

# Known CV section headers (all-caps, standalone lines)
CV_HEADER_RE = re.compile(
    r"^(" + "|".join(settings.cv_section_headers) + r")$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    section: str
    chunk_index: int
    chars: int


# Short all-caps tokens that are NOT section headers
_HEADER_BLOCKLIST = set(settings.header_blocklist)


def is_cv_section_header(line: str) -> bool:
    """Check if a line is a CV section header."""
    stripped = line.strip()
    if not stripped:
        return False
    # Must be a short line
    if len(stripped) > 40:
        return False
    # Reject blocklisted abbreviations
    lower = stripped.lower().replace(" ", "_").replace("-", "_")
    if lower in _HEADER_BLOCKLIST:
        return False
    # Match known patterns
    if CV_HEADER_RE.match(stripped):
        return True
    # Heuristic: all-caps, short, standalone — but reject if single word + blocklisted
    if stripped.isupper() and len(stripped.split()) <= 4 and len(stripped) <= 30:
        # Single all-caps words need to be in the header patterns or long enough
        if len(stripped.split()) == 1 and len(stripped) <= 4:
            return False  # Too short, likely an abbreviation
        return True
    return False


def detect_cv_sections(text: str) -> list[tuple[str, str]]:
    """Split text into named sections based on CV headers.

    Returns list of (section_name, section_content) tuples.
    """
    lines = text.split("\n")
    sections = []
    current_name = "header"
    current_lines = []

    for line in lines:
        if is_cv_section_header(line):
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = line.strip().lower().replace(" ", "_").replace("-", "_")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))

    return sections


def chunk_document(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Chunk a document using section-aware splitting.

    1. Detect CV sections by header patterns
    2. Split each section with RecursiveCharacterTextSplitter
    3. Tag each chunk with its section name

    Args:
        text: Full document text
        chunk_size: Target chunk size in characters (default 500)
        chunk_overlap: Overlap between chunks in characters (default 100)

    Returns:
        List of Chunk objects with section metadata
    """
    sections = detect_cv_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks = []
    global_index = 0

    for section_name, section_content in sections:
        if not section_content.strip():
            continue

        section_chunks = splitter.split_text(section_content)
        for chunk_text in section_chunks:
            all_chunks.append(Chunk(
                text=chunk_text,
                section=section_name,
                chunk_index=global_index,
                chars=len(chunk_text),
            ))
            global_index += 1

    return all_chunks
