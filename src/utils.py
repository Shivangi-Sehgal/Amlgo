"""
utils.py
--------
Shared helper functions used across the project.
- Section detection
- Word count
- Chunk formatting for display
"""

import re
import json
import os


def detect_section(text: str) -> str:
    """Detect which eBay agreement section a chunk belongs to."""
    pattern = re.compile(
        r'\b(\d{1,2})\.\s+(Introduction|About eBay|Using eBay|Vehicle|Policy|Fees|'
        r'Listing|Purchase|International|Content|Notice|Holds|Authorization|'
        r'Additional|Payment|Disclaimer|Release|Indemnity|Legal|General)\b',
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return f"Section {match.group(1)} - {match.group(2)}"
    return "General"


def word_count(text: str) -> int:
    """Return number of words in text."""
    return len(text.split())


def save_chunks_to_json(chunks, output_path: str):
    """
    Save LangChain Document chunks to JSON with metadata.
    Each entry: chunk_id, text, word_count, char_count, section, source_page
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        output.append({
            "chunk_id":    i,
            "text":        text,
            "word_count":  word_count(text),
            "char_count":  len(text),
            "section":     detect_section(text),
            "source_page": chunk.metadata.get("page", 0) + 1
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[Utils] Saved {len(output)} chunks → {output_path}")
    return output


def print_chunk_stats(chunks_data: list):
    """Print summary stats about chunks."""
    wc = [c["word_count"] for c in chunks_data]
    sections = set(c["section"] for c in chunks_data)
    print("\n Chunk Statistics ")
    print(f"  Total chunks    : {len(chunks_data)}")
    print(f"  Min words       : {min(wc)}")
    print(f"  Max words       : {max(wc)}")
    print(f"  Avg words       : {sum(wc) // len(wc)}")
    print(f"  Sections found  : {len(sections)}")
    for s in sorted(sections):
        print(f"    → {s}")


def format_source_chunks(source_docs: list) -> str:
    """Format retrieved source chunks for display in Streamlit."""
    formatted = []
    for i, doc in enumerate(source_docs):
        page = doc.metadata.get("page", 0) + 1
        section = detect_section(doc.page_content)
        preview = doc.page_content[:300].strip()
        formatted.append(
            f"**Source {i+1}** | Page {page} | {section}\n\n{preview}..."
        )
    return "\n\n---\n\n".join(formatted)