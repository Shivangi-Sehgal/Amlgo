"""
utils.py
--------
Shared helper functions used across the project.
- Section detection (with propagation fix)
- Word count
- Chunk formatting for display

Fix for "General" tagging issue:
  Old approach: only looked for section headers inside chunk text.
  Problem: continuation chunks don't contain the section header — they
           just have body text, so they got tagged as "General".
  Fix: detect_section_from_page() maps page number → section using the
       eBay document's known structure. save_chunks_to_json() then
       propagates the last known section forward to continuation chunks.
"""

import re
import json
import os


# ── Page → Section map for eBay User Agreement ───────────────────────────────
# Each entry: (start_page, section_label)
# Based on the actual document structure (1-indexed pages)
PAGE_SECTION_MAP = [
    (1,  "Section 1 - Introduction"),
    (1,  "Section 2 - About eBay"),
    (2,  "Section 3 - Using eBay"),
    (3,  "Section 4 - Vehicle Purchases & Sales"),
    (4,  "Section 5 - Policy Enforcement"),
    (4,  "Section 6 - Fees and Taxes"),
    (5,  "Section 7 - Listing Conditions"),
    (6,  "Section 8 - Purchase Conditions"),
    (7,  "Section 9 - International Buying and Selling"),
    (7,  "Section 10 - Content"),
    (8,  "Section 11 - Notice for IP Violations"),
    (8,  "Section 12 - Holds and Restricted Funds"),
    (8,  "Section 13 - Authorization to Contact You"),
    (9,  "Section 14 - Additional Terms"),
    (10, "Section 15 - Payment Services"),
    (11, "Section 16 - Disclaimer of Warranties"),
    (12, "Section 17 - Release"),
    (13, "Section 18 - Indemnity"),
    (13, "Section 19 - Legal Disputes"),
    (19, "Section 20 - General"),
]


def detect_section_from_header(text: str) -> str:
    """
    Try to detect section from explicit header text in the chunk.
    e.g. '3. Using eBay' or '19. Legal Disputes'
    Returns section string if found, else None.
    """
    pattern = re.compile(
    r'(\d{1,2})\.\s+(Introduction|About eBay|Using eBay|Vehicle Purchases and Sales|Policy Enforcement|Fees and Taxes|'
    r'Listing Conditions|Purchase Conditions|International Buying and Selling; Translation|Content|'
    r'Notice for Claims of Intellectual Property Violations and Copyright Infringement Pursuant to Section 512\(c\) of Title 17 of the United States Code|'
    r'Holds and Restricted Funds|Authorization to Contact You; Recording Calls; Analyzing Message Content|'
    r'Additional Terms|Payment Services|Disclaimer of Warranties; Limitation of Liability|Release|Indemnity|Legal Disputes|General)',
    re.IGNORECASE | re.MULTILINE
)
    match = pattern.search(text)
    if match:
        return f"Section {match.group(1)} - {match.group(2)}"
    return None


def detect_section_from_page(page: int) -> str:
    """
    Map a page number to a section using the document's known structure.
    Falls back to the last section that started on or before this page.
    """
    best = PAGE_SECTION_MAP[0][1]
    for start_page, section in PAGE_SECTION_MAP:
        if page >= start_page:
            best = section
        else:
            break
    return best


def detect_section(text: str, page: int = None) -> str:
    """
    Detect which eBay agreement section a chunk belongs to.

    Strategy (in priority order):
    1. Look for explicit section header in the chunk text
    2. If not found and page number is provided, use page → section map
    3. Fall back to 'General' only if both above fail
    """
    # Priority 1: explicit header in text
    from_header = detect_section_from_header(text)
    if from_header:
        return from_header

    # Priority 2: page number map
    if page is not None:
        return detect_section_from_page(page)

    return "General"


def word_count(text: str) -> int:
    """Return number of words in text."""
    return len(text.split())


def save_chunks_to_json(chunks, output_path: str):
    """
    Save LangChain Document chunks to JSON with metadata.
    Each entry: chunk_id, text, word_count, char_count, section, source_page

    Uses section propagation:
    - First tries to detect section from header text
    - Falls back to page-based mapping
    - Propagates the last known section to continuation chunks
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = []
    last_known_section = "Section 1 - Introduction"  # safe default

    for i, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        page = chunk.metadata.get("page", 0) + 1  # 1-indexed

        # Try header detection first
        section = detect_section_from_header(text)

        if section:
            # Found a header — update last known
            last_known_section = section
        else:
            # No header — use page map
            page_section = detect_section_from_page(page)

            # Page map is more reliable than propagation when available
            # but propagation wins if page map gives a generic result
            if page_section != "General":
                section = page_section
                last_known_section = page_section
            else:
                # Fall back to carrying forward the last known section
                section = last_known_section

        output.append({
            "chunk_id":    i,
            "text":        text,
            "word_count":  word_count(text),
            "char_count":  len(text),
            "section":     section,
            "source_page": page
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[Utils] Saved {len(output)} chunks → {output_path}")
    return output


def print_chunk_stats(chunks_data: list):
    """Print summary stats about chunks."""
    wc = [c["word_count"] for c in chunks_data]
    sections = {}
    for c in chunks_data:
        sections[c["section"]] = sections.get(c["section"], 0) + 1

    print("\n── Chunk Statistics ──")
    print(f"  Total chunks    : {len(chunks_data)}")
    print(f"  Min words       : {min(wc)}")
    print(f"  Max words       : {max(wc)}")
    print(f"  Avg words       : {sum(wc) // len(wc)}")
    print(f"  Sections found  : {len(sections)}")
    for s in sorted(sections):
        print(f"    → {s} ({sections[s]} chunks)")
    print("------------------------------------\n")


def format_source_chunks(source_docs: list) -> str:
    """Format retrieved source chunks for display in Streamlit."""
    formatted = []
    for i, doc in enumerate(source_docs):
        page = doc.metadata.get("page", 0) + 1
        section = detect_section(doc.page_content, page=page)
        preview = doc.page_content[:300].strip()
        formatted.append(
            f"**Source {i+1}** | Page {page} | {section}\n\n{preview}..."
        )
    return "\n\n---\n\n".join(formatted)