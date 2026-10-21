"""
=========================================================
AI AGENT CONTEXT LOG FOR PERSON 2 (DATA ENGINEER)
=========================================================
Project: Synthexis (Multimodal RAG)
Your Scope: ONLY edit `extractors.py`. 
Task: Build robust PDF extraction handling both text and embedded charts/images.

I/O CONTRACT:
- Input: `file_path` (str)
- Output: dict -> {"text_chunks": [list of strings], "images": [list of image paths/bytes]}

Instructions for AI Agent:
1. Implement the extraction logic. Do not write server or UI code.
2. Ensure tables are extracted as markdown, and text reading order is preserved.
"""
"""
=============================================================================
SYNTHEXIS - MULTIMODAL RAG PIPELINE
PERSON 2: DATA INGESTION & DOCUMENT EXTRACTION ENGINE
=============================================================================
Module: extractors.py
Owner: Person 2 (Data Ingestion Engineer)

Responsibilities:
- Ingest complex PDF structures (multi-column text, tables, vector charts, images).
- Detect and isolate images, vector charts, and tables as visual crops (byte-streams / base64).
- Extract table contents as Markdown for hybrid search context.
- Segment document content using LangChain's RecursiveCharacterTextSplitter.

I/O Contract:
- Input: file_path (str or Path)
- Output: dict -> {
    "text_chunks": List[str],
    "images": List[str]  # Base64 strings (or byte streams via helper)
}
=============================================================================
"""

import os
import re
import io
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger("synthexis.extractors")

# ---------------------------------------------------------------------------
# Core Dependencies
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.error("PyMuPDF (fitz) is required for visual extraction. Install via: pip install PyMuPDF")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_SPLITTER_AVAILABLE = True
    except ImportError:
        LANGCHAIN_SPLITTER_AVAILABLE = False


# ===========================================================================
# 1. VISUAL EXTRACTION: DETECT & ISOLATE IMAGES, CHARTS & TABLES
# ===========================================================================

class VisualArtifact:
    """Represents an isolated visual element (image, chart, or table)."""
    def __init__(self, artifact_type: str, page_num: int, image_bytes: bytes, bbox: Optional[Tuple[float, float, float, float]] = None):
        self.artifact_type = artifact_type  # 'image', 'chart', 'table'
        self.page_num = page_num
        self.bytes = image_bytes
        self.bbox = bbox

    def to_base64(self, mime_type: str = "image/png") -> str:
        """Converts raw byte-stream to a base64 Data URI for LLM consumption."""
        b64 = base64.b64encode(self.bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def to_byte_stream(self) -> io.BytesIO:
        """Returns artifact as an in-memory byte-stream (BytesIO)."""
        return io.BytesIO(self.bytes)


def _merge_rects(rects: List[fitz.Rect], padding: float = 10.0) -> List[fitz.Rect]:
    """Merges overlapping or closely clustered bounding boxes (e.g., chart components)."""
    if not rects:
        return []

    merged = []
    for r in rects:
        expanded = fitz.Rect(r.x0 - padding, r.y0 - padding, r.x1 + padding, r.y1 + padding)
        matched = False
        for i, existing in enumerate(merged):
            if existing.intersects(expanded):
                merged[i] = existing | expanded
                matched = True
                break
        if not matched:
            merged.append(expanded)
    return merged


def extract_visual_artifacts(
    file_path: str,
    include_tables: bool = True,
    include_charts: bool = True,
    include_images: bool = True,
    dpi: int = 150
) -> List[VisualArtifact]:
    """
    Detects and isolates:
    1. Tables (via PyMuPDF table detection -> clipped into high-res images).
    2. Charts & Diagrams (via vector drawing clustering -> clipped into images).
    3. Embedded Raster Images (via PyMuPDF image stream extraction).
    """
    artifacts: List[VisualArtifact] = []

    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF unavailable. Cannot perform visual extraction.")
        return artifacts

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF for visual extraction {file_path}: {e}")
        return artifacts

    seen_hashes = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        # A. Isolate Tables Visually
        if include_tables and hasattr(page, "find_tables"):
            try:
                tables = page.find_tables()
                for tab in tables:
                    bbox = fitz.Rect(tab.bbox)
                    if bbox.width < 80 or bbox.height < 40:
                        continue
                    
                    pix = page.get_pixmap(clip=bbox, dpi=dpi)
                    img_bytes = pix.tobytes("png")

                    h = hash(img_bytes[:256])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        artifacts.append(VisualArtifact("table", page_num, img_bytes, tuple(bbox)))
            except Exception as e:
                logger.debug(f"Table visual isolation error on page {page_num}: {e}")

        # B. Detect & Isolate Vector Charts & Graphs
        if include_charts:
            try:
                drawings = page.get_drawings()
                candidate_rects = [
                    d["rect"] for d in drawings 
                    if d["rect"].width > 70 and d["rect"].height > 60
                ]
                
                chart_clusters = _merge_rects(candidate_rects, padding=12.0)
                for cluster_rect in chart_clusters:
                    safe_rect = cluster_rect & page.rect
                    if safe_rect.width < 100 or safe_rect.height < 80:
                        continue
                    
                    is_table_overlap = any(
                        safe_rect.intersects(fitz.Rect(art.bbox)) 
                        for art in artifacts if art.page_num == page_num and art.bbox
                    )
                    if is_table_overlap:
                        continue

                    pix = page.get_pixmap(clip=safe_rect, dpi=dpi)
                    img_bytes = pix.tobytes("png")

                    h = hash(img_bytes[:256])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        artifacts.append(VisualArtifact("chart", page_num, img_bytes, tuple(safe_rect)))
            except Exception as e:
                logger.debug(f"Chart visual isolation error on page {page_num}: {e}")

        # C. Extract Embedded Raster Images & Figures
        if include_images:
            try:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img.get("image")
                    w = base_img.get("width", 0)
                    h = base_img.get("height", 0)

                    if w < 100 or h < 100:
                        continue

                    img_hash = hash(img_bytes[:256])
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    artifacts.append(VisualArtifact("image", page_num, img_bytes))
            except Exception as e:
                logger.debug(f"Raster image extraction error on page {page_num}: {e}")

    doc.close()
    return artifacts


# ===========================================================================
# 2. TABLE PARSING (MARKDOWN TEXT FOR HYBRID RETRIEVAL)
# ===========================================================================

def format_table_as_markdown(table_rows: List[List[Optional[str]]]) -> str:
    """Converts 2D table grid into GitHub-style Markdown."""
    if not table_rows:
        return ""
    cleaned = []
    for row in table_rows:
        cleaned_row = [(str(cell).strip().replace("\n", " ") if cell is not None else "") for cell in row]
        if any(cleaned_row):
            cleaned.append(cleaned_row)
    if not cleaned:
        return ""
    max_cols = max(len(r) for r in cleaned)
    padded = [r + [""] * (max_cols - len(r)) for r in cleaned]
    header = padded[0]
    separator = ["---"] * max_cols
    body = padded[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n" + "\n".join(lines) + "\n"


def extract_tables_as_markdown(file_path: str) -> Dict[int, List[str]]:
    """Extracts tables as Markdown text per page."""
    tables_by_page: Dict[int, List[str]] = {}
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    md_list = []
                    for t in page.extract_tables():
                        md = format_table_as_markdown(t)
                        if md.strip():
                            md_list.append(md)
                    if md_list:
                        tables_by_page[idx + 1] = md_list
            return tables_by_page
        except Exception:
            pass

    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(file_path)
            for idx, page in enumerate(doc):
                if hasattr(page, "find_tables"):
                    md_list = []
                    for tab in page.find_tables():
                        md = format_table_as_markdown(tab.extract())
                        if md.strip():
                            md_list.append(md)
                    if md_list:
                        tables_by_page[idx + 1] = md_list
            doc.close()
        except Exception:
            pass
    return tables_by_page


# ===========================================================================
# 3. TEXT CHUNKING (LANGCHAIN RECURSIVE TEXT SPLITTER)
# ===========================================================================

def clean_text(text: str) -> str:
    """Removes noise, cleans hyphenation and normalizes whitespace."""
    if not text:
        return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_document_text(
    pages_data: List[Tuple[int, str]],
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[str]:
    """Chunks text preserving citation tags: [filename | Page N]."""
    if LANGCHAIN_SPLITTER_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:
        splitter = None

    chunks = []
    for page_num, content in pages_data:
        if not content.strip():
            continue
        if splitter:
            splits = splitter.split_text(content)
        else:
            splits = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size - chunk_overlap)]

        for s in splits:
            s_clean = s.strip()
            if len(s_clean) > 30:
                chunks.append(f"[{filename} | Page {page_num}]\n{s_clean}")
    return chunks


# ===========================================================================
# 4. PRIMARY INGESTION CONTRACT
# ===========================================================================

def process_document(
    file_path: str,
    return_bytes: bool = False
) -> Dict[str, Any]:
    """
    Person 2 Primary Ingestion Function.

    Parameters:
    - file_path: Path to target PDF document.
    - return_bytes: If True, returns images as raw byte-streams (bytes).
                    If False (default), returns base64 data URIs for direct LLM/FastAPI usage.

    Returns:
    - dict -> {
        "text_chunks": List[str],
        "images": List[Union[str, bytes]]
      }
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        logger.error(f"Document not found: {file_path}")
        return {"text_chunks": [], "images": []}

    filename = path_obj.name
    logger.info(f"Ingesting '{filename}' with visual chart, table, and image extraction.")

    # 1. Extract tabular markdown
    tables_by_page = extract_tables_as_markdown(str(path_obj))

    # 2. Extract plain text page-by-page and inject table markdown
    pages_data: List[Tuple[int, str]] = []
    if PYMUPDF_AVAILABLE:
        doc = fitz.open(str(path_obj))
        for idx, page in enumerate(doc):
            p_num = idx + 1
            raw_text = clean_text(page.get_text("text") or "")
            if p_num in tables_by_page:
                raw_text += "\n\n### Extracted Table:\n" + "\n".join(tables_by_page[p_num])
            if raw_text.strip():
                pages_data.append((p_num, raw_text))
        doc.close()

    # 3. Text chunking using LangChain
    text_chunks = chunk_document_text(pages_data, filename=filename)

    # 4. Visual artifact detection: isolate tables, vector charts, and images
    visual_artifacts = extract_visual_artifacts(
        file_path=str(path_obj),
        include_tables=True,
        include_charts=True,
        include_images=True,
        dpi=150
    )

    # Format images according to desired delivery (bytes vs base64 strings)
    if return_bytes:
        formatted_images = [art.bytes for art in visual_artifacts]
    else:
        formatted_images = [art.to_base64() for art in visual_artifacts]

    logger.info(
        f"Completed '{filename}': {len(text_chunks)} chunks, "
        f"{len(formatted_images)} visual items (tables/charts/images) extracted."
    )

    return {
        "text_chunks": text_chunks,
        "images": formatted_images
    }
