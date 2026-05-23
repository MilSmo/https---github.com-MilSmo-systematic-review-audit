import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple

import requests
from bs4 import BeautifulSoup

_log = logging.getLogger(__name__)


GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
INPUT_DIR = Path("scipdf")
OUTPUT_DIR = Path("output_grobid")

MIN_PARAGRAPH_LEN = 30
MERGE_SHORT_PARAGRAPHS_BELOW = 120

MIN_CHUNK_CHARS = 700
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP_PARAGRAPHS = 1

REQUEST_TIMEOUT = 300
SKIP_REFERENCES_IN_CHUNKS = True


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def normalize_section_name(section_name: str) -> str:
    section_name = normalize_inline_whitespace(section_name)
    section_name = re.sub(r"^\d+(\.\d+)*\.?\s*", "", section_name)
    return section_name.strip()


def classify_section(section_name: str) -> str:
    s = normalize_section_name(section_name).lower()

    if not s:
        return "other"

    if s in {"front", "title", "keywords"}:
        return "front"

    if "abstract" in s or "summary" in s:
        return "abstract"

    if "introduction" in s or "background" in s:
        return "introduction"

    if (
        "method" in s
        or "materials" in s
        or "methodology" in s
        or "study design" in s
        or "experimental" in s
        or "procedure" in s
    ):
        return "methods"

    if "result" in s or "finding" in s:
        return "results"

    if "discussion" in s:
        return "discussion"

    if "conclusion" in s:
        return "conclusion"

    if "limitation" in s or "strengths and limitations" in s:
        return "limitations"

    if "funding" in s or "financial support" in s or "source of funding" in s:
        return "funding"

    if (
        "conflict" in s
        or "competing interest" in s
        or "declaration" in s
        or "disclosure" in s
    ):
        return "conflict_of_interest"

    if "author contribution" in s or "authors' contribution" in s:
        return "author_contributions"

    if "acknowledgement" in s or "acknowledgment" in s:
        return "acknowledgements"

    if "reference" in s or "bibliography" in s:
        return "references"

    if "appendix" in s or "appendices" in s:
        return "appendix"

    if "supplement" in s:
        return "supplementary"

    if "registration" in s or "protocol" in s:
        return "registration_protocol"

    if "data availability" in s or "data sharing" in s:
        return "data_availability"

    return "other"


def get_section_weight(section_type: str) -> float:
    weights = {
        "front": 0.8,
        "abstract": 1.35,
        "introduction": 1.0,
        "methods": 1.25,
        "results": 1.3,
        "discussion": 1.15,
        "conclusion": 1.15,
        "limitations": 1.1,
        "funding": 0.85,
        "conflict_of_interest": 0.85,
        "author_contributions": 0.75,
        "acknowledgements": 0.65,
        "references": 0.2,
        "appendix": 0.8,
        "supplementary": 0.95,
        "registration_protocol": 1.0,
        "data_availability": 0.9,
        "other": 1.0,
    }
    return weights.get(section_type, 1.0)


def save_jsonl(records: List[Dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def grobid_parse_pdf(pdf_path: Path, tei_output_path: Path) -> str:
    with pdf_path.open("rb") as f:
        response = requests.post(
            GROBID_URL,
            files={"input": f},
            timeout=REQUEST_TIMEOUT,
        )

    response.raise_for_status()
    tei_xml = response.text
    tei_output_path.write_text(tei_xml, encoding="utf-8")
    return tei_xml


def extract_title(soup: BeautifulSoup, fallback: str) -> str:
    tag = soup.find("title", attrs={"type": "main"})
    if tag:
        title = normalize_inline_whitespace(tag.get_text(" ", strip=True))
        if title:
            return title

    title_stmt = soup.find("titleStmt")
    if title_stmt:
        tag = title_stmt.find("title")
        if tag:
            title = normalize_inline_whitespace(tag.get_text(" ", strip=True))
            if title:
                return title

    first_title = soup.find("title")
    if first_title:
        title = normalize_inline_whitespace(first_title.get_text(" ", strip=True))
        if title:
            return title

    return fallback


def extract_abstract_paragraphs(soup: BeautifulSoup) -> List[str]:
    abstract = soup.find("abstract")
    if not abstract:
        return []

    paragraphs = []

    for p in abstract.find_all("p"):
        txt = normalize_inline_whitespace(p.get_text(" ", strip=True))
        if txt and len(txt) >= MIN_PARAGRAPH_LEN:
            paragraphs.append(txt)

    if not paragraphs:
        txt = normalize_inline_whitespace(abstract.get_text(" ", strip=True))
        if txt and len(txt) >= MIN_PARAGRAPH_LEN:
            paragraphs.append(txt)

    return paragraphs


def extract_sections_from_container(container, default_name: str) -> List[Dict]:
    sections = []

    for div in container.find_all("div", recursive=False):
        head = div.find("head", recursive=False)
        section_name = normalize_section_name(
            head.get_text(" ", strip=True)
        ) if head else default_name

        paragraphs = []

        for p in div.find_all("p"):
            txt = normalize_inline_whitespace(p.get_text(" ", strip=True))
            if txt and len(txt) >= MIN_PARAGRAPH_LEN:
                paragraphs.append(txt)

        if paragraphs:
            section_type = classify_section(section_name)
            sections.append(
                {
                    "section": section_name,
                    "section_type": section_type,
                    "paragraphs": paragraphs,
                }
            )

        nested_divs = div.find_all("div", recursive=False)
        for nested in nested_divs:
            nested_head = nested.find("head", recursive=False)
            nested_name = normalize_section_name(
                nested_head.get_text(" ", strip=True)
            ) if nested_head else section_name

            nested_paragraphs = []
            for p in nested.find_all("p"):
                txt = normalize_inline_whitespace(p.get_text(" ", strip=True))
                if txt and len(txt) >= MIN_PARAGRAPH_LEN:
                    nested_paragraphs.append(txt)

            if nested_paragraphs:
                nested_type = classify_section(nested_name)
                sections.append(
                    {
                        "section": nested_name,
                        "section_type": nested_type,
                        "paragraphs": nested_paragraphs,
                    }
                )

    return sections


def extract_body_sections(soup: BeautifulSoup) -> List[Dict]:
    body = soup.find("body")
    if not body:
        return []

    sections = extract_sections_from_container(body, "Unknown")

    if sections:
        return sections

    paragraphs = []
    for p in body.find_all("p"):
        txt = normalize_inline_whitespace(p.get_text(" ", strip=True))
        if txt and len(txt) >= MIN_PARAGRAPH_LEN:
            paragraphs.append(txt)

    if paragraphs:
        return [
            {
                "section": "Body",
                "section_type": "other",
                "paragraphs": paragraphs,
            }
        ]

    return []


def extract_back_sections(soup: BeautifulSoup) -> List[Dict]:
    back = soup.find("back")
    if not back:
        return []

    sections = extract_sections_from_container(back, "Back matter")

    list_bibl = back.find("listBibl")
    if list_bibl:
        refs = []

        for bibl in list_bibl.find_all(["biblStruct", "bibl"]):
            txt = normalize_inline_whitespace(bibl.get_text(" ", strip=True))
            if txt:
                refs.append(txt)

        if refs:
            sections.append(
                {
                    "section": "References",
                    "section_type": "references",
                    "paragraphs": refs,
                }
            )

    return sections


def build_paragraph_records(
    doc_id: str,
    title: str,
    source_file: str,
    section_blocks: List[Dict],
) -> List[Dict]:
    paragraphs = []
    para_id = 0

    for block in section_blocks:
        section = block["section"]
        section_type = block["section_type"]

        for para in block["paragraphs"]:
            para_id += 1
            paragraphs.append(
                {
                    "paragraph_id": para_id,
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "title": title,
                    "section": section,
                    "section_type": section_type,
                    "section_weight": get_section_weight(section_type),
                    "text": para,
                }
            )

    return paragraphs


def merge_short_paragraphs(
    paragraphs: List[Dict],
    min_len: int = MERGE_SHORT_PARAGRAPHS_BELOW,
) -> List[Dict]:
    if not paragraphs:
        return []

    merged = []
    buffer = None
    new_para_id = 0

    for p in paragraphs:
        p = dict(p)

        if buffer is None:
            buffer = p
            continue

        same_section = (
            buffer["section"] == p["section"]
            and buffer["section_type"] == p["section_type"]
        )

        if len(buffer["text"]) < min_len and same_section:
            buffer["text"] = buffer["text"].strip() + "\n\n" + p["text"].strip()
        else:
            new_para_id += 1
            buffer["paragraph_id"] = new_para_id
            merged.append(buffer)
            buffer = p

    if buffer is not None:
        new_para_id += 1
        buffer["paragraph_id"] = new_para_id
        merged.append(buffer)

    return merged


def make_chunk(
    doc_id: str,
    chunk_num: int,
    title: str,
    source_file: str,
    section: str,
    section_type: str,
    section_weight: float,
    paragraph_ids: List[int],
    text: str,
) -> Dict:
    embedding_text = (
        f"Document title: {title}\n"
        f"Source file: {source_file}\n"
        f"Section heading: {section}\n"
        f"Section category: {section_type}\n\n"
        f"{text}"
    )

    return {
        "chunk_id": f"{doc_id}_chunk_{chunk_num}",
        "doc_id": doc_id,
        "source_file": source_file,
        "title": title,
        "section": section,
        "section_type": section_type,
        "section_weight": section_weight,
        "paragraph_ids": paragraph_ids,
        "text": text,
        "embedding_text": embedding_text,
        "char_count": len(text),
    }


def build_chunks(
    paragraphs: List[Dict],
    doc_id: str,
    title: str,
    source_file: str,
) -> List[Dict]:
    if not paragraphs:
        return []

    chunks = []
    chunk_id = 0
    sections_map: Dict[str, List[Dict]] = {}

    for p in paragraphs:
        sections_map.setdefault(p["section"], []).append(p)

    for section_name, sec_paragraphs in sections_map.items():
        if not sec_paragraphs:
            continue

        section_type = sec_paragraphs[0]["section_type"]

        if SKIP_REFERENCES_IN_CHUNKS and section_type == "references":
            continue

        section_weight = sec_paragraphs[0]["section_weight"]

        buffer_texts = []
        buffer_ids = []
        current_len = 0

        for p in sec_paragraphs:
            txt = p["text"].strip()

            should_flush = (
                buffer_texts
                and current_len + len(txt) > MAX_CHUNK_CHARS
                and current_len >= MIN_CHUNK_CHARS
            )

            if should_flush:
                chunk_id += 1
                chunk_text = "\n\n".join(buffer_texts).strip()

                chunks.append(
                    make_chunk(
                        doc_id=doc_id,
                        chunk_num=chunk_id,
                        title=title,
                        source_file=source_file,
                        section=section_name,
                        section_type=section_type,
                        section_weight=section_weight,
                        paragraph_ids=buffer_ids,
                        text=chunk_text,
                    )
                )

                if CHUNK_OVERLAP_PARAGRAPHS > 0:
                    buffer_texts = buffer_texts[-CHUNK_OVERLAP_PARAGRAPHS:]
                    buffer_ids = buffer_ids[-CHUNK_OVERLAP_PARAGRAPHS:]
                    current_len = sum(len(x) for x in buffer_texts)
                else:
                    buffer_texts = []
                    buffer_ids = []
                    current_len = 0

            buffer_texts.append(txt)
            buffer_ids.append(p["paragraph_id"])
            current_len += len(txt)

        if buffer_texts:
            chunk_id += 1
            chunk_text = "\n\n".join(buffer_texts).strip()

            chunks.append(
                make_chunk(
                    doc_id=doc_id,
                    chunk_num=chunk_id,
                    title=title,
                    source_file=source_file,
                    section=section_name,
                    section_type=section_type,
                    section_weight=section_weight,
                    paragraph_ids=buffer_ids,
                    text=chunk_text,
                )
            )

    return chunks


def process_single_pdf(pdf_path: Path, output_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    start_time = time.time()

    doc_id = pdf_path.stem
    doc_output_dir = output_dir / doc_id
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    tei_path = doc_output_dir / f"{doc_id}.tei.xml"
    tei_xml = grobid_parse_pdf(pdf_path, tei_path)

    soup = BeautifulSoup(tei_xml, "xml")
    title = extract_title(soup, fallback=doc_id)

    section_blocks = []

    abstract_paragraphs = extract_abstract_paragraphs(soup)
    if abstract_paragraphs:
        section_blocks.append(
            {
                "section": "Abstract",
                "section_type": "abstract",
                "paragraphs": abstract_paragraphs,
            }
        )

    section_blocks.extend(extract_body_sections(soup))
    section_blocks.extend(extract_back_sections(soup))

    paragraphs = build_paragraph_records(
        doc_id=doc_id,
        title=title,
        source_file=pdf_path.name,
        section_blocks=section_blocks,
    )

    paragraphs = merge_short_paragraphs(
        paragraphs,
        min_len=MERGE_SHORT_PARAGRAPHS_BELOW,
    )

    chunks = build_chunks(
        paragraphs=paragraphs,
        doc_id=doc_id,
        title=title,
        source_file=pdf_path.name,
    )

    save_jsonl(paragraphs, doc_output_dir / f"{doc_id}-paragraphs.jsonl")
    save_jsonl(chunks, doc_output_dir / f"{doc_id}-chunks.jsonl")

    elapsed = time.time() - start_time
    _log.info(
        "Done %s in %.2fs | paragraphs=%d | chunks=%d",
        pdf_path.name,
        elapsed,
        len(paragraphs),
        len(chunks),
    )

    return paragraphs, chunks


def main():
    logging.basicConfig(level=logging.INFO)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        _log.warning("No PDF files found in %s", INPUT_DIR.resolve())
        return

    all_paragraphs = []
    all_chunks = []

    global_start = time.time()

    for pdf_path in pdf_files:
        try:
            paragraphs, chunks = process_single_pdf(pdf_path, OUTPUT_DIR)
            all_paragraphs.extend(paragraphs)
            all_chunks.extend(chunks)
        except Exception:
            _log.exception("Failed processing %s", pdf_path.name)

    save_jsonl(all_paragraphs, OUTPUT_DIR / "all_paragraphs.jsonl")
    save_jsonl(all_chunks, OUTPUT_DIR / "all_chunks.jsonl")

    elapsed = time.time() - global_start
    _log.info(
        "Batch done in %.2fs | total_paragraphs=%d | total_chunks=%d",
        elapsed,
        len(all_paragraphs),
        len(all_chunks),
    )


if __name__ == "__main__":
    main()
    
    