from __future__ import annotations

import argparse
import base64
import bisect
import json
import mimetypes
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Optional, Set, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import PictureItem, SectionHeaderItem
from docling_core.types.doc.labels import DocItemLabel
from openai import OpenAI

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

try:  # optional across docling-core versions
    from docling_core.types.doc import TableItem
except ImportError:  # pragma: no cover
    TableItem = None


@dataclass
class RetrieverConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    initial_top_k: int = 50
    final_top_k: int = 10
    max_query_keywords: int = 12


class PrismaFaissRetriever:
    def __init__(self, store_dir: str | Path, config: Optional[RetrieverConfig] = None):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or RetrieverConfig()
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedder = None
        self.reranker = None

    def build(self, chunks_path: str | Path) -> None:
        chunks = self._load_chunks(Path(chunks_path))
        if not chunks:
            raise ValueError(f"No chunks found in {chunks_path}")

        self._load_embedder()

        texts = [self._get_embedding_text(c) for c in chunks]

        embeddings = self.embedder.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = [
            self._metadata_from_chunk(i, c)
            for i, c in enumerate(chunks)
        ]

        self.save()

    def query(
        self,
        item: Dict[str, Any],
        initial_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:

        self._ensure_index_loaded()
        self._load_embedder()
        self._load_reranker()

        initial_top_k = initial_top_k or self.config.initial_top_k
        final_top_k = final_top_k or self.config.final_top_k

        # semantic retrieval query -> FAISS
        semantic_query = self.build_dense_query_text(item)

        retrieved = self._retrieve(
            item=item,
            query_text=semantic_query,
            top_k=initial_top_k,
        )

        if debug:
            print("\nSEMANTIC QUERY USED FOR FAISS:")
            print(semantic_query)

            print("\nTOP BEFORE RERANK:")
            for r in retrieved[:15]:
                c = r["chunk"]
                print(
                    r["rank"],
                    "faiss=", round(r["faiss_score"], 4),
                    "adjusted=", round(r["adjusted_retrieval_score"], 4),
                    "section_bonus=", round(r["section_bonus"], 4),
                    "keyword_bonus=", round(r["keyword_bonus"], 4),
                    c.get("section", ""),
                    c.get("text", "")[:220].replace("\n", " "),
                )

        reranked = self._rerank(
            query_text=semantic_query,
            retrieved=retrieved,
            final_top_k=final_top_k,
        )

        return self._make_evidence_payload(
            item=item,
            query_text=semantic_query,
            reranked=reranked,
        )

    def build_dense_query_text(self, item: Dict[str, Any]) -> str:
        """
        Build the dense retrieval query from the semantic description,
        checklist wording, and selected high-value terminology.

        The complete keyword list is still used separately for lexical boosting.
        """
        parts = [
            item.get("semantic_query", ""),
            item.get("retrieval_query", ""),
            item.get("retrieval_query_dense", ""),
            item.get("checklist_item", ""),
        ]

        if not any(
            isinstance(part, str) and part.strip()
            for part in parts
        ):
            parts.append(item.get("query_template", ""))

        keywords = [
            str(keyword).strip()
            for keyword in item.get("keywords", [])
            if str(keyword).strip()
        ]

        ranked_keywords = sorted(
            dict.fromkeys(keywords),
            key=lambda value: (
                " " not in value,
                -len(value),
            ),
        )

        selected_keywords = ranked_keywords[
            : self.config.max_query_keywords
        ]

        if selected_keywords:
            parts.append(
                "Relevant terminology: "
                + "; ".join(selected_keywords)
            )

        return "\n".join(
            part.strip()
            for part in parts
            if isinstance(part, str) and part.strip()
        )

    def _retrieve(
        self,
        item: Dict[str, Any],
        query_text: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:

        query_vec = self.embedder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        ).astype("float32")

        safe_top_k = min(top_k, len(self.metadata))
        scores, indices = self.index.search(query_vec, safe_top_k)

        # keywords -> lexical boost
        expanded_keywords = self._expand_keywords(item)

        # sections -> metadata boost
        requested_sections = item.get("sections_to_search") or []

        candidates = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue

            chunk = self.metadata[idx]
            text = chunk.get("text", "").strip()
            image_captions = " ".join(
                str(image.get("caption", "")).strip()
                for image in chunk.get("images", [])
                if isinstance(image, dict)
            ).strip()
            body_text = " ".join(
                part for part in (text, image_captions) if part
            )
            section = chunk.get("section", "")
            section_type = self._section_to_type(section)

            # Headings are matched lexically but do not count towards the
            # minimum body length, so a heading alone can never keep a
            # near-empty chunk alive.
            searchable_text = " ".join(
                part
                for part in (
                    body_text,
                    str(chunk.get("heading_path", "")),
                    " ".join(chunk.get("merged_headings", []) or []),
                )
                if part
            )

            if section_type in {
                "references",
                "acknowledgements",
                "author_contributions",
            }:
                continue

            if len(body_text) < 40:
                continue

            keyword_bonus = self._keyword_overlap_bonus(
                text=searchable_text,
                keywords=expanded_keywords,
            )

            section_bonus = self._section_match_bonus(
                chunk_section=" ".join(
                    part
                    for part in (
                        section,
                        str(chunk.get("heading_path", "")),
                        " ".join(chunk.get("merged_headings", []) or []),
                    )
                    if part
                ),
                requested_sections=requested_sections,
            )

            adjusted_score = float(score) + keyword_bonus + section_bonus

            candidates.append(
                {
                    "rank": rank,
                    "faiss_score": float(score),
                    "keyword_bonus": keyword_bonus,
                    "section_bonus": section_bonus,
                    "adjusted_retrieval_score": adjusted_score,
                    "chunk": chunk,
                }
            )

        candidates.sort(
            key=lambda x: x["adjusted_retrieval_score"],
            reverse=True,
        )

        return candidates

    def _rerank(
        self,
        query_text: str,
        retrieved: List[Dict[str, Any]],
        final_top_k: int,
    ) -> List[Dict[str, Any]]:

        if not retrieved:
            return []

        pairs = [
            (query_text, self._build_rerank_text(r["chunk"]))
            for r in retrieved
        ]

        rerank_scores = self.reranker.predict(pairs)

        reranked = []
        for row, score in zip(retrieved, rerank_scores):
            out = dict(row)
            out["rerank_score"] = float(score)
            reranked.append(out)

        self._minmax_normalize_scores(reranked, "adjusted_retrieval_score")
        self._minmax_normalize_scores(reranked, "rerank_score")

        for r in reranked:
            r["final_score"] = (
                0.65 * r["adjusted_retrieval_score_norm"]
                + 0.35 * r["rerank_score_norm"]
            )

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:final_top_k]

    @staticmethod
    def _chunk_context_text(chunk: Dict[str, Any]) -> str:
        """
        Single representation used for both embedding and reranking.

        The heading path and any merged parent headings are included so that
        a section nested under a heading that carries no body text of its own
        is still retrievable through that heading.
        """
        image_captions = [
            str(image.get("caption", "")).strip()
            for image in chunk.get("images", [])
            if isinstance(image, dict)
            and str(image.get("caption", "")).strip()
        ]

        parts = [f"Section: {chunk.get('section', '')}"]

        heading_path = str(chunk.get("heading_path", "") or "").strip()
        if heading_path and heading_path != chunk.get("section", ""):
            parts.append(f"Section path: {heading_path}")

        merged_headings = [
            str(heading).strip()
            for heading in chunk.get("merged_headings", []) or []
            if str(heading).strip()
        ]
        if merged_headings:
            parts.append(
                "Also covers: " + "; ".join(merged_headings)
            )

        parts.append(f"Type: {chunk.get('type', '')}")
        parts.append(f"Text: {chunk.get('text', '')}")

        if image_captions:
            parts.append(
                "Attached image captions: "
                + " | ".join(image_captions)
            )

        return "\n".join(parts).strip()

    @classmethod
    def _build_rerank_text(cls, chunk: Dict[str, Any]) -> str:
        return cls._chunk_context_text(chunk)

    @classmethod
    def _get_embedding_text(cls, chunk: Dict[str, Any]) -> str:
        return cls._chunk_context_text(chunk)

    def _make_evidence_payload(
        self,
        item: Dict[str, Any],
        query_text: str,
        reranked: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        return {
            "item_id": item.get("item_id"),
            "item_label": item.get("item_label"),
            "checklist_item": item.get("checklist_item"),
            "semantic_query": query_text,
            "top_evidence": [
                {
                    "chunk_id": r["chunk"].get("chunk_id"),
                    "document_title": r["chunk"].get("document_title"),
                    "type": r["chunk"].get("type"),
                    "section": r["chunk"].get("section"),
                    "merged_headings": r["chunk"].get("merged_headings", []),
                    "pages": r["chunk"].get("pages", []),
                    "word_count": r["chunk"].get("word_count"),
                    "faiss_score": r["faiss_score"],
                    "keyword_bonus": r["keyword_bonus"],
                    "section_bonus": r["section_bonus"],
                    "adjusted_retrieval_score": r["adjusted_retrieval_score"],
                    "adjusted_retrieval_score_norm": r.get(
                        "adjusted_retrieval_score_norm"
                    ),
                    "rerank_score": r.get("rerank_score"),
                    "rerank_score_norm": r.get("rerank_score_norm"),
                    "final_score": r.get("final_score"),
                    "text": r["chunk"].get("text"),
                    "images": r["chunk"].get("images", []),
                }
                for r in reranked
            ],
        }

    def load(self) -> None:
        index_path = self.store_dir / "chunks.index"
        metadata_path = self.store_dir / "chunks_metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = self._load_json(metadata_path)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No FAISS index to save.")
        if not self.metadata:
            raise ValueError("No metadata to save.")

        faiss.write_index(self.index, str(self.store_dir / "chunks.index"))
        self._save_json(self.store_dir / "chunks_metadata.json", self.metadata)
        self._save_json(self.store_dir / "retriever_config.json", asdict(self.config))

    @staticmethod
    def _metadata_from_chunk(
        faiss_id: int,
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "faiss_id": faiss_id,
            "chunk_id": chunk.get("chunk_id", f"chunk_{faiss_id}"),
            "document_title": chunk.get("document_title", ""),
            "type": chunk.get("type", "text"),
            "section": chunk.get("section", ""),
            "heading": chunk.get("heading", ""),
            "heading_path": chunk.get("heading_path", ""),
            "merged_headings": chunk.get("merged_headings", []),
            "section_index": chunk.get("section_index"),
            "section_occurrence": chunk.get("section_occurrence", 0),
            "pages": chunk.get("pages", []),
            "text": chunk.get("text", ""),
            "word_count": chunk.get("word_count", 0),
            "images": chunk.get("images", []),
        }

    def _load_embedder(self) -> None:
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.config.embedding_model_name)

    def _load_reranker(self) -> None:
        if self.reranker is None:
            self.reranker = CrossEncoder(self.config.reranker_model_name)

    def _ensure_index_loaded(self) -> None:
        if self.index is None or not self.metadata:
            self.load()

    @staticmethod
    def _load_chunks(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path.resolve()}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "chunks" in data:
            data = data["chunks"]

        if not isinstance(data, list):
            raise ValueError("Expected JSON array or object with 'chunks' key.")

        return data

    @staticmethod
    def _load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json(path: Path, obj: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _expand_keywords(item: Dict[str, Any]) -> List[str]:
        terms: Set[str] = set()

        for kw in item.get("keywords") or []:
            kw_l = str(kw).lower().strip()
            if not kw_l:
                continue

            terms.add(kw_l)
            terms.add(kw_l.replace("-", " "))
            terms.add(kw_l.replace(" ", "-"))

            if kw_l.endswith("y"):
                terms.add(kw_l[:-1] + "ies")

            if not kw_l.endswith("s"):
                terms.add(kw_l + "s")

        return sorted(t for t in terms if len(t) > 2)

    @staticmethod
    def _keyword_overlap_bonus(text: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0

        text_l = f" {text.lower()} "
        bonus = 0.0

        for kw in keywords:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue

            pattern = r"\b" + re.escape(kw_l) + r"\b"

            if re.search(pattern, text_l):
                bonus += 0.08 if " " in kw_l else 0.03

        return min(0.5, bonus)

    @staticmethod
    def _section_match_bonus(
        chunk_section: str,
        requested_sections: List[str],
    ) -> float:

        if not requested_sections:
            return 0.0

        section_text = chunk_section.lower()

        aliases = {
            "abstract": ["abstract", "summary"],
            "structured abstract": ["abstract", "structured abstract", "summary"],
            "summary": ["abstract", "summary"],

            "introduction": ["introduction", "background"],
            "background": ["background", "introduction"],
            "rationale": ["rationale", "introduction", "background"],

            "methods": [
                "methods",
                "methodology",
                "materials and methods",
                "evidence acquisition",
            ],
            "search strategy": ["search strategy", "search"],

            "study selection": [
                "study selection",
                "inclusion",
                "exclusion",
                "eligibility",
                "inclusion and exclusion criteria",
            ],
            "eligibility": [
                "eligibility",
                "inclusion",
                "exclusion",
                "inclusion and exclusion criteria",
            ],

            "risk of bias": [
                "risk of bias",
                "quality assessment",
                "methodological quality",
            ],

            "results": ["results", "evidence synthesis"],
            "synthesis": ["synthesis", "evidence synthesis"],

            "discussion": ["discussion"],
            "limitations": ["limitations", "limitations of the study"],
            "conclusion": ["conclusion", "conclusions"],
        }

        expanded = set()

        for sec in requested_sections:
            sec_l = sec.strip().lower()
            expanded.add(sec_l)
            expanded.update(aliases.get(sec_l, []))

        if any(term in section_text for term in expanded):
            return 0.15

        return 0.0

    @staticmethod
    def _section_to_type(section: str) -> str:
        return section.lower().strip().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _minmax_normalize_scores(rows: List[Dict[str, Any]], key: str) -> None:
        values = [float(r[key]) for r in rows]

        mn = min(values)
        mx = max(values)

        for r in rows:
            if mx == mn:
                r[f"{key}_norm"] = 0.0
            else:
                r[f"{key}_norm"] = (float(r[key]) - mn) / (mx - mn)

# ---------------------------------------------------------------------------
# PDF parsing and chunking
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_heading(heading: str) -> str:
    heading = heading.strip().replace("\n", " ")
    heading = re.sub(r"^#+\s*", "", heading)
    heading = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", heading)

    if re.fullmatch(r"(?:[A-Za-z]\s+)+[A-Za-z]", heading):
        heading = heading.replace(" ", "")

    heading = clean_text(heading)
    fixes = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "background": "Background",
        "methods": "Methods",
        "method": "Methods",
        "materials and methods": "Materials and Methods",
        "material and methods": "Materials and Methods",
        "results": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "references": "References",
        "acknowledgements": "Acknowledgements",
        "acknowledgments": "Acknowledgements",
        "funding": "Funding",
        "conflict of interest": "Conflict of Interest",
        "conflicts of interest": "Conflict of Interest",
    }
    return fixes.get(heading.lower(), heading.title())


def fix_abstract_intro_order(markdown: str) -> str:
    abstract_match = re.search(
        r"##\s*A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*\n(?P<abstract>.*?)(?=\n##|\Z)",
        markdown,
        flags=re.I | re.S,
    )
    intro_match = re.search(r"##\s*\d*\.?\s*Introduction\s*", markdown, flags=re.I)

    if not abstract_match or not intro_match or intro_match.start() > abstract_match.start():
        return markdown

    abstract_block = abstract_match.group(0).strip()
    markdown_without_abstract = (
        markdown[:abstract_match.start()] + markdown[abstract_match.end():]
    ).strip()
    abstract_block = re.sub(
        r"##\s*A\s*B\s*S\s*T\s*R\s*A\s*C\s*T",
        "## Abstract",
        abstract_block,
        flags=re.I,
    )
    intro_match_2 = re.search(
        r"##\s*\d*\.?\s*Introduction\s*",
        markdown_without_abstract,
        flags=re.I,
    )
    if not intro_match_2:
        return markdown

    before_intro = markdown_without_abstract[:intro_match_2.start()].strip()
    intro_and_after = markdown_without_abstract[intro_match_2.start():].strip()
    return f"{before_intro}\n\n{abstract_block}\n\n{intro_and_after}".strip()


# ---------------------------------------------------------------------------
# Layout-aware parsing
#
# export_to_markdown() linearises the page, so on a multi-column journal
# article the marginal rail, the boxed "key messages" panel, and the body
# columns are interleaved into one stream. The functions below rebuild the
# reading order from Docling provenance boxes instead:
#
#   1. columns are detected per page from the vertical whitespace gutters
#   2. items spanning several columns split the page into horizontal bands
#   3. bands are read column by column, boxes are emitted as their own chunks
#   4. a narrow metadata rail (correspondence, DOI, received/accepted) is
#      routed to a separate front matter chunk instead of the body flow
#   5. headings are tracked on a stack, so a heading directly followed by
#      another heading is preserved instead of being dropped
# ---------------------------------------------------------------------------

@dataclass
class LayoutConfig:
    # Minimum gutter width, as a fraction of page width, that separates columns.
    column_gap_ratio: float = 0.025
    # Items wider than this fraction of the page are ignored when locating
    # gutters, otherwise a full-width figure would bridge every column.
    narrow_item_ratio: float = 0.62
    # Fraction of a column an item must cover to count as occupying it.
    span_overlap_ratio: float = 0.35
    # A marginal rail must be narrower than this fraction of the median column.
    marginalia_width_ratio: float = 0.88
    # ... and narrower than this fraction of the page.
    marginalia_max_page_ratio: float = 0.26
    # Vertical gap, as a fraction of page height, that ends a spanning block.
    span_group_gap_ratio: float = 0.03
    min_column_width_ratio: float = 0.06
    max_columns: int = 6
    promote_caps_headings: bool = True
    detect_marginalia: bool = True
    detect_callouts: bool = True


MARGINALIA_PATTERNS = re.compile(
    r"(correspondence to|cite this as|for numbered affiliation"
    r"|supplemental material|https?://doi\.org|\bdoi\s*:"
    r"|(?:academic\s+|handling\s+)?editor\s*:"
    r"|received\s*:|revised\s*:|accepted\s*:|published\s*:"
    r"|open access|check for updates"
    r"|ÄÂÄš \s*\d{4}|creative commons)",
    re.I,
)

# Editorial metadata is not always placed in a narrow side rail by Docling.
# In some journal layouts it has the same width as a body column, so geometry
# alone cannot keep it out of the active section (for example, Abstract).
FRONT_MATTER_START_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:academic\s+|handling\s+)?editor|"
    r"received|revised|accepted|published|"
    r"citation|copyright|doi"
    r")\s*:",
    re.I,
)

CALLOUT_HEADING_PATTERNS = re.compile(
    r"^\s*("
    r"what (is|this|was) [\w\s]+|"
    r"how this study might affect[\w\s,/&\-]*|"
    r"key (messages|points|findings)|"
    r"summary (box|points)|"
    r"(box|panel)\s*\d*\s*[:.\-]?.*|"
    r"research in context|"
    r"highlights|"
    r"strengths and limitations[\w\s]*"
    r")\s*$",
    re.I,
)

CAPS_HEADING_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9 \-/&,:'()\.]{2,69}$")

SKIP_LABELS = {"page_header", "page_footer"}

HEADING_LABELS = {"section_header", "title"}

TEXT_LABELS = {
    "text",
    "paragraph",
    "list_item",
    "caption",
    "footnote",
    "formula",
    "code",
    "checkbox_selected",
    "checkbox_unselected",
}


def _label_of(element: Any) -> str:
    label = getattr(element, "label", "")
    value = getattr(label, "value", label)
    return str(value).lower()

TITLE_METADATA_PATTERN = re.compile(
    r"(?:https?://|www\.|\bdoi\b|\bissn\b|\bcopyright\b|"
    r"\breceived\b|\baccepted\b|\bpublished\b|\beditor\b|"
    r"\bvolume\b|\bissue\b|\bcorrespondence\b|"
    r"\bauthor affiliations?\b|\bopen access\b|@)",
    re.I,
)

TITLE_STOP_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materialsandmethods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
}


def _is_plausible_title(text: str) -> bool:
    text = clean_text(text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    words = text.split()
    compact = re.sub(r"[^a-z]", "", text.lower())

    if not 3 <= len(words) <= 80:
        return False
    if not 15 <= len(text) <= 600:
        return False
    if compact in TITLE_STOP_HEADINGS:
        return False
    if TITLE_METADATA_PATTERN.search(text):
        return False
    if not any(character.isalpha() for character in text):
        return False

    return True


def _clean_document_title(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text.strip(" |").strip()


def _title_item_geometry(
    element: Any,
    doc: Any,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Return page number, top ratio and width ratio for a Docling item."""
    provenance = list(getattr(element, "prov", None) or [])
    if not provenance:
        return None, None, None

    provenance = provenance[0]
    page_no = getattr(provenance, "page_no", None)
    try:
        page_no = int(page_no) if page_no is not None else None
    except (TypeError, ValueError):
        page_no = None

    page_width, page_height = _page_size(doc, page_no)
    bbox = getattr(provenance, "bbox", None)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return page_no, None, None

    coordinates = _bbox_top_left(bbox, page_height)
    if coordinates is None:
        return page_no, None, None

    left, top, right, _bottom = coordinates
    return page_no, top / page_height, (right - left) / page_width


def extract_document_title(
    doc: Any,
    markdown: str = "",
) -> str:
    """Extract an article title using Docling labels, then safe fallbacks."""
    candidates: Dict[str, Tuple[float, int, str, str]] = {}

    def add_candidate(
        text: str,
        source_score: float,
        order: int,
        source: str,
        top: Optional[float] = None,
        width: Optional[float] = None,
    ) -> None:
        text = _clean_document_title(text)
        if not _is_plausible_title(text):
            return

        score = source_score
        word_count = len(text.split())

        if top is not None:
            if 0.06 <= top <= 0.50:
                score += 30
            elif top < 0.06:
                score -= 20
            elif top > 0.65:
                score -= 35

        if width is not None:
            if width >= 0.50:
                score += 15
            elif width < 0.20:
                score -= 10

        if 6 <= word_count <= 30:
            score += 15
        elif word_count > 45:
            score -= 20

        if text.count(",") >= 4 and word_count <= 35:
            score -= 25
        if re.search(r"\bet\s+al\.?\b", text, re.I):
            score -= 40

        key = text.casefold()
        candidate = (score, -order, text, source)
        if key not in candidates or candidate > candidates[key]:
            candidates[key] = candidate

    # 1. Native Docling TitleItem / DocItemLabel.TITLE candidates.
    title_group: List[Tuple[int, Any, str]] = []

    def flush_title_group() -> None:
        if not title_group:
            return

        text = " ".join(part[2] for part in title_group)
        first_order, first_item, _first_text = title_group[0]
        _page, top, width = _title_item_geometry(first_item, doc)
        add_candidate(
            text,
            source_score=150,
            order=first_order,
            source="docling_title",
            top=top,
            width=width,
        )
        title_group.clear()

    for order, item in enumerate(getattr(doc, "texts", []) or []):
        label = getattr(item, "label", None)
        is_title = label == DocItemLabel.TITLE or _label_of(item) == "title"
        text = _clean_document_title(str(getattr(item, "text", "") or ""))
        page_no, _top, _width = _title_item_geometry(item, doc)

        if is_title and text and page_no in {None, 1}:
            title_group.append((order, item, text))
        else:
            flush_title_group()

    flush_title_group()

    # 2. Docling Markdown may serialize the article title as H1 or H2.
    for match in re.finditer(r"(?m)^(#{1,2})\s+(.+?)\s*$", markdown):
        level = len(match.group(1))
        add_candidate(
            match.group(2),
            source_score=120 if level == 1 else 95,
            order=match.start(),
            source=f"markdown_h{level}",
        )

    # 3. Page-one layout fallback when Docling missed the TITLE label.
    for order, (element, _level) in enumerate(doc.iterate_items()):
        if order >= 120:
            break

        label = _label_of(element)
        if label in SKIP_LABELS:
            continue

        text = _clean_document_title(
            str(getattr(element, "text", "") or "")
        )
        if not text:
            continue

        page_no, top, width = _title_item_geometry(element, doc)
        if page_no is not None and page_no != 1:
            continue
        if label not in {"title", "section_header"} and top is not None and top > 0.55:
            continue

        source_score = {
            "title": 140,
            "section_header": 75,
            "text": 20,
            "paragraph": 20,
        }.get(label, 15)

        add_candidate(
            text,
            source_score=source_score,
            order=order,
            source=f"docling_{label}",
            top=top,
            width=width,
        )

    if candidates:
        ranked = sorted(candidates.values(), reverse=True)
        score, _negative_order, title, source = ranked[0]

        if os.environ.get("DEBUG_TITLE_EXTRACTION") == "1":
            print("\nTitle candidates:")
            for candidate_score, _order, candidate_title, candidate_source in ranked[:10]:
                print(
                    f"  {candidate_score:6.1f} | "
                    f"{candidate_source:24s} | {candidate_title}"
                )
            print(f"Selected title ({source}, score={score:.1f}): {title}")

        return title

    raise ValueError(
        "Could not extract the article title from the document. "
        "No filename fallback is used."
    )


def _kind_of(element: Any, label: str) -> Optional[str]:
    if isinstance(element, PictureItem):
        return "picture"
    if TableItem is not None and isinstance(element, TableItem):
        return "table"
    if isinstance(element, SectionHeaderItem) or label in HEADING_LABELS:
        return "heading"
    if label in TEXT_LABELS:
        return "text"
    if getattr(element, "text", None):
        return "text"
    return None


def _page_size(doc: Any, page_no: Optional[int]) -> Tuple[float, float]:
    pages = getattr(doc, "pages", None) or {}
    page = None

    if isinstance(pages, dict):
        page = pages.get(page_no)
    elif isinstance(pages, (list, tuple)) and page_no:
        index = page_no - 1
        if 0 <= index < len(pages):
            page = pages[index]

    size = getattr(page, "size", None)

    try:
        width = float(getattr(size, "width", 0) or 0)
        height = float(getattr(size, "height", 0) or 0)
    except (TypeError, ValueError):
        width, height = 0.0, 0.0

    return width, height


def _bbox_top_left(bbox: Any, page_height: float) -> Optional[Tuple[float, float, float, float]]:
    try:
        left = float(bbox.l)
        right = float(bbox.r)
        top_raw = float(bbox.t)
        bottom_raw = float(bbox.b)
    except (AttributeError, TypeError, ValueError):
        return None

    origin = str(getattr(bbox, "coord_origin", "")).upper()

    if "BOTTOM" in origin and page_height > 0:
        top = page_height - max(top_raw, bottom_raw)
        bottom = page_height - min(top_raw, bottom_raw)
    else:
        top = min(top_raw, bottom_raw)
        bottom = max(top_raw, bottom_raw)

    return min(left, right), top, max(left, right), bottom


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _detect_columns(
    spans: List[Tuple[float, float]],
    page_width: float,
    cfg: LayoutConfig,
) -> List[Tuple[float, float]]:
    if not spans or page_width <= 0:
        return [(0.0, max(page_width, 1.0))]

    merged: List[List[float]] = []
    for x0, x1 in sorted(spans):
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    min_gap = max(cfg.column_gap_ratio * page_width, 1.0)

    columns: List[List[float]] = []
    for x0, x1 in merged:
        if columns and (x0 - columns[-1][1]) < min_gap:
            columns[-1][1] = max(columns[-1][1], x1)
        else:
            columns.append([x0, x1])

    # Drop slivers (footnote markers, drop caps) into the nearest neighbour.
    min_width = cfg.min_column_width_ratio * page_width
    cleaned: List[List[float]] = []
    for column in columns:
        if cleaned and (column[1] - column[0]) < min_width:
            cleaned[-1][1] = max(cleaned[-1][1], column[1])
        else:
            cleaned.append(column)

    if len(cleaned) > cfg.max_columns:
        return [(cleaned[0][0], cleaned[-1][1])]

    return [(c[0], c[1]) for c in cleaned] or [(0.0, page_width)]


def _marginalia_columns(
    columns: List[Tuple[float, float]],
    entries: List[Dict[str, Any]],
    page_width: float,
    median_column_width: float,
    cfg: LayoutConfig,
) -> Set[int]:
    if not cfg.detect_marginalia or len(columns) < 2:
        return set()

    marginal: Set[int] = set()

    for index in {0, len(columns) - 1}:
        c0, c1 = columns[index]
        width = c1 - c0

        if width > cfg.marginalia_max_page_ratio * page_width:
            continue
        if median_column_width and width >= cfg.marginalia_width_ratio * median_column_width:
            continue

        column_text = " ".join(
            entry["text"]
            for entry in entries
            if entry["text"]
            and _overlap(entry["x0"], entry["x1"], c0, c1)
            > 0.5 * max(entry["x1"] - entry["x0"], 1e-6)
        )

        if MARGINALIA_PATTERNS.search(column_text):
            marginal.add(index)

    return marginal


def _callout_columns(
    entries: List[Dict[str, Any]],
    marginal_columns: Set[int],
    cfg: LayoutConfig,
) -> Set[int]:
    """
    Detect a dedicated side column containing a summary/key-points panel.

    Such panels do not necessarily span multiple article columns. A common
    journal layout uses a normal-width right-hand column headed, for example,
    "WHAT IS ALREADY KNOWN ON THIS TOPIC". Geometry alone therefore makes the
    panel look like ordinary Abstract or Introduction text.

    Only the first few entries are inspected and a specific callout heading is
    required. This avoids treating an ordinary body column beginning with a
    generic heading such as METHODS as a callout.
    """
    if not cfg.detect_callouts:
        return set()

    by_column: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for entry in entries:
        if entry.get("spanning") or entry.get("marginalia"):
            continue
        by_column[int(entry.get("column", 0))].append(entry)

    callout_columns: Set[int] = set()

    for column, column_entries in by_column.items():
        if column in marginal_columns:
            continue

        ordered_entries = sorted(
            (
                entry
                for entry in column_entries
                if entry.get("text")
                and entry.get("kind") in {"heading", "text"}
            ),
            key=lambda entry: (
                entry.get("top", 0.0),
                entry.get("x0", 0.0),
            ),
        )

        if any(
            CALLOUT_HEADING_PATTERNS.match(entry["text"].strip())
            for entry in ordered_entries[:4]
        ):
            callout_columns.add(column)

    return callout_columns


def _group_spanning(
    spanning: List[Dict[str, Any]],
    page_height: float,
    cfg: LayoutConfig,
) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    previous_bottom: Optional[float] = None
    max_gap = cfg.span_group_gap_ratio * max(page_height, 1.0)

    for entry in spanning:
        if current and previous_bottom is not None and (entry["top"] - previous_bottom) > max_gap:
            groups.append(current)
            current = []
            previous_bottom = None

        current.append(entry)
        previous_bottom = (
            entry["bottom"]
            if previous_bottom is None
            else max(previous_bottom, entry["bottom"])
        )

    if current:
        groups.append(current)

    return groups


def _looks_like_caps_heading(text: str) -> bool:
    candidate = text.strip()

    if not candidate or len(candidate) > 70 or candidate.endswith("."):
        return False

    letters = [c for c in candidate if c.isalpha()]
    if len(letters) < 3:
        return False

    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio < 0.85:
        return False

    return bool(CAPS_HEADING_PATTERN.match(candidate))


def _is_front_matter_text(text: str) -> bool:
    """Identify standalone editorial metadata independently of its box width."""
    return bool(FRONT_MATTER_START_PATTERN.match(clean_text(text)))


def _is_callout_group(group: List[Dict[str, Any]], cfg: LayoutConfig) -> bool:
    if not cfg.detect_callouts or not group:
        return False

    first = group[0]
    first_text = first["text"].strip()

    if not first_text:
        return False

    if CALLOUT_HEADING_PATTERNS.match(first_text):
        return True

    # A genuine document heading that happens to span the columns (article
    # title, full-width section header) must not be turned into a callout.
    if first["kind"] == "heading":
        return False

    return _looks_like_caps_heading(first_text)


def _order_page_entries(
    entries: List[Dict[str, Any]],
    page_no: Optional[int],
    page_width: float,
    page_height: float,
    cfg: LayoutConfig,
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    if page_width <= 0:
        page_width = max((e["x1"] for e in entries), default=1.0) or 1.0
    if page_height <= 0:
        page_height = max((e["bottom"] for e in entries), default=1.0) or 1.0

    narrow = [
        e for e in entries
        if (e["x1"] - e["x0"]) <= cfg.narrow_item_ratio * page_width
    ]
    source = narrow or entries

    columns = _detect_columns(
        [(e["x0"], e["x1"]) for e in source],
        page_width,
        cfg,
    )
    column_widths = [max(c1 - c0, 1e-6) for c0, c1 in columns]
    ordered_widths = sorted(column_widths)
    median_column_width = ordered_widths[len(ordered_widths) // 2]

    marginal_columns = _marginalia_columns(
        columns,
        entries,
        page_width,
        median_column_width,
        cfg,
    )

    for entry in entries:
        overlaps = [
            _overlap(entry["x0"], entry["x1"], c0, c1)
            for c0, c1 in columns
        ]
        best = max(range(len(columns)), key=lambda i: overlaps[i])
        occupied = sum(
            1
            for i, value in enumerate(overlaps)
            if value >= cfg.span_overlap_ratio * column_widths[i]
        )

        entry["column"] = best
        entry["spanning"] = len(columns) > 1 and occupied >= 2
        entry["marginalia"] = (not entry["spanning"]) and best in marginal_columns

    dedicated_callout_columns = _callout_columns(
        entries=entries,
        marginal_columns=marginal_columns,
        cfg=cfg,
    )

    for entry in entries:
        if (
            not entry["spanning"]
            and not entry["marginalia"]
            and entry["column"] in dedicated_callout_columns
        ):
            entry["callout"] = True
            entry["callout_column"] = True
            entry["span_group"] = (
                f"p{page_no}_c{entry['column']}_callout"
            )

    spanning = sorted(
        (e for e in entries if e["spanning"]),
        key=lambda e: (e["top"], e["x0"]),
    )
    body = [e for e in entries if not e["spanning"] and not e["marginalia"]]
    margin = sorted(
        (e for e in entries if e["marginalia"]),
        key=lambda e: (e["column"], e["top"], e["x0"]),
    )

    groups = _group_spanning(spanning, page_height, cfg)

    paired = sorted(
        (
            (
                sum((g["top"] + g["bottom"]) / 2 for g in group) / len(group),
                index,
                group,
            )
            for index, group in enumerate(groups)
        ),
        key=lambda triple: triple[0],
    )
    centers = [center for center, _, _ in paired]
    groups = [group for _, _, group in paired]

    for group_index, group in enumerate(groups):
        group_key = f"p{page_no}_g{group_index}"
        is_callout = _is_callout_group(group, cfg)
        for entry in group:
            entry["span_group"] = group_key
            entry["callout"] = is_callout

    banded: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in body:
        center = (entry["top"] + entry["bottom"]) / 2
        banded[bisect.bisect_left(centers, center)].append(entry)

    ordered: List[Dict[str, Any]] = []
    for band in range(len(groups) + 1):
        ordered.extend(
            sorted(
                banded.get(band, []),
                key=lambda e: (e["column"], e["top"], e["x0"]),
            )
        )
        if band < len(groups):
            ordered.extend(sorted(groups[band], key=lambda e: (e["top"], e["x0"])))

    ordered.extend(margin)
    return ordered


def order_document_items(
    doc: Any,
    cfg: Optional[LayoutConfig] = None,
) -> List[Dict[str, Any]]:
    """Return Docling items in a column-corrected reading order."""
    cfg = cfg or LayoutConfig()

    positioned: List[Dict[str, Any]] = []
    floating: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    last_positioned_order = -1
    order = 0

    for element, _level in doc.iterate_items():
        label = _label_of(element)

        if label in SKIP_LABELS:
            continue

        kind = _kind_of(element, label)
        if kind is None:
            continue

        text = clean_text(str(getattr(element, "text", "") or ""))
        level = getattr(element, "level", None)

        entry: Dict[str, Any] = {
            "element": element,
            "kind": kind,
            "label": label,
            "text": text,
            "level": int(level) if isinstance(level, int) else None,
            "order": order,
            "page": None,
            "column": 0,
            "spanning": False,
            "marginalia": False,
            "callout": False,
            "callout_column": False,
            "span_group": None,
        }
        order += 1

        provenance = list(getattr(element, "prov", None) or [])
        geometry = None
        page_no = None

        if provenance:
            raw_page = getattr(provenance[0], "page_no", None)
            try:
                page_no = int(raw_page) if raw_page is not None else None
            except (TypeError, ValueError):
                page_no = None

            _, page_height = _page_size(doc, page_no)
            geometry = _bbox_top_left(getattr(provenance[0], "bbox", None), page_height)

        if geometry is None:
            # No usable box: keep it next to the previous positioned item so
            # nothing silently drops out of the document.
            floating[last_positioned_order].append(entry)
            continue

        entry["page"] = page_no
        entry["x0"], entry["top"], entry["x1"], entry["bottom"] = geometry
        positioned.append(entry)
        last_positioned_order = entry["order"]

    by_page: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
    for entry in positioned:
        by_page[entry["page"]].append(entry)

    ordered: List[Dict[str, Any]] = []
    ordered.extend(floating.get(-1, []))

    for page_no in sorted(by_page, key=lambda p: (p is None, p)):
        page_width, page_height = _page_size(doc, page_no)
        page_entries = _order_page_entries(
            by_page[page_no],
            page_no,
            page_width,
            page_height,
            cfg,
        )
        for entry in page_entries:
            ordered.append(entry)
            ordered.extend(floating.get(entry["order"], []))

    return ordered


def _new_block(
    block_type: str,
    heading: str,
    heading_path: List[str],
    page: Optional[int],
) -> Dict[str, Any]:
    heading = heading or "Unknown"
    return {
        "type": block_type,
        "section": heading,
        "heading": heading,
        "heading_path": " > ".join(heading_path) if heading_path else heading,
        "merged_headings": [],
        "pages": [page] if page else [],
        "text": "",
        "images": [],
        "_parts": [],
    }


def _track_page(block: Dict[str, Any], page: Optional[int]) -> None:
    if page and page not in block["pages"]:
        block["pages"].append(page)


def _table_markdown(table: Any, doc: Any) -> str:
    for call in (lambda: table.export_to_markdown(doc), lambda: table.export_to_markdown()):
        try:
            markdown = call()
        except TypeError:
            continue
        except Exception:
            continue
        if markdown:
            return str(markdown)

    try:
        return str(table)
    except Exception:
        return ""


def _caption_of(element: Any, doc: Any) -> str:
    try:
        return clean_text(element.caption_text(doc))
    except Exception:
        return ""


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value or "image"


def _picture_page_number(picture: Any) -> Optional[int]:
    provenance = getattr(picture, "prov", None) or []

    if not provenance:
        return None

    page_number = getattr(provenance[0], "page_no", None)

    try:
        return int(page_number) if page_number is not None else None
    except (TypeError, ValueError):
        return None


def _export_picture(
    doc: Any,
    element: Any,
    images_dir: Path,
    picture_count: int,
    section: str,
) -> Optional[Dict[str, Any]]:
    try:
        image = element.get_image(doc)
    except Exception:
        image = None

    if image is None:
        return None

    image_name = (
        f"figure_{picture_count:03d}_"
        f"{_safe_filename(section)[:50]}.png"
    )
    image_path = images_dir / image_name

    try:
        image.save(image_path, "PNG")
    except Exception:
        return None

    return {
        "image_id": f"figure_{picture_count}",
        "path": str(image_path.resolve()),
        "relative_path": str(Path("images") / image_name),
        "caption": _caption_of(element, doc),
        "section": section,
        "page_number": _picture_page_number(element),
    }


def _finalise_blocks(
    blocks: List[Dict[str, Any]],
    marginalia_parts: List[str],
) -> List[Dict[str, Any]]:
    """
    Assign identifiers and make sure no heading is discarded.

    A heading whose section carries no body text (two consecutive headings,
    or a heading immediately followed by a figure) is merged into the next
    chunk through `merged_headings` rather than being dropped.
    """
    finalised: List[Dict[str, Any]] = []
    carried: List[str] = []
    type_counters: Dict[str, int] = defaultdict(int)
    occurrences: Dict[str, int] = defaultdict(int)
    section_index = 0

    for block in blocks:
        block.pop("_parts", None)
        text = block.get("text", "")
        is_section = block["type"] in {"text", "callout"}

        if is_section and not text:
            if block["images"]:
                # Keep the chunk: the heading is the only text anchor the
                # attached figure has.
                block["text"] = block["heading"]
                text = block["text"]
            else:
                if block["heading"] and block["heading"] != "Unknown":
                    carried.append(block["heading"])
                continue

        if not text:
            continue

        if carried and is_section:
            block["merged_headings"] = carried + block.get("merged_headings", [])
            carried = []

        counter = type_counters[block["type"]]
        type_counters[block["type"]] = counter + 1
        block["chunk_id"] = f"{block['type']}_{counter}"

        section_name = block["section"]
        block["section_occurrence"] = occurrences[section_name]
        occurrences[section_name] += 1

        if is_section:
            block["section_index"] = section_index
            section_index += 1
        else:
            block.setdefault("section_index", section_index)

        block["word_count"] = len(text.split())
        finalised.append(block)

    if carried:
        for block in reversed(finalised):
            if block["type"] in {"text", "callout"}:
                block["merged_headings"] = block.get("merged_headings", []) + carried
                carried = []
                break

    if carried:
        finalised.append(
            {
                "chunk_id": "text_orphan_headings",
                "type": "text",
                "section": carried[0],
                "heading": carried[0],
                "heading_path": " > ".join(carried),
                "merged_headings": carried,
                "section_index": section_index,
                "section_occurrence": 0,
                "pages": [],
                "text": ". ".join(carried),
                "word_count": len(" ".join(carried).split()),
                "images": [],
            }
        )

    marginalia_text = clean_text(" ".join(marginalia_parts))
    if marginalia_text:
        finalised.append(
            {
                "chunk_id": "marginalia_0",
                "type": "marginalia",
                "section": "Front Matter",
                "heading": "Front Matter",
                "heading_path": "Front Matter",
                "merged_headings": [],
                "section_index": section_index,
                "section_occurrence": 0,
                "pages": [],
                "text": marginalia_text,
                "word_count": len(marginalia_text.split()),
                "images": [],
            }
        )

    return finalised


def collapse_chunk_section_metadata(
    chunks: List[Dict[str, Any]],
    document_title: str,
) -> None:
    """
    Store the complete section hierarchy in `section` only.

    During parsing, `heading_path` and `merged_headings` remain useful for
    assigning content, tables, and images. Before the chunks are saved, their
    values are combined into a single section label such as
    "Abstract > Objectives", and both auxiliary fields are removed.
    """
    title_key = clean_text(document_title).casefold()

    for chunk in chunks:
        section = clean_text(str(chunk.get("section", "") or ""))
        raw_path = str(chunk.get("heading_path", "") or "")
        raw_merged = chunk.get("merged_headings", []) or []

        path_parts = [
            clean_text(part)
            for part in raw_path.split(">")
            if clean_text(part)
        ]
        merged_parts = [
            clean_text(part)
            for heading in raw_merged
            for part in str(heading).split(">")
            if clean_text(part)
        ]

        # The article title is an identifier, not a section in the reporting
        # hierarchy. Keep it only for its own title/authors chunk.
        if section.casefold() != title_key:
            path_parts = [
                part
                for part in path_parts
                if part.casefold() != title_key
            ]
            merged_parts = [
                part
                for part in merged_parts
                if part.casefold() != title_key
            ]

        path_parts = [
            part for part in path_parts
            if part.casefold() != "unknown"
        ]
        merged_parts = [
            part for part in merged_parts
            if part.casefold() != "unknown"
        ]

        if not path_parts:
            path_parts = [section] if section else []
        elif section and path_parts[-1].casefold() != section.casefold():
            path_parts.append(section)

        existing = {part.casefold() for part in path_parts}
        new_merged = [
            part
            for part in merged_parts
            if part.casefold() not in existing
        ]

        # Carried headings describe parents of the leaf section, so insert
        # them immediately before the last path component.
        if path_parts:
            combined = path_parts[:-1] + new_merged + path_parts[-1:]
        else:
            combined = new_merged

        unique_parts: List[str] = []
        seen: Set[str] = set()
        for part in combined:
            key = part.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_parts.append(part)

        chunk["section"] = " > ".join(unique_parts) or section or "Unknown"
        chunk.pop("heading_path", None)
        chunk.pop("merged_headings", None)


def build_chunks_from_layout(
    doc: Any,
    ordered: List[Dict[str, Any]],
    images_dir: str | Path,
    cfg: Optional[LayoutConfig] = None,
) -> List[Dict[str, Any]]:
    """Turn ordered layout items into section, callout, and table chunks."""
    cfg = cfg or LayoutConfig()
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    blocks: List[Dict[str, Any]] = []
    marginalia_parts: List[str] = []
    heading_stack: List[Tuple[int, str]] = []
    picture_count = 0

    current: Optional[Dict[str, Any]] = None
    active_callout: Optional[Dict[str, Any]] = None
    active_callout_key: Optional[str] = None

    def close_current() -> None:
        nonlocal current
        if current is not None:
            current["text"] = clean_text(" ".join(current["_parts"]))
            blocks.append(current)
            current = None

    def ensure_current(page: Optional[int]) -> Dict[str, Any]:
        nonlocal current
        if current is None:
            headings = [h for _, h in heading_stack]
            current = _new_block(
                "text",
                headings[-1] if headings else "Unknown",
                headings,
                page,
            )
        return current

    def close_callout() -> None:
        nonlocal active_callout, active_callout_key
        if active_callout is not None:
            active_callout["text"] = clean_text(" ".join(active_callout["_parts"]))
            blocks.append(active_callout)
        active_callout = None
        active_callout_key = None

    for entry in ordered:
        kind = entry["kind"]
        page = entry.get("page")

        # Content-based routing is required in addition to the layout-based
        # marginalia flag. A line such as
        # "Editor: ... Received: ..." may occupy a normal-width column and
        # would otherwise be appended to whichever section is currently open.
        is_front_matter = (
            kind in {"text", "heading"}
            and bool(entry["text"])
            and _is_front_matter_text(entry["text"])
        )

        if entry.get("marginalia") or is_front_matter:
            if kind in {"text", "heading"} and entry["text"]:
                marginalia_parts.append(entry["text"])
            continue

        if entry.get("callout"):
            key = entry.get("span_group")
            if key != active_callout_key:
                close_callout()
                heading = (
                    normalize_heading(entry["text"])
                    if entry["text"]
                    else "Callout"
                )
                # A dedicated side column is an independent document region,
                # not a subsection of whichever body heading happened to be
                # processed immediately before it (for example, Abstract or
                # Introduction). An inline/spanning callout may still retain
                # the surrounding body section in its heading path.
                path = (
                    [heading]
                    if entry.get("callout_column")
                    else [h for _, h in heading_stack] + [heading]
                )
                active_callout = _new_block("callout", heading, path, page)
                active_callout_key = key
                if kind in {"heading", "text"}:
                    # The first line is the box title; it is already the
                    # section name, so it is not duplicated into the body.
                    continue
        elif active_callout is not None:
            close_callout()

        target_callout = active_callout

        if kind == "picture":
            target = target_callout or ensure_current(page)
            picture_count += 1
            record = _export_picture(
                doc=doc,
                element=entry["element"],
                images_dir=images_dir,
                picture_count=picture_count,
                section=target["section"],
            )
            if record is not None:
                target["images"].append(record)
            _track_page(target, page)
            continue

        if kind == "table":
            markdown = _table_markdown(entry["element"], doc)
            if not clean_text(markdown):
                continue

            owner = target_callout or ensure_current(page)
            caption = _caption_of(entry["element"], doc)
            blocks.append(
                {
                    "type": "table",
                    "section": owner["section"],
                    "heading": owner["heading"],
                    "heading_path": owner["heading_path"],
                    "merged_headings": [],
                    "caption": caption,
                    "pages": [page] if page else [],
                    "text": markdown,
                    "images": [],
                }
            )
            continue

        if not entry["text"]:
            continue

        is_heading = kind == "heading"

        if (
            not is_heading
            and target_callout is None
            and cfg.promote_caps_headings
            and _looks_like_caps_heading(entry["text"])
        ):
            is_heading = True
            entry["level"] = entry.get("level") or 2

        if is_heading and target_callout is None:
            heading = normalize_heading(entry["text"])
            if not heading:
                continue

            level = entry.get("level") or (0 if entry["label"] == "title" else 1)

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))

            close_current()
            current = _new_block(
                "text",
                heading,
                [h for _, h in heading_stack],
                page,
            )
            continue

        target = target_callout or ensure_current(page)
        target["_parts"].append(entry["text"])
        _track_page(target, page)

    close_callout()
    close_current()

    return _finalise_blocks(blocks, marginalia_parts)


def chunk_markdown(
    markdown: str,
    max_words: int = 450,
    overlap_words: int = 60,
) -> List[Dict[str, Any]]:
    """
    Create exactly one chunk for each complete Markdown section.

    Used by --parse-mode markdown and by the automatic fallback when Docling's
    Markdown reading order is more reliable than its provenance boxes.
    Recognised key-points headings become independent callout chunks, while
    editorial material before the first heading becomes a marginalia chunk.
    Headings that carry no body text are carried forward and attached to the
    next ordinary text chunk through `merged_headings`.

    max_words and overlap_words are retained for CLI compatibility,
    but are intentionally not used.
    """
    sections = re.split(r"\n(?=#{1,6}\s+)", markdown)

    chunks: List[Dict[str, Any]] = []
    type_counters: Dict[str, int] = defaultdict(int)
    section_occurrences: Dict[str, int] = {}
    heading_stack: List[Tuple[int, str]] = []
    carried: List[str] = []

    for section_index, section_text in enumerate(sections):
        section_text = section_text.strip()

        if not section_text:
            continue

        lines = section_text.splitlines()
        first_line = lines[0]

        if first_line.startswith("#"):
            level = len(first_line) - len(first_line.lstrip("#"))
            raw_heading = first_line.lstrip("#").strip()
            section = normalize_heading(raw_heading)
            body = "\n".join(lines[1:]).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, section))
        else:
            section = "Unknown"
            body = section_text

        # Image placeholders belong to the separate image records and should
        # not become literal chunk text.
        body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
        text = clean_text(body)

        if not text:
            if section and section != "Unknown":
                carried.append(section)
            continue

        chunk_type = "text"

        if (
            first_line.startswith("#")
            and CALLOUT_HEADING_PATTERNS.match(raw_heading.strip())
        ):
            chunk_type = "callout"
        elif (
            section == "Unknown"
            and MARGINALIA_PATTERNS.search(text)
        ):
            chunk_type = "marginalia"
            section = "Front Matter"

        occurrence = section_occurrences.get(section, 0)
        section_occurrences[section] = occurrence + 1

        heading_path = (
            section
            if chunk_type in {"callout", "marginalia"}
            else " > ".join(h for _, h in heading_stack) or section
        )

        chunk_index = type_counters[chunk_type]
        type_counters[chunk_type] += 1

        chunks.append(
            {
                "chunk_id": f"{chunk_type}_{chunk_index}",
                "type": chunk_type,
                "section": section,
                "heading": section,
                "heading_path": heading_path,
                "merged_headings": (
                    carried if chunk_type == "text" else []
                ),
                "section_index": section_index,
                "section_occurrence": occurrence,
                "pages": [],
                "text": text,
                "word_count": len(text.split()),
                "images": [],
            }
        )
        carried = []

    if carried and chunks:
        chunks[-1]["merged_headings"] = chunks[-1]["merged_headings"] + carried

    return chunks


def extract_tables(doc: Any) -> List[Dict[str, Any]]:
    table_chunks: List[Dict[str, Any]] = []
    for i, table in enumerate(getattr(doc, "tables", [])):
        markdown = _table_markdown(table, doc)

        if not clean_text(markdown):
            continue

        table_chunks.append({
            "chunk_id": f"table_{i}",
            "type": "table",
            "section": "Tables",
            "heading": "Tables",
            "heading_path": "Tables",
            "merged_headings": [],
            "table_number": i + 1,
            "caption": _caption_of(table, doc) or f"Table {i + 1}",
            "pages": [],
            "text": markdown,
            "word_count": len(clean_text(markdown).split()),
            "images": [],
        })
    return table_chunks


def extract_and_attach_images(
    doc: Any,
    chunks: List[Dict[str, Any]],
    images_dir: Path,
) -> int:
    """
    Export Docling PictureItem images and append them to the section chunk
    that is active when the picture occurs in document reading order.

    Used by the markdown fallback parser only. Section occurrences are matched
    by consuming chunks per section name, because the markdown chunker skips
    heading-only sections and the raw heading counter would otherwise drift.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    remaining_by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if chunk.get("type") in {"text", "callout"}:
            remaining_by_section[str(chunk.get("section", "Unknown"))].append(chunk)

    text_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("type") in {"text", "callout"}
    ]

    current_section = "Unknown"
    last_text_chunk = text_chunks[0] if text_chunks else None
    picture_count = 0

    for element, _level in doc.iterate_items():
        if isinstance(element, SectionHeaderItem):
            current_section = normalize_heading(
                str(getattr(element, "text", "") or "Unknown")
            )

            pending = remaining_by_section.get(current_section)
            if pending:
                last_text_chunk = pending.pop(0)

            continue

        if not isinstance(element, PictureItem):
            continue

        target_chunk = last_text_chunk

        if target_chunk is None:
            continue

        picture_count += 1

        record = _export_picture(
            doc=doc,
            element=element,
            images_dir=images_dir,
            picture_count=picture_count,
            section=str(target_chunk.get("section", "Unknown")),
        )

        if record is None:
            continue

        target_chunk.setdefault("images", []).append(record)

    return sum(
        len(chunk.get("images", []))
        for chunk in chunks
    )


def _markdown_has_callout_headings(markdown: str) -> bool:
    """Return True when Markdown exposes a recognised key-points section."""
    headings = re.findall(
        r"(?m)^#{1,6}\s+(.+?)\s*$",
        markdown,
    )
    return any(
        CALLOUT_HEADING_PATTERNS.match(heading.strip())
        for heading in headings
    )


def _build_markdown_chunks(
    doc: Any,
    markdown: str,
    images_dir: Path,
    max_words: int,
    overlap_words: int,
) -> Tuple[List[Dict[str, Any]], int]:
    text_chunks = chunk_markdown(
        markdown,
        max_words,
        overlap_words,
    )
    table_chunks = extract_tables(doc)
    chunks = text_chunks + table_chunks

    num_images = extract_and_attach_images(
        doc=doc,
        chunks=chunks,
        images_dir=images_dir,
    )
    return chunks, num_images


def _image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Retrieved image not found: {path}"
        )

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"

    supported_mime_types = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }
    if mime_type not in supported_mime_types:
        raise ValueError(
            f"Unsupported image type {mime_type!r}: {path}"
        )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def collect_evidence_images(
    evidence_payload: Dict[str, Any],
    max_images: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return unique, readable images in evidence ranking order."""
    selected: List[Dict[str, Any]] = []
    seen_paths: Set[str] = set()

    for row in evidence_payload.get("top_evidence", []):
        chunk_id = str(row.get("chunk_id", ""))

        for image in row.get("images", []):
            if not isinstance(image, dict):
                continue

            image_path = str(image.get("path", "")).strip()
            if not image_path:
                continue

            resolved_path = str(Path(image_path).resolve())
            if resolved_path in seen_paths:
                continue
            if not Path(resolved_path).is_file():
                continue

            seen_paths.add(resolved_path)
            selected.append(
                {
                    "chunk_id": chunk_id,
                    "image_id": image.get("image_id"),
                    "caption": image.get("caption"),
                    "section": image.get("section"),
                    "page_number": image.get("page_number"),
                    "path": resolved_path,
                }
            )

            if max_images is not None and len(selected) >= max_images:
                return selected

    return selected


def build_multimodal_user_content(
    model_input: str,
    evidence_payload: Dict[str, Any],
    include_images: bool,
    max_images: int,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible multimodal user content.

    Each image is preceded by a text label identifying the chunk it belongs to,
    so the model can cite the parent chunk_id in its result.
    """
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": model_input,
        }
    ]

    if not include_images:
        return content

    images = collect_evidence_images(
        evidence_payload,
        max_images=max_images,
    )

    for image in images:
        label = {
            key: image.get(key)
            for key in (
                "chunk_id",
                "image_id",
                "caption",
                "section",
                "page_number",
            )
        }

        content.append(
            {
                "type": "text",
                "text": (
                    "The next image belongs to this evidence record:\n"
                    + json.dumps(label, ensure_ascii=False)
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_data_url(image["path"]),
                    "detail": "high",
                },
            }
        )

    return content


def evidence_has_images(
    evidence_payload: Dict[str, Any],
) -> bool:
    return bool(collect_evidence_images(evidence_payload, max_images=1))


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    max_words: int,
    overlap_words: int,
    parse_mode: str = "auto",
    layout_config: Optional[LayoutConfig] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    # Table structure is what keeps a data table from being flattened into the
    # surrounding column text.
    if hasattr(pipeline_options, "do_table_structure"):
        pipeline_options.do_table_structure = True
    table_options = getattr(pipeline_options, "table_structure_options", None)
    if table_options is not None and hasattr(table_options, "do_cell_matching"):
        table_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document
    markdown = fix_abstract_intro_order(doc.export_to_markdown())
    document_title = extract_document_title(
        doc,
        markdown=markdown,
    )

    markdown_path = output_dir / f"{pdf_path.stem}.md"
    chunks_path = output_dir / f"{pdf_path.stem}_chunks.json"
    markdown_path.write_text(markdown, encoding="utf-8")

    images_dir = output_dir / "images"

    requested_parse_mode = parse_mode
    effective_parse_mode = parse_mode

    if parse_mode in {"auto", "layout"}:
        ordered = order_document_items(doc, layout_config)
        chunks = build_chunks_from_layout(
            doc=doc,
            ordered=ordered,
            images_dir=images_dir,
            cfg=layout_config,
        )
        num_images = sum(len(c.get("images", [])) for c in chunks)

        layout_callouts = sum(
            chunk.get("type") == "callout"
            for chunk in chunks
        )
        should_fallback = (
            parse_mode == "auto"
            and _markdown_has_callout_headings(markdown)
            and layout_callouts == 0
        )

        if should_fallback:
            print(
                "    Layout boxes did not identify callout regions; "
                "using Docling's Markdown reading order instead."
            )
            chunks, num_images = _build_markdown_chunks(
                doc=doc,
                markdown=markdown,
                images_dir=images_dir,
                max_words=max_words,
                overlap_words=overlap_words,
            )
            effective_parse_mode = "markdown_fallback"
    else:
        chunks, num_images = _build_markdown_chunks(
            doc=doc,
            markdown=markdown,
            images_dir=images_dir,
            max_words=max_words,
            overlap_words=overlap_words,
        )
    collapse_chunk_section_metadata(
        chunks=chunks,
        document_title=document_title,
    )

    for chunk in chunks:
        chunk["document_title"] = document_title
    print(f"{document_title}: {len(chunks)} chunks, {num_images} images")
    
    counts = defaultdict(int)
    for chunk in chunks:
        counts[chunk.get("type", "text")] += 1

    payload = {
        "source_file": str(pdf_path.resolve()),
        "document_title": document_title,
        "markdown_file": str(markdown_path.resolve()),
        "requested_parse_mode": requested_parse_mode,
        "parse_mode": effective_parse_mode,
        "num_chunks": len(chunks),
        "num_text_chunks": counts["text"],
        "num_table_chunks": counts["table"],
        "num_callout_chunks": counts["callout"],
        "num_marginalia_chunks": counts["marginalia"],
        "num_images": num_images,
        "chunks": chunks,
    }
    chunks_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "markdown_path": markdown_path,
        "chunks_path": chunks_path,
        "markdown": markdown,
        "document_title": document_title,
        **{
            key: payload[key]
            for key in (
                "num_chunks",
                "num_text_chunks",
                "num_table_chunks",
                "num_callout_chunks",
                "num_marginalia_chunks",
                "num_images",
            )
        },
    }


# ---------------------------------------------------------------------------
# Checklist loading, evidence retrieval, and LLM evaluation
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "z-ai/glm-5.3-flash",
    "qwen/qwen3.5-35b-a3b",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-chat-v3.1",
    "openai/gpt-5.4-nano",
]

SYSTEM_PROMPT = """
You are an expert evaluator of systematic review reporting.

Your task is to evaluate a single reporting criterion using only the retrieved evidence provided.

Each reporting element is classified as either REQUIRED or CONDITIONAL.

Possible element statuses:
- PRESENT
- PARTIAL
- MISSING
- NOT_APPLICABLE

Instructions:

1. Evaluate every supplied reporting element independently.
2. Use only the retrieved evidence. Do not infer or assume information that is not explicitly reported.
3. When image bytes are attached, each image is paired with metadata identifying its parent chunk. When visible image evidence is used, cite that parent chunk_id and describe the visible evidence precisely. A caption or image metadata without attached image bytes is text evidence only, not visual evidence.
4. Return one result for every supplied element_id.
5. For every PRESENT or PARTIAL element, provide:
   - the supporting chunk_id
   - an exact quotation
   - a concise explanation.
6. REQUIRED elements must never be marked NOT_APPLICABLE.
7. CONDITIONAL elements may be marked NOT_APPLICABLE only when the reporting requirement genuinely does not apply to the manuscript and the evidence supports that conclusion.
8. Do not calculate the overall reporting status or missing elements. Those are calculated by the evaluation pipeline.
9. Return exactly one JSON object whose top-level keys are item_id, elements, and confidence.
10. Do not wrap the result inside required_output, output, response, result, or any other object.
11. Do not repeat the input, criterion, retrieved_evidence, or output template.
12. Return valid JSON only, without Markdown code fences or explanatory text.
13. Note that abstarct section may not be explicitly labeled as "Abstract" in the manuscript. It may be part of the introduction or background section. Use your best judgment to identify the abstract content based on the provided evidence.
"""

def load_checklist(
    path: Path,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load checklist metadata and items.

    Item-level applicability and decision rules are ignored. Applicability is
    defined only for atomic elements as REQUIRED or CONDITIONAL.
    """
    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    checklist_config: Dict[str, Any] = {}

    if isinstance(raw_data, dict):
        checklist_config = {
            key: value
            for key, value in raw_data.items()
            if key not in {"items", "checklist", "criteria"}
        }
        data = raw_data.get(
            "items",
            raw_data.get(
                "checklist",
                raw_data.get("criteria"),
            ),
        )
    else:
        data = raw_data

    if not isinstance(data, list):
        raise ValueError(
            "Checklist must be a JSON array or contain "
            "items/checklist/criteria."
        )

    items: List[Dict[str, Any]] = []

    for index, raw in enumerate(data, start=1):
        if isinstance(raw, str):
            raw = {"checklist_item": raw}

        if not isinstance(raw, dict):
            raise ValueError(
                f"Checklist item {index} must be a string or object."
            )

        item = dict(raw)
        item.pop("applicability", None)
        item.pop("decision_rules", None)

        item.setdefault("item_id", str(index))
        item.setdefault(
            "item_label",
            item.get("checklist_item", f"Item {index}"),
        )
        item.setdefault(
            "semantic_query",
            item.get("checklist_item", ""),
        )
        item.setdefault("keywords", [])
        item.setdefault("sections_to_search", [])
        item.setdefault("required_elements", [])

        for element_index, element in enumerate(
            item["required_elements"],
            start=1,
        ):
            if not isinstance(element, dict):
                raise ValueError(
                    f"{item['item_id']} element {element_index} "
                    "must be an object."
                )

            element_id = element.get("element_id")
            if not element_id:
                raise ValueError(
                    f"{item['item_id']} element {element_index} "
                    "is missing element_id."
                )

            applicability = str(
                element.get("applicability", "REQUIRED")
            ).upper()

            if applicability not in {
                "REQUIRED",
                "CONDITIONAL",
            }:
                raise ValueError(
                    f"{item['item_id']} / {element_id}: "
                    f"invalid applicability {applicability!r}."
                )

            element["applicability"] = applicability

        items.append(item)

    return checklist_config, items


def build_model_input(
    document_title: str,
    checklist_item: Dict[str, Any],
    evidence_payload: Dict[str, Any],
) -> str:
    compact_evidence = [
        {
            "chunk_id": row.get("chunk_id"),
            "section": row.get("section"),
            "document_title": row.get(
                "document_title",
                document_title,
            ),
            "type": row.get("type"),
            "text": row.get("text"),
            "images": [
                {
                    "image_id": image.get("image_id"),
                    "caption": image.get("caption"),
                    "section": image.get("section"),
                    "page_number": image.get("page_number"),
                }
                for image in row.get("images", [])
                if isinstance(image, dict)
            ],
            "retrieval_score": row.get("final_score"),
        }
        for row in evidence_payload.get("top_evidence", [])
    ]

    request = {
        "document_title": document_title,
        "criterion": {
            "standard": checklist_item.get("standard"),
            "section": checklist_item.get("section"),
            "topic": checklist_item.get("topic"),
            "item_id": checklist_item.get("item_id"),
            "item_label": checklist_item.get("item_label"),
            "checklist_item": checklist_item.get(
                "checklist_item"
            ),
            "required_elements": checklist_item.get(
                "required_elements",
                [],
            ),
        },
        "retrieved_evidence": compact_evidence,
    }

    output_template = {
        "item_id": checklist_item.get("item_id"),
        "elements": [
            {
                "element_id": "copy the supplied element_id exactly",
                "status": (
                    "PRESENT | PARTIAL | MISSING | "
                    "NOT_APPLICABLE"
                ),
                "chunk_id": "supporting chunk_id or null",
                "quote": "verbatim evidence or null",
                "reason": "concise explanation",
            }
        ],
        "confidence": "integer 0-100",
    }

    return (
        "Evaluate the input data below.\n\n"
        "OUTPUT CONTRACT:\n"
        "Return the JSON object shown below directly as the entire "
        "response. Do not wrap it in required_output, output, "
        "response, result, or any other key. Return one element for "
        "every supplied element_id.\n"
        + json.dumps(
            output_template,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nINPUT DATA:\n"
        + json.dumps(
            request,
            ensure_ascii=False,
            indent=2,
        )
    )



VALID_ELEMENT_STATUSES = {
    "PRESENT",
    "PARTIAL",
    "MISSING",
    "NOT_APPLICABLE",
}

VALID_ELEMENT_APPLICABILITY = {
    "REQUIRED",
    "CONDITIONAL",
}


def _element_definitions(
    item: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(element.get("element_id")): element
        for element in item.get("required_elements", [])
        if isinstance(element, dict)
        and element.get("element_id")
    }


def _returned_elements(
    model_result: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(element.get("element_id")): element
        for element in model_result.get("elements", [])
        if isinstance(element, dict)
        and element.get("element_id")
    }


def _normalise_element_status(status: Any) -> str:
    normalised = str(status or "MISSING").upper()

    if normalised not in VALID_ELEMENT_STATUSES:
        return "MISSING"

    return normalised


def _normalise_applicability(
    applicability: Any,
) -> str:
    normalised = str(
        applicability or "REQUIRED"
    ).upper()

    if normalised not in VALID_ELEMENT_APPLICABILITY:
        return "REQUIRED"

    return normalised


def calculate_overall_status(
    item: Dict[str, Any],
    model_result: Dict[str, Any],
) -> str:
    """
    Calculate item status from element statuses.

    REQUIRED:
    - always scored
    - NOT_APPLICABLE is converted to MISSING

    CONDITIONAL:
    - PRESENT, PARTIAL, and MISSING are scored
    - NOT_APPLICABLE is excluded from scoring

    Overall:
    - all scored elements PRESENT -> FULLY_REPORTED
    - all scored elements MISSING -> NOT_REPORTED
    - any other scored combination -> PARTIALLY_REPORTED
    - no scored elements -> NOT_APPLICABLE
    """
    definitions = _element_definitions(item)
    returned = _returned_elements(model_result)

    scored_statuses: List[str] = []

    for element_id, definition in definitions.items():
        applicability = _normalise_applicability(
            definition.get("applicability")
        )

        model_element = returned.get(element_id)

        if model_element is None:
            scored_statuses.append("MISSING")
            continue

        status = _normalise_element_status(
            model_element.get("status")
        )

        if applicability == "REQUIRED":
            if status == "NOT_APPLICABLE":
                status = "MISSING"

            scored_statuses.append(status)
            continue

        if status == "NOT_APPLICABLE":
            continue

        scored_statuses.append(status)

    if not scored_statuses:
        return "NOT_APPLICABLE"

    if all(
        status == "PRESENT"
        for status in scored_statuses
    ):
        return "FULLY_REPORTED"

    if all(
        status == "MISSING"
        for status in scored_statuses
    ):
        return "NOT_REPORTED"

    return "PARTIALLY_REPORTED"


def calculate_missing_elements(
    item: Dict[str, Any],
    model_result: Dict[str, Any],
) -> List[str]:
    """
    Return element IDs scored as MISSING or PARTIAL.

    Conditional elements marked NOT_APPLICABLE are excluded.
    Missing model responses are included as missing.
    """
    definitions = _element_definitions(item)
    returned = _returned_elements(model_result)

    missing: List[str] = []

    for element_id, definition in definitions.items():
        applicability = _normalise_applicability(
            definition.get("applicability")
        )

        model_element = returned.get(element_id)

        if model_element is None:
            missing.append(element_id)
            continue

        status = _normalise_element_status(
            model_element.get("status")
        )

        if (
            applicability == "REQUIRED"
            and status == "NOT_APPLICABLE"
        ):
            status = "MISSING"

        if (
            applicability == "CONDITIONAL"
            and status == "NOT_APPLICABLE"
        ):
            continue

        if status in {"MISSING", "PARTIAL"}:
            missing.append(element_id)

    return missing


def add_deterministic_status(
    item: Dict[str, Any],
    parsed_result: Any,
) -> Any:
    if not isinstance(parsed_result, dict):
        return parsed_result

    if parsed_result.get("parse_error"):
        return parsed_result

    parsed_result.pop("applicability", None)

    parsed_result["overall_status"] = (
        calculate_overall_status(
            item=item,
            model_result=parsed_result,
        )
    )

    parsed_result["missing_elements"] = (
        calculate_missing_elements(
            item=item,
            model_result=parsed_result,
        )
    )

    return parsed_result


def extract_json_response(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return {"raw_response": content, "parse_error": True}


def discover_image_input_support(
    client: OpenAI,
    models: List[str],
) -> Dict[str, Optional[bool]]:
    """
    Read OpenRouter model metadata once.

    True means image input is declared, False means it is explicitly absent,
    and None means the catalog could not determine the capability.
    """
    support: Dict[str, Optional[bool]] = {
        model: None for model in models
    }

    try:
        catalog = client.models.list()
    except Exception:
        return support

    by_id: Dict[str, Dict[str, Any]] = {}
    for model_info in getattr(catalog, "data", []):
        if hasattr(model_info, "model_dump"):
            raw = model_info.model_dump()
        elif isinstance(model_info, dict):
            raw = model_info
        else:
            continue

        model_id = str(raw.get("id", "")).strip()
        if model_id:
            by_id[model_id] = raw

    for model in models:
        raw = by_id.get(model)
        if raw is None:
            continue

        architecture = raw.get("architecture") or {}
        input_modalities = architecture.get("input_modalities")
        if isinstance(input_modalities, list):
            support[model] = "image" in {
                str(value).lower() for value in input_modalities
            }

    return support


def _is_unsupported_image_error(exc: Exception) -> bool:
    """Identify only errors that clearly say image input is unsupported."""
    message = str(exc).lower()
    markers = (
        "does not support image",
        "doesn't support image",
        "image input is not supported",
        "image inputs are not supported",
        "unsupported image input",
        "unsupported image modality",
        "unsupported modality: image",
        "no endpoints found that support image input",
        "non-multimodal model",
        "only text content is supported",
        "only supports text input",
    )
    return any(marker in message for marker in markers)


def _response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif hasattr(part, "text") and isinstance(part.text, str):
                parts.append(part.text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def ask_model(
    client: OpenAI,
    model: str,
    model_input: str,
    evidence_payload: Dict[str, Any],
    max_tokens: int,
    image_input_supported: Optional[bool],
    max_images: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    available_images = collect_evidence_images(evidence_payload)
    has_images = bool(available_images)
    should_send_images = has_images and image_input_supported is not False
    reasoning_config = {
        "effort": reasoning_effort,
        # The model still reasons, but the reasoning trace is omitted from
        # the response. Reasoning-token usage is unaffected.
        "exclude": True,
    }

    if should_send_images:
        user_content: Any = build_multimodal_user_content(
            model_input=model_input,
            evidence_payload=evidence_payload,
            include_images=True,
            max_images=max_images,
        )
        images_sent = min(len(available_images), max_images)
    else:
        # Use a plain string for text-only models. Some providers reject a
        # multipart content array even when it contains only a text part.
        user_content = model_input
        images_sent = 0

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            extra_body={"reasoning": reasoning_config},
        )
        multimodal_used = images_sent > 0
        multimodal_fallback_error = None

    except Exception as multimodal_error:
        if not should_send_images or not _is_unsupported_image_error(
            multimodal_error
        ):
            raise

        # Retry only when the provider explicitly rejects image input. Network,
        # authentication, rate-limit, and unrelated bad-request errors remain
        # visible instead of being hidden by a text-only retry.
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": model_input,
                },
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            extra_body={"reasoning": reasoning_config},
        )
        multimodal_used = False
        images_sent = 0
        multimodal_fallback_error = (
            f"{type(multimodal_error).__name__}: "
            f"{multimodal_error}"
        )

    content = _response_content_to_text(
        response.choices[0].message.content
    )

    return {
        "raw": content,
        "parsed": extract_json_response(content),
        "images_available": has_images,
        "images_available_count": len(available_images),
        "images_sent": images_sent,
        "image_input_supported": image_input_supported,
        "reasoning_effort": reasoning_effort,
        "multimodal_used": multimodal_used,
        "multimodal_fallback_error": (
            multimodal_fallback_error
        ),
    }


def evaluate_models(
    *,
    document_title: str,
    items: List[Dict[str, Any]],
    checklist_config: Dict[str, Any],
    retriever: PrismaFaissRetriever,
    output_dir: Path,
    models: List[str],
    api_key: str,
    base_url: str,
    max_tokens: int,
    max_images: int,
    debug_retrieval: bool,
    max_workers: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    metadata_client = OpenAI(base_url=base_url, api_key=api_key)
    image_support = discover_image_input_support(metadata_client, models)
    evaluations_dir = output_dir / "evaluations"
    evidence_dir = output_dir / "evidence"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    all_evidence: List[Dict[str, Any]] = []
    prepared_items: List[
        Tuple[int, Dict[str, Any], Dict[str, Any], str]
    ] = []

    # Retrieval uses shared embedding/reranking models and is therefore kept
    # sequential. The network-bound LLM calls below are run concurrently.
    for item_index, item in enumerate(items):
        position = item_index + 1
        print(
            f"[{position}/{len(items)}] Retrieving evidence for "
            f"{item.get('item_id')}"
        )
        evidence = retriever.query(item, debug=debug_retrieval)
        all_evidence.append(evidence)
        evidence_file = evidence_dir / f"item_{item.get('item_id')}.json"
        evidence_file.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        prepared_items.append(
            (
                item_index,
                item,
                evidence,
                build_model_input(document_title, item, evidence),
            )
        )

    # One client per worker thread avoids sharing mutable client state while
    # still reusing each thread's HTTP connection pool across requests.
    worker_state = local()

    def get_worker_client() -> OpenAI:
        client = getattr(worker_state, "client", None)
        if client is None:
            client = OpenAI(base_url=base_url, api_key=api_key)
            worker_state.client = client
        return client

    def evaluate_one(
        item_index: int,
        item: Dict[str, Any],
        evidence: Dict[str, Any],
        model_input: str,
        model: str,
    ) -> Tuple[int, str, Dict[str, Any]]:
        try:
            answer = ask_model(
                get_worker_client(),
                model,
                model_input,
                evidence,
                max_tokens,
                image_input_supported=image_support.get(model),
                max_images=max_images,
                reasoning_effort=reasoning_effort,
            )
            answer["parsed"] = add_deterministic_status(
                item=item,
                parsed_result=answer.get("parsed"),
            )
            row = {
                "item_id": item.get("item_id"),
                "item_label": item.get("item_label"),
                "model": model,
                **answer,
            }
        except Exception as exc:
            row = {
                "item_id": item.get("item_id"),
                "item_label": item.get("item_label"),
                "model": model,
                "reasoning_effort": reasoning_effort,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return item_index, model, row

    # Preallocate slots so completion order never changes output order.
    ordered_results: Dict[str, List[Optional[Dict[str, Any]]]] = {
        model: [None] * len(items) for model in models
    }
    total_jobs = len(prepared_items) * len(models)
    worker_count = min(max_workers, total_jobs) if total_jobs else 1

    print(
        f"Evaluating {total_jobs} item-model combinations "
        f"with {worker_count} parallel workers..."
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                item_index,
                item,
                evidence,
                model_input,
                model,
            )
            for item_index, item, evidence, model_input in prepared_items
            for model in models
        ]

        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            item_index, model, row = future.result()
            ordered_results[model][item_index] = row
            outcome = "error" if "error" in row else "done"
            print(
                f"    [{completed}/{total_jobs}] {outcome}: "
                f"{row.get('item_id')} / {model}"
            )

    model_results: Dict[str, List[Dict[str, Any]]] = {
        model: [row for row in rows if row is not None]
        for model, rows in ordered_results.items()
    }

    for model, rows in model_results.items():
        safe_name = model.replace("/", "_").replace(":", "_")
        (evaluations_dir / f"{safe_name}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    combined = {
        "document_title": document_title,
        "standard": checklist_config.get("standard"),
        "schema_version": checklist_config.get("schema_version"),
        "models": models,
        "reasoning_effort": reasoning_effort,
        "image_input_support": image_support,
        "num_checklist_items": len(items),
        "evidence": all_evidence,
        "evaluations": model_results,
    }
    combined_path = output_dir / "combined_evaluation.json"
    combined_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"combined_path": combined_path, "results": combined}


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def parse_models(value: str) -> List[str]:
    models = [model.strip() for model in value.split(",") if model.strip()]
    if not models:
        raise argparse.ArgumentTypeError("At least one model is required.")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: PDF -> Docling chunks -> FAISS retrieval -> "
            "PRISMA/ROSES/MECIR evaluation through OpenRouter."
        )
    )
    parser.add_argument("pdf", type=Path, help="Input manuscript PDF")
    parser.add_argument(
        "checklist",
        nargs="?",
        type=Path,
        help=(
            "Checklist JSON. Required for a full evaluation, but not with "
            "--parse-only or --skip-evaluation."
        ),
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("pipeline_output"))
    parser.add_argument("--max-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument(
        "--parse-mode",
        choices=("auto", "layout", "markdown"),
        default="auto",
        help=(
            "auto: use layout boxes, but fall back to Markdown when known "
            "callout headings are missed (default). layout: always rebuild "
            "reading order from page geometry. markdown: always follow "
            "Docling's exported Markdown order."
        ),
    )
    parser.add_argument(
        "--column-gap-ratio",
        type=float,
        default=LayoutConfig.column_gap_ratio,
        help=(
            "Minimum gutter width as a fraction of page width. Raise it if "
            "columns are split too eagerly, lower it if they are merged."
        ),
    )
    parser.add_argument(
        "--no-marginalia-detection",
        action="store_true",
        help="Keep the narrow metadata rail inside the body flow.",
    )
    parser.add_argument(
        "--no-callout-detection",
        action="store_true",
        help="Keep boxed panels inside the surrounding section.",
    )
    parser.add_argument(
        "--no-caps-headings",
        action="store_true",
        help="Do not promote all-caps text lines to headings.",
    )
    parser.add_argument("--initial-top-k", type=int, default=50)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--embedding-model", default=RetrieverConfig.embedding_model_name)
    parser.add_argument("--reranker-model", default=RetrieverConfig.reranker_model_name)
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "max"),
        default="low",
        help=(
            "Reasoning effort for all selected models (default: low). "
            "The setting is recorded in every evaluation result."
        ),
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=8,
        help=(
            "Maximum number of concurrent item-model API calls "
            "(default: 8). Reduce this if the provider rate-limits requests."
        ),
    )
    parser.add_argument(
        "--max-images-per-request",
        type=int,
        default=6,
        help=(
            "Maximum number of retrieved figures sent with one checklist "
            "item (default: 6)."
        ),
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--debug-retrieval", action="store_true")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--parse-only",
        action="store_true",
        help=(
            "Only parse the PDF and create Markdown, chunks, and extracted "
            "images. Do not load a checklist, build FAISS, or call models."
        ),
    )
    run_mode.add_argument(
        "--skip-evaluation",
        action="store_true",
        help=(
            "Parse and chunk the PDF and build the FAISS index, but do not "
            "load a checklist or call models."
        ),
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    checklist_required = not (
        args.parse_only or args.skip_evaluation
    )
    if checklist_required and args.checklist is None:
        parser.error(
            "the checklist argument is required unless --parse-only or "
            "--skip-evaluation is used"
        )
    if args.checklist is not None and not args.checklist.is_file():
        raise FileNotFoundError(f"Checklist not found: {args.checklist}")
    if args.max_images_per_request < 1:
        raise ValueError("--max-images-per-request must be at least 1")
    if args.evaluation_workers < 1:
        raise ValueError("--evaluation-workers must be at least 1")

    layout_config = LayoutConfig(
        column_gap_ratio=args.column_gap_ratio,
        detect_marginalia=not args.no_marginalia_detection,
        detect_callouts=not args.no_callout_detection,
        promote_caps_headings=not args.no_caps_headings,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 1 if args.parse_only else (2 if args.skip_evaluation else 4)
    print(f"1/{total_steps} Parsing PDF and creating chunks...")
    converted = convert_pdf(
        args.pdf,
        args.output_dir,
        args.max_words,
        args.overlap_words,
        parse_mode=args.parse_mode,
        layout_config=layout_config,
    )
    print(
        f"    text={converted['num_text_chunks']} "
        f"tables={converted['num_table_chunks']} "
        f"callouts={converted['num_callout_chunks']} "
        f"front matter={converted['num_marginalia_chunks']} "
        f"images={converted['num_images']}"
    )

    if args.parse_only:
        print("Parsing complete.")
        print(f"Markdown: {converted['markdown_path']}")
        print(f"Chunks:   {converted['chunks_path']}")
        print(f"Images:   {converted['num_images']}")
        return

    print(f"2/{total_steps} Building FAISS index...")
    retriever_config = RetrieverConfig(
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        initial_top_k=args.initial_top_k,
        final_top_k=args.final_top_k,
    )
    retriever = PrismaFaissRetriever(args.output_dir / "faiss_store", retriever_config)
    retriever.build(converted["chunks_path"])

    if args.skip_evaluation:
        print("Indexing complete; evaluation skipped.")
        print(f"Markdown: {converted['markdown_path']}")
        print(f"Chunks:   {converted['chunks_path']}")
        print(f"Index:    {retriever.store_dir}")
        return

    print("3/4 Loading checklist...")
    assert args.checklist is not None
    checklist_config, items = load_checklist(args.checklist)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set the {args.api_key_env} environment variable."
        )

    print("4/4 Retrieving evidence and evaluating models...")
    evaluation = evaluate_models(
        document_title=converted["document_title"],
        items=items,
        checklist_config=checklist_config,
        retriever=retriever,
        output_dir=args.output_dir,
        models=args.models,
        api_key=api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        max_images=args.max_images_per_request,
        debug_retrieval=args.debug_retrieval,
        max_workers=args.evaluation_workers,
        reasoning_effort=args.reasoning_effort,
    )

    print("Done.")
    print(f"Markdown:  {converted['markdown_path']}")
    print(f"Chunks:    {converted['chunks_path']}")
    print(f"Images:    {converted['num_images']}")
    print(f"FAISS:     {retriever.store_dir}")
    print(f"Evaluation:{evaluation['combined_path']}")


if __name__ == "__main__":
    main()
