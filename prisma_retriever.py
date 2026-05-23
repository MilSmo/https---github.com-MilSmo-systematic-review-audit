from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class RetrieverConfig:
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
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

    def build(self, chunks_jsonl_path: str | Path) -> None:
        chunks = self._load_jsonl(Path(chunks_jsonl_path))
        if not chunks:
            raise ValueError(f"No chunks found in {chunks_jsonl_path}")

        self._load_embedder()

        texts = [self._get_embedding_text(c) for c in chunks]

        embeddings = self.embedder.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        ).astype("float32")

        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.metadata = [self._metadata_from_chunk(i, c) for i, c in enumerate(chunks)]

        self.save()

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

        query_text = self.build_dense_query_text(item)

        retrieved = self._retrieve(
            item=item,
            query_text=query_text,
            top_k=initial_top_k,
        )

        if debug:
            print("\nDENSE QUERY:")
            print(query_text)
            print("\nTOP BEFORE RERANK:")
            for r in retrieved[:15]:
                c = r["chunk"]
                print(
                    r["rank"],
                    "faiss=", round(r["faiss_score"], 4),
                    "adjusted=", round(r["adjusted_retrieval_score"], 4),
                    "section_bonus=", round(r["section_bonus"], 4),
                    "keyword_bonus=", round(r["keyword_bonus"], 4),
                    c["section"],
                    c["text"][:220].replace("\n", " "),
                )

        reranked = self._rerank(
            query_text=query_text,
            retrieved=retrieved,
            final_top_k=final_top_k,
        )

        return self._make_evidence_payload(
            item=item,
            query_text=query_text,
            reranked=reranked,
        )

    # --------------------------
    # Query logic
    # --------------------------

    def build_dense_query_text(self, item: Dict[str, Any]) -> str:
        parts = []

        for field in (
            "checklist_item",
            "query_template",
            "query_template_expanded",
            "topic",
            "section",
        ):
            value = (item.get(field) or "").strip()
            if value:
                parts.append(value)

        if item.get("evidence_requirements"):
            parts.append(
                "Evidence requirements: " + " ".join(item["evidence_requirements"])
            )
        if item.get("evidence_requirements_expanded"):
            parts.append(
                "Expanded evidence guidance: "
                + " ".join(item["evidence_requirements_expanded"])
            )

        key_phrases = self._select_query_keywords(item)
        if key_phrases:
            parts.append(
                "Important evidence phrases: "
                + ", ".join(key_phrases[: self.config.max_query_keywords])
            )

        return " ".join(parts).strip()

    def _select_query_keywords(self, item: Dict[str, Any]) -> List[str]:
        raw_keywords = item.get("keywords") or []
        tokens = []
        seen = set()

        for kw in raw_keywords:
            cleaned = self._clean_keyword(kw)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered not in seen:
                tokens.append(cleaned)
                seen.add(lowered)

        # Backfill with informative keyphrases extracted from checklist/query text
        combined_text = " ".join(
            str(item.get(k, "") or "")
            for k in (
                "checklist_item",
                "query_template",
                "query_template_expanded",
                "topic",
                "section",
            )
        )
        for phrase in self._extract_candidate_phrases(combined_text):
            lowered = phrase.lower()
            if lowered not in seen:
                tokens.append(phrase)
                seen.add(lowered)

        tokens.sort(key=self._keyword_priority, reverse=True)
        return tokens

    @staticmethod
    def _clean_keyword(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip())
        text = re.sub(r"^[\-–—•\d\.\)\(]+", "", text).strip()
        return text

    @staticmethod
    def _extract_candidate_phrases(text: str) -> List[str]:
        text = text or ""
        # prioritize quoted phrases and multi-word noun-ish spans
        quoted = re.findall(r'"([^"]{3,80})"', text)
        spans = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}(?:\s+[a-zA-Z][a-zA-Z\-]{2,}){0,3}\b", text)
        return [s.strip() for s in quoted + spans if s.strip()]

    @staticmethod
    def _keyword_priority(term: str) -> tuple:
        words = term.split()
        long_phrase = 1 if len(words) > 1 else 0
        has_digit = 1 if re.search(r"\d", term) else 0
        length_score = min(len(term), 60)
        return (long_phrase, has_digit, length_score)

    # --------------------------
    # Retrieval
    # --------------------------

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

        scores, indices = self.index.search(query_vec, top_k)

        requested_sections = item.get("sections_to_search") or []
        expanded_keywords = self._expand_keywords(item)

        candidates = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue

            chunk = self.metadata[idx]
            text = chunk.get("text", "").strip()
            section_type = chunk.get("section_type", "")

            # Remove noisy material before reranking.
            if section_type in {
                "references",
                "acknowledgements",
                "conflict_of_interest",
                "funding",
                "author_contributions",
            }:
                continue

            if len(text) < 80:
                continue

            section_bonus = self._section_match_bonus(
                chunk_section=chunk.get("section", ""),
                section_type=section_type,
                requested_sections=requested_sections,
            )

            keyword_bonus = self._keyword_overlap_bonus(
                text=text,
                keywords=expanded_keywords,
            )

            section_weight_bonus = self._section_weight_bonus(
                chunk.get("section_weight", 1.0)
            )

            adjusted = float(score) + section_bonus + keyword_bonus + section_weight_bonus

            candidates.append(
                {
                    "rank": rank,
                    "faiss_score": float(score),
                    "section_bonus": section_bonus,
                    "keyword_bonus": keyword_bonus,
                    "section_weight_bonus": section_weight_bonus,
                    "adjusted_retrieval_score": adjusted,
                    "chunk": chunk,
                }
            )

        candidates.sort(key=lambda x: x["adjusted_retrieval_score"], reverse=True)
        return candidates

    def _expand_keywords(self, item: Dict[str, Any]) -> List[str]:
        terms: Set[str] = set()
        for kw in self._select_query_keywords(item):
            terms.add(kw.lower())
            # lightweight inflection/format variants
            terms.add(kw.lower().replace("-", " "))
            terms.add(kw.lower().replace(" ", "-"))
            if kw.endswith("y"):
                terms.add(kw[:-1].lower() + "ies")
            if not kw.endswith("s"):
                terms.add(kw.lower() + "s")
        return sorted(t for t in terms if len(t) > 2)

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
    def _build_rerank_text(chunk: Dict[str, Any]) -> str:
        return (
            f"Title: {chunk.get('title', '')}\n"
            f"Section: {chunk.get('section', '')}\n"
            f"Section type: {chunk.get('section_type', '')}\n"
            f"Text: {chunk.get('text', '')}"
        ).strip()

    # --------------------------
    # Scoring helpers
    # --------------------------

    @staticmethod
    def _section_match_bonus(
        chunk_section: str,
        section_type: str,
        requested_sections: List[str],
    ) -> float:
        if not requested_sections:
            return 0.0

        section_text = f"{chunk_section} {section_type}".lower()

        aliases = {
            "abstract": ["abstract", "summary"],
            "introduction": ["introduction", "background"],
            "background": ["background", "introduction"],
            "objectives": ["objective", "objectives", "aim", "aims", "purpose"],
            "objective": ["objective", "objectives", "aim", "aims", "purpose"],
            "aim": ["aim", "aims", "purpose", "objective"],
            "aims": ["aim", "aims", "purpose", "objective"],
            "purpose": ["purpose", "aim", "objective"],
            "research question": [
                "research question",
                "research questions",
                "review question",
                "review questions",
            ],
            "review question": [
                "research question",
                "research questions",
                "review question",
                "review questions",
            ],
            "discussion": ["discussion"],
            "conclusion": ["conclusion", "conclusions"],
            "methods": ["methods", "methodology", "materials and methods"],
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
                if " " in kw_l:
                    bonus += 0.08
                else:
                    bonus += 0.03

        return min(0.5, bonus)

    @staticmethod
    def _section_weight_bonus(section_weight: float) -> float:
        try:
            weight = float(section_weight)
        except (TypeError, ValueError):
            weight = 1.0

        return max(-0.03, min(0.05, (weight - 1.0) * 0.05))

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

    # --------------------------
    # Output
    # --------------------------

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
            "query_text": query_text,
            "top_evidence": [
                {
                    "chunk_id": r["chunk"]["chunk_id"],
                    "doc_id": r["chunk"]["doc_id"],
                    "source_file": r["chunk"]["source_file"],
                    "title": r["chunk"]["title"],
                    "section": r["chunk"]["section"],
                    "section_type": r["chunk"]["section_type"],
                    "paragraph_ids": r["chunk"].get("paragraph_ids", []),
                    "faiss_score": r["faiss_score"],
                    "section_bonus": r["section_bonus"],
                    "keyword_bonus": r["keyword_bonus"],
                    "section_weight_bonus": r["section_weight_bonus"],
                    "adjusted_retrieval_score": r["adjusted_retrieval_score"],
                    "adjusted_retrieval_score_norm": r.get("adjusted_retrieval_score_norm"),
                    "rerank_score": r["rerank_score"],
                    "rerank_score_norm": r.get("rerank_score_norm"),
                    "final_score": r.get("final_score"),
                    "text": r["chunk"]["text"],
                }
                for r in reranked
            ],
            "decision_rules": item.get("decision_rules", {}),
        }

    # --------------------------
    # Model loading
    # --------------------------

    def _load_embedder(self) -> None:
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.config.embedding_model_name)

    def _load_reranker(self) -> None:
        if self.reranker is None:
            self.reranker = CrossEncoder(self.config.reranker_model_name)

    def _ensure_index_loaded(self) -> None:
        if self.index is None or not self.metadata:
            self.load()

    # --------------------------
    # IO helpers
    # --------------------------

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json(path: Path, obj: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _get_embedding_text(chunk: Dict[str, Any]) -> str:
        embedding_text = chunk.get("embedding_text")
        if embedding_text:
            return embedding_text

        return (
            f"Title: {chunk.get('title', '')}\n"
            f"Section: {chunk.get('section', '')}\n"
            f"Section type: {chunk.get('section_type', '')}\n"
            f"Text: {chunk.get('text', '')}"
        ).strip()

    @staticmethod
    def _metadata_from_chunk(faiss_id: int, chunk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "faiss_id": faiss_id,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source_file": chunk["source_file"],
            "title": chunk["title"],
            "section": chunk["section"],
            "section_type": chunk["section_type"],
            "section_weight": chunk.get("section_weight", 1.0),
            "paragraph_ids": chunk.get("paragraph_ids", []),
            "text": chunk["text"],
        }


if __name__ == "__main__":
    retriever = PrismaFaissRetriever(store_dir="faiss_store")

    if not Path("faiss_store/chunks.index").exists():
        retriever.build("output_grobid/all_chunks.jsonl")

    sample_item = {
        "standard": "PRISMA_2020",
        "section": "Introduction",
        "topic": "Objectives",
        "item_id": "PRISMA_2020_4",
        "item_label": "4",
        "checklist_item": "Provide an explicit statement of the objective(s) or question(s) the review addresses.",
        "sections_to_search": [
            "Introduction",
            "Objectives",
            "Research question",
            "Abstract",
            "Background",
            "Aim",
            "Aims",
            "Purpose",
            "Discussion",
            "Conclusion"
        ],
        "query_template": "Are the objectives or research questions explicitly stated? Provide evidence.",
        "keywords": [
            "objective",
            "objectives",
            "aim",
            "aims",
            "purpose",
            "research question",
            "research questions",
            "review question",
            "review questions",
            "we aimed to",
            "we aim to",
            "our aim was",
            "our objective was",
            "the aim of this review",
            "the objective of this review",
            "the purpose of this review",
            "this review aims",
            "this systematic review aimed",
            "PICO",
            "PECO",
            "PICOS"
        ],
        "evidence_requirements": [
            "Explicit objective/question statement; ideally structured."
        ],
        "decision_rules": {
            "YES": "Explicit objective(s) or question(s) stated clearly.",
            "PARTIAL": "General aim stated but lacks specificity.",
            "NO": "Objectives/questions not explicitly stated.",
            "NA": "Not applicable."
        }
    }

    result = retriever.query(
        sample_item,
        initial_top_k=50,
        final_top_k=10,
        debug=True,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
