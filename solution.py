from prisma_retriever import PrismaFaissRetriever
import json

retriever = PrismaFaissRetriever(store_dir="faiss_store")

prisma_item_dict = {
    "standard": "PRISMA_2020",
    "section": "Title",
    "topic": "Title",
    "item_id": "PRISMA_2020_1",
    "item_label": "1",
    "checklist_item": "Identify the report as a systematic review.",
    "sections_to_search": [
      "Title",
      "Front matter",
      "Abstract",
      "Header",
      "Citation"
    ],
    "query_template": "Does the title (or header) explicitly identify the report as a systematic review (and/or meta-analysis)? Provide evidence from the title.",
    "keywords": [
      "systematic review",
      "meta-analysis",
      "review",
      "systematic literature review",
      "evidence synthesis",
      "systematic reviews",
      "systematic scoping review",
      "meta analysis",
      "metaanalysis",
      "quantitative synthesis",
      "qualitative synthesis",
      "review protocol",
      "overview of reviews",
      "umbrella review"
    ],
    "evidence_requirements": [
      "Title contains explicit phrase indicating systematic review (or equivalent)."
    ],
    "decision_rules": {
      "YES": "Title explicitly identifies the report as a systematic review (or meta-analysis).",
      "PARTIAL": "Title implies a review but does not clearly state systematic review/meta-analysis.",
      "NO": "Title does not indicate it is a systematic review.",
      "NA": "Not applicable (rare; typically always applicable)."
    },
    "query_template_expanded": "Does the title (or header) explicitly identify the report as a systematic review (and/or meta-analysis)? Provide evidence from the title. Search for exact phrases, synonyms, nearby section headings, and explicit negative statements. Return concise evidence quotes with section names and explain whether the evidence satisfies the checklist item.",
    "evidence_requirements_expanded": [
      "Title contains explicit phrase indicating systematic review (or equivalent).",
      "Prefer explicit statements over inferred evidence.",
      "Accept evidence from the primary target sections first, then abstract/supplementary material if relevant.",
      "Use section headings, nearby context, and exact terminology to judge whether the item is fully reported.",
      "If information is absent, distinguish between not reported and not applicable.",
      "When evidence is partial, identify the missing element needed for a YES decision."
    ]
  }

result = retriever.query(
    prisma_item_dict,
    initial_top_k=75,
    final_top_k=5
)

with open("retriever_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("Saved to retriever_result.json")