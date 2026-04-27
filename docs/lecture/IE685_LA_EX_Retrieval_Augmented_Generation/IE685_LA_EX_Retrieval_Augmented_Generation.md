# IE685 Exercise: Retrieval-Augmented Generation with LangChain

This notebook focuses on the retrieval side of RAG using the **SciFact** benchmark in **BEIR format**.

- Target course: **IE685 Large Language Models and Agents (FSS 2026)**
- Focus in this notebook: **chunking, indexing, retrieval, query expansion, filtering, reranking, and evaluation**

Documentation quick links:
- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain text splitters](https://python.langchain.com/api_reference/text_splitters/)
- [LangChain FAISS vector store](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
- [Sentence Transformers model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [BEIR repository](https://github.com/beir-cellar/beir)

---

## Why this lab matters

In the lecture, RAG was broken down into a sequence:
1. chunk documents
2. embed and index them
3. retrieve evidence
4. improve retrieval with query expansion or filtering
5. evaluate where retrieval still fails

That is exactly what we do here.

Design goal for this lab:
- keep the code inspectable,
- keep the dataset realistic,
- spend most time on retrieval behavior rather than setup friction.

---

## 1. Setup

This notebook assumes that the course zip already contains:
- the **SciFact benchmark** in `data/scifact/`
- teacher-precomputed assets in `data/scifact_assets/minilm_l6_v2_cs500_ov100/`

We still show the embedding code on a tiny sample, but for the full corpus we load the local FAISS index so the lab starts quickly.

### Install Required Packages

```python
%pip install -qU beir faiss-cpu==1.13.2 langchain==1.2.10 langchain-community==0.4.1 langchain-core==1.2.14 langchain-huggingface==1.0.1 langchain-openai==1.1.10 langchain-text-splitters==1.1.1 pypdf python-dotenv rank_bm25 sentence-transformers==5.2.2
```

### Import Libraries and Configure Paths

```python
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "scifact"
ASSET_DIR = ROOT / "data" / "scifact_assets" / "minilm_l6_v2_cs500_ov100"
CHUNK_PATH = ASSET_DIR / "chunks.jsonl"
FAISS_DIR = ASSET_DIR / "faiss"
MANIFEST_PATH = ASSET_DIR / "manifest.json"
PDF_PATH = ROOT / "data" / "rag_demo" / "mmds_exam_regulations_2024.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Data dir:", DATA_DIR)
print("Asset dir:", ASSET_DIR)
print("PDF path:", PDF_PATH)
```

**Output:**
```
Data dir: /Users/aaronsteiner/Documents/GitHub/llmsandagents/data/scifact
Asset dir: /Users/aaronsteiner/Documents/GitHub/llmsandagents/data/scifact_assets/minilm_l6_v2_cs500_ov100
PDF path: /Users/aaronsteiner/Documents/GitHub/llmsandagents/data/rag_demo/mmds_exam_regulations_2024.pdf
```

### Verify Required Assets

```python
required_paths = {
    "corpus": DATA_DIR / "corpus.jsonl",
    "queries": DATA_DIR / "queries.jsonl",
    "qrels_test": DATA_DIR / "qrels" / "test.tsv",
    "chunks": CHUNK_PATH,
    "faiss_index": FAISS_DIR / "index.faiss",
    "faiss_store": FAISS_DIR / "index.pkl",
    "manifest": MANIFEST_PATH,
    "demo_pdf": PDF_PATH,
}

status_rows = []
for name, path in required_paths.items():
    status_rows.append(
        {
            "artifact": name,
            "exists": path.exists(),
            "path": str(path),
        }
    )

pd.DataFrame(status_rows)
```

| artifact | exists | path |
|----------|--------|------|
| corpus | True | ... |
| queries | True | ... |
| qrels_test | True | ... |
| chunks | True | ... |
| faiss_index | True | ... |
| faiss_store | True | ... |
| manifest | True | ... |
| demo_pdf | True | ... |

> If one or more artifacts are missing, the expected instructor-side preparation step is:
> ```bash
> python scripts/prepare_scifact_rag_assets.py
> ```
> We do **not** run that preprocessing in class because embedding the full benchmark takes too long for a 90-minute slot.

---

## 2. Inspect the Benchmark

Before building retrieval, inspect what the benchmark actually contains:
- a `corpus` of scientific abstracts
- `queries` or claims
- `qrels` with relevance labels for evaluation

### Load the SciFact Dataset

```python
corpus, queries, qrels = GenericDataLoader(data_folder=str(DATA_DIR)).load(split="test")

print("Documents:", len(corpus))
print("Queries:", len(queries))
print("Queries with qrels:", len(qrels))
```

**Output:**
```
Documents: 5183
Queries: 300
Queries with qrels: 300
```

### Examine a Sample Query

```python
sample_query_id = next(iter(qrels))
sample_query = queries[sample_query_id]
gold_doc_ids = list(qrels[sample_query_id].keys())

print("Sample query id:", sample_query_id)
print("Claim / question:")
print(sample_query)
print("\nGold document ids:", gold_doc_ids[:5])

first_gold = corpus[gold_doc_ids[0]]
print("\nGold document title:")
print(first_gold.get("title", ""))
print("\nGold document text preview:")
print(first_gold.get("text", "")[:900] + "...")
```

**Output:**
```
Sample query id: 1
Claim / question:
0-dimensional biomaterials show inductive properties.

Gold document ids: ['31715818']

Gold document title:
New opportunities: the use of nanotechnologies to manipulate and track stem cells.

Gold document text preview:
Nanotechnologies are emerging platforms that could be useful in measuring, understanding, and manipulating stem cells. Examples include magnetic nanoparticles and quantum dots for stem cell labeling and in vivo tracking; nanoparticles, carbon nanotubes, and polyplexes for the intracellular delivery of genes/oligonucleotides and protein/peptides; and engineered nanometer-scale scaffolds for stem cell differentiation and transplantation. This review examines the use of nanotechnologies for stem cell tracking, differentiation, and transplantation. We further discuss their utility and the potential concerns regarding their cytotoxicity....
```

> **Try it out:**
> - Read the claim above and rewrite it manually in a way that might retrieve better evidence.
> - Which terms in the claim are likely to matter most for retrieval?

---

## 3. Chunking a Long PDF

The SciFact abstracts are too short to make chunking feel very important.
So lets use a genuinely longer document: the local MMDS exam regulations PDF.

We compare three practical chunking strategies:
1. page-based chunks,
2. fixed-size recursive chunks,
3. structure-aware recursive chunks.

Later, for the benchmark retrieval pipeline, we load the precomputed SciFact chunks from disk.

### Load PDF and Create Different Chunk Strategies

```python
pdf_loader = PyPDFLoader(str(PDF_PATH))
pdf_pages = pdf_loader.load()
pdf_pages = [p for p in pdf_pages if p.metadata.get("page", 0) >= 2] # Skip table of contents and cover page
pdf_full_text = "\n\n".join(
    f"[Page {page.metadata.get('page', 0) + 1}]\n{page.page_content}"
    for page in pdf_pages
)

page_chunks = pdf_pages

fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True,
)
fixed_chunks = fixed_splitter.create_documents(
    [pdf_full_text],
    metadatas=[{"source": str(PDF_PATH), "strategy": "fixed_recursive"}],
)

structured_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=[
        "\n§ ",
        "\nAbschnitt",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
    add_start_index=True,
)
structured_chunks = structured_splitter.create_documents(
    [pdf_full_text],
    metadatas=[{"source": str(PDF_PATH), "strategy": "structure_aware"}],
)

print("PDF page count (excl. table of contents):", len(pdf_pages))
print("Original length:", len(pdf_full_text), "characters")
print("Page-based chunks:", len(page_chunks))
print("Fixed recursive chunks:", len(fixed_chunks))
print("Structure-aware chunks:", len(structured_chunks))
```

**Output:**
```
PDF page count (excl. table of contents): 26
Original length: 110395 characters
Page-based chunks: 26
Fixed recursive chunks: 267
Structure-aware chunks: 192
```

### Compare Chunk Statistics

```python
def summarize_chunk_collection(name, docs):
    lengths = [len(doc.page_content) for doc in docs]
    return {
        "strategy": name,
        "chunks": len(docs),
        "avg_chars": round(float(np.mean(lengths)), 1),
        "median_chars": int(np.median(lengths)),
        "min_chars": int(np.min(lengths)),
        "max_chars": int(np.max(lengths)),
    }


chunk_strategy_summary = pd.DataFrame(
    [
        summarize_chunk_collection("page-based", page_chunks),
        summarize_chunk_collection("fixed recursive (500/100)", fixed_chunks),
        summarize_chunk_collection("structure-aware recursive (800/120)", structured_chunks),
    ]
)
chunk_strategy_summary
```

| strategy | chunks | avg_chars | median_chars | min_chars | max_chars |
|----------|--------|-----------|--------------|-----------|-----------|
| page-based | 26 | 4234.3 | 4469 | 341 | 5287 |
| fixed recursive (500/100) | 267 | 430.7 | 438 | 57 | 498 |
| structure-aware recursive (800/120) | 192 | 634.4 | 730 | 61 | 797 |

### Preview Chunks from Each Strategy

```python
def preview_chunks(name, docs, start_index=0, n=1):
    print(f"\n=== {name} ===")
    for i, doc in enumerate(docs[start_index:start_index+n], start=start_index+1):
        print(f"Chunk {i} | chars={len(doc.page_content)}")
        print(doc.page_content)
        print("-" * 100)


preview_chunks("Page-based", page_chunks)
preview_chunks("Fixed recursive", fixed_chunks)
preview_chunks("Structure-aware recursive", structured_chunks)
```

### Search for a Term Across Chunking Strategies

```python
target_term = "Nachteilsausgleich"

def first_chunk_containing(term, docs):
    term_lower = term.lower()
    for idx, doc in enumerate(docs):
        if term_lower in doc.page_content.lower():
            return idx, doc
    return None, None


for strategy_name, docs in [
    ("page-based", page_chunks),
    ("fixed recursive", fixed_chunks),
    ("structure-aware recursive", structured_chunks),
]:
    idx, doc = first_chunk_containing(target_term, docs)
    print(f"\n=== {strategy_name} ===")
    if doc is None:
        print(f"'{target_term}' not found.")
        continue
    print(f"First matching chunk index: {idx}")
    print(doc.page_content)
```

> **Try it out:**
> - Change the `chunk_size` and `chunk_overlap` of the fixed recursive splitter.
> - Add or remove separators in the structure-aware splitter.
> - Compare which strategy preserves a whole legal section best.
> - Search for a different term such as `Masterarbeit`, `Zulassung`, or `Prüfungsleistung`.

### Load Precomputed SciFact Chunks

```python
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
chunks = []
with CHUNK_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        chunks.append(record)

print("Chunk manifest:")
print(json.dumps(manifest, indent=2))
print("\nLoaded precomputed SciFact chunks:", len(chunks))
print("Example chunk metadata:", chunks[0]["metadata"])
```

**Output:**
```
Chunk manifest:
{
  "dataset_name": "scifact",
  "dataset_dir": "data/scifact",
  "split": "test",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunk_size": 500,
  "chunk_overlap": 100,
  "document_count": 5183,
  "chunk_count": 24678,
  "asset_dir": "data/scifact_assets/minilm_l6_v2_cs500_ov100",
  "chunk_path": "data/scifact_assets/minilm_l6_v2_cs500_ov100/chunks.jsonl",
  "embedding_path": "data/scifact_assets/minilm_l6_v2_cs500_ov100/chunk_embeddings.npy",
  "faiss_dir": "data/scifact_assets/minilm_l6_v2_cs500_ov100/faiss"
}

Loaded precomputed SciFact chunks: 24678
Example chunk metadata: {'doc_id': '4983', 'title': 'Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.', 'primary_topic': 'neuro', 'char_length': 1942, 'word_length': 295, 'source': 'beir_scifact', 'start_index': 0, 'chunk_index': 0, 'chunk_id': '4983-chunk-0000'}
```

---

## 4. Embeddings and Indexing

We use the local model `sentence-transformers/all-MiniLM-L6-v2`.
It runs on CPU and is a good classroom default for semantic retrieval.

For the **full corpus**, we load the precomputed FAISS index.
For intuition, we still embed a tiny sample live.

### Create Embedding Model

```python
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=False,
)

sample_texts = [
    queries[sample_query_id],
    chunks[0]["page_content"],
    chunks[1]["page_content"],
]
sample_vectors = embeddings.embed_documents(sample_texts)
print("Embedded", len(sample_vectors), "texts locally.")
print("Vector dimension:", len(sample_vectors[0]))
print("First five values:", np.array(sample_vectors[0][:5]))
```

**Output:**
```
Embedded 3 texts locally.
Vector dimension: 384
First five values: [-0.07656382  0.03013049  0.0247815  -0.01896523 -0.02126034]
```

### Load Precomputed FAISS Index

```python
vectorstore = FAISS.load_local(
    str(FAISS_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)
print("FAISS index loaded.")
```

---

## 5. Baseline Dense Retrieval

At this point we finally get the payoff:
use the claim as a query and inspect the top retrieved chunks.

Focus on:
1. the text that was retrieved,
2. the parent document ids,
3. whether those document ids match the gold evidence in `qrels`.

### Define Retrieval Helper Functions

```python
def get_unique_doc_ids(results):
    seen = set()
    ordered = []
    for doc, score in results:
        doc_id = str(doc.metadata["doc_id"])
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def print_retrieval_results(query, results, limit=3):
    print("Query:", query)
    print()
    for rank, (doc, score) in enumerate(results[:limit], start=1):
        meta = doc.metadata
        print(f"Rank {rank} | score={score:.4f} | doc_id={meta['doc_id']} | topic={meta.get('primary_topic', 'n/a')}")
        print("Title:", meta.get("title", ""))
        print(doc.page_content[:500].replace("\n", " "))
        print("-" * 100)


def dense_retrieve(query, k=5, fetch_k=None):
    fetch_k = fetch_k or k
    return vectorstore.similarity_search_with_score(query, k=fetch_k)
```

### Run Baseline Retrieval

```python
baseline_results = dense_retrieve(sample_query, k=5)
print_retrieval_results(sample_query, baseline_results, limit=5)

baseline_doc_ids = get_unique_doc_ids(baseline_results)
gold_doc_ids = list(qrels[sample_query_id].keys())

print("Retrieved doc ids:", baseline_doc_ids)
print("Gold doc ids:", gold_doc_ids)
```

**Output:**
```
Query: 0-dimensional biomaterials show inductive properties.

Rank 1 | score=1.2699 | doc_id=4346436 | topic=other
Title: Nonlinear Elasticity in Biological Gels
Unlike most synthetic materials, biological materials often stiffen as they are deformed. This nonlinear elastic response, critical for the physiological function of some tissues, has been documented since at least the 19th century, and the molecular structure and the design principles responsible for it are unknown. Current models for this response require geometrically complex ordered structures unique to each material. In this Article we show that a much simpler molecular theory accounts for
----------------------------------------------------------------------------------------------------
Rank 2 | score=1.2739 | doc_id=29638116 | topic=genetics
Title: Complex Tissue and Disease Modeling using hiPSCs.
Defined genetic models based on human pluripotent stem cells have opened new avenues for understanding disease mechanisms and drug screening. Many of these models assume cell-autonomous mechanisms of disease but it is possible that disease phenotypes or drug responses will only be evident if all cellular and extracellular components of a tissue are present and functionally mature. To derive optimal benefit from such models, complex multicellular structures with vascular components that mimic
----------------------------------------------------------------------------------------------------
Rank 3 | score=1.3306 | doc_id=6863070 | topic=other
Title: Three-dimensional superresolution colocalization of intracellular protein superstructures and the cell surface in live Caulobacter crescentus.
Recently, single-molecule imaging and photocontrol have enabled superresolution optical microscopy of cellular structures beyond Abbe's diffraction limit, extending the frontier of noninvasive imaging of structures within living cells. However, live-cell superresolution imaging has been challenged by the need to image three-dimensional (3D) structures relative to their biological context, such as the cellular membrane. We have developed a technique, termed superresolution by power-dependent
----------------------------------------------------------------------------------------------------
Rank 4 | score=1.3340 | doc_id=16532419 | topic=cancer
Title: Induction of stem-like cells with malignant properties by chronic exposure of human lung epithelial cells to single-walled carbon nanotubes
Induction of stem-like cells with malignant properties by chronic exposure of human lung epithelial cells to single-walled carbon nanotubes
----------------------------------------------------------------------------------------------------
Rank 5 | score=1.3359 | doc_id=4346436 | topic=other
Title: Nonlinear Elasticity in Biological Gels
Nonlinear Elasticity in Biological Gels
----------------------------------------------------------------------------------------------------
Retrieved doc ids: ['4346436', '29638116', '6863070', '16532419']
Gold doc ids: ['31715818']
```

> **Try it out:**
> - Rephrase the claim slightly and rerun retrieval.
> - Increase `k` from `5` to `10`.
> - Which retrieved chunk is semantically close but still not actually useful?

---

## 6. Lightweight Filtering and Reranking

Real systems often do more than one retrieval step.
Here we keep it simple:
- retrieve a larger candidate pool,
- optionally filter candidates,
- rerank with a cheap lexical-overlap heuristic.

This is not meant to be a production reranker. It is here so you can inspect how the ranking changes.

### Define Filtering and Reranking Functions

```python
def lexical_overlap_score(query, text):
    query_terms = {term for term in query.lower().split() if len(term) > 2}
    text_terms = set(text.lower().split())
    return len(query_terms & text_terms)


def filter_candidates(results, allowed_topics=None, title_keyword=None):
    filtered = []
    for doc, score in results:
        if allowed_topics and doc.metadata.get("primary_topic") not in allowed_topics:
            continue
        if title_keyword and title_keyword.lower() not in doc.metadata.get("title", "").lower():
            continue
        filtered.append((doc, score))
    return filtered


def rerank_candidates(query, results):
    rescored = []
    for doc, dense_score in results:
        overlap = lexical_overlap_score(query, doc.page_content)
        rerank_score = overlap - dense_score
        rescored.append((doc, dense_score, overlap, rerank_score))
    rescored.sort(key=lambda row: row[3], reverse=True)
    return rescored
```

### Apply Filtering and Reranking

```python
candidate_results = dense_retrieve(sample_query, fetch_k=12)
filtered_results = filter_candidates(candidate_results, allowed_topics={"covid", "infection", "other"})
reranked = rerank_candidates(sample_query, filtered_results)

for rank, (doc, dense_score, overlap, rerank_score) in enumerate(reranked[:5], start=1):
    print(
        f"Rank {rank} | doc_id={doc.metadata['doc_id']} | dense_score={dense_score:.4f} | "
        f"overlap={overlap} | rerank_score={rerank_score:.4f}"
    )
    print("Title:", doc.metadata.get("title", ""))
    print(doc.page_content[:320].replace("\n", " "))
    print("-" * 100)
```

**Output:**
```
Rank 1 | doc_id=4346436 | dense_score=1.2699 | overlap=1 | rerank_score=-0.2699
Title: Nonlinear Elasticity in Biological Gels
Unlike most synthetic materials, biological materials often stiffen as they are deformed. This nonlinear elastic response, critical for the physiological function of some tissues, has been documented since at least the 19th century, and the molecular structure and the design principles responsible for it are unknown. C
----------------------------------------------------------------------------------------------------
Rank 2 | doc_id=6863070 | dense_score=1.3306 | overlap=0 | rerank_score=-1.3306
Title: Three-dimensional superresolution colocalization of intracellular protein superstructures and the cell surface in live Caulobacter crescentus.
Recently, single-molecule imaging and photocontrol have enabled superresolution optical microscopy of cellular structures beyond Abbe's diffraction limit, extending the frontier of noninvasive imaging of structures within living cells. However, live-cell superresolution imaging has been challenged by the need to image
----------------------------------------------------------------------------------------------------
Rank 3 | doc_id=4346436 | dense_score=1.3359 | overlap=0 | rerank_score=-1.3359
Title: Nonlinear Elasticity in Biological Gels
Nonlinear Elasticity in Biological Gels
----------------------------------------------------------------------------------------------------
```

> **Try it out:**
> - Change the topic filter.
> - Remove filtering completely.
> - Increase the candidate pool from `12` to `20`.
> - Find one claim where filtering helps and one where it hurts.

---

## 7. Query Expansion

In the lecture, query expansion was presented as a way to improve recall.
Here we make it optional:
- if `OPENAI_API_KEY` is available, use a small LLM prompt,
- otherwise use a tiny manual fallback so the section still runs.

### Setup Query Expansion (with Optional LLM)

```python
llm = None
if os.getenv("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        temperature=0.2,
    )


def parse_expansion_lines(text):
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line:
            line = line.split(".", 1)[1].strip()
        lines.append(line)
    return lines


def expand_query(query):
    if llm is None:
        words = query.split()
        return [
            query,
            "scientific evidence " + query,
            "research studies " + " ".join(words[: min(6, len(words))]),
        ]

    prompt = f"""You are helping a retrieval system.
Generate 3 alternative search queries for the user query below.
Keep them concise and retrieval-oriented.

User query: {query}
"""
    response = llm.invoke(prompt)
    lines = parse_expansion_lines(response.content)
    return [query] + lines[:3]
```

### Test Query Expansion

```python
expanded_queries = expand_query(sample_query)
expanded_queries
```

**Output (fallback mode):**
```
['0-dimensional biomaterials show inductive properties.',
 '0D (zero-dimensional) nanomaterials osteoinductive properties',
 'zero-dimensional biomaterials electromagnetic inductance nanoparticles',
 'quantum dots bioinductive tissue regeneration 0D biomaterials']
```

### Retrieve with Expanded Queries

```python
def retrieve_with_expansion(query, per_query_k=4):
    all_results = []
    seen_chunk_ids = set()
    for expanded_query in expand_query(query):
        for doc, score in dense_retrieve(expanded_query, k=per_query_k):
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            all_results.append((expanded_query, doc, score))
    all_results.sort(key=lambda row: row[2])
    return all_results


expansion_results = retrieve_with_expansion(sample_query, per_query_k=4)
for rank, (expanded_query, doc, score) in enumerate(expansion_results[:6], start=1):
    print(f"Rank {rank} | via={expanded_query!r} | score={score:.4f} | doc_id={doc.metadata['doc_id']}")
    print(doc.metadata.get("title", ""))
    print(doc.page_content[:260].replace("\n", " "))
    print("-" * 100)
```

**Output:**
```
Rank 1 | via='"nanoparticles inductive effects biomedical applications"' | score=0.7938 | doc_id=31715818
New opportunities: the use of nanotechnologies to manipulate and track stem cells.
Nanotechnologies are emerging platforms that could be useful in measuring, understanding, and manipulating stem cells. Examples include magnetic nanoparticles and quantum dots for stem cell labeling and in vivo tracking; nanoparticles, carbon nanotubes, and po
----------------------------------------------------------------------------------------------------
Rank 2 | via='"nanoparticles inductive effects biomedical applications"' | score=0.8488 | doc_id=31715818
New opportunities: the use of nanotechnologies to manipulate and track stem cells.
New opportunities: the use of nanotechnologies to manipulate and track stem cells.
----------------------------------------------------------------------------------------------------
Rank 3 | via='"nanoparticles inductive effects biomedical applications"' | score=0.9560 | doc_id=10982689
Nanotoxicology: An Emerging Discipline Evolving from Studies of Ultrafine Particles
medicine, molecular biology, and bioinformatics, to name a few) is mandatory for nanotoxicology research to arrive at an appropriate risk assessment.
----------------------------------------------------------------------------------------------------
Rank 4 | via='"0D nanomaterials induce cellular responses mechanism"' | score=0.9982 | doc_id=4423327
Nanog safeguards pluripotency and mediates germline development
stem cells. Transient downregulation of Nanog appears to predispose cells towards differentiation but does not mark commitment. By genetic deletion we show that, although they are prone to differentiate, embryonic stem cells can self-renew indefinitely in the 
----------------------------------------------------------------------------------------------------
Rank 5 | via='"0D nanomaterials induce cellular responses mechanism"' | score=1.0412 | doc_id=4423327
Nanog safeguards pluripotency and mediates germline development
and germ cell states rather than in the housekeeping machinery of pluripotency. We surmise that Nanog stabilizes embryonic stem cells in culture by resisting or reversing alternative gene expression states.
----------------------------------------------------------------------------------------------------
Rank 6 | via='"0D nanomaterials induce cellular responses mechanism"' | score=1.0656 | doc_id=31715818
New opportunities: the use of nanotechnologies to manipulate and track stem cells.
cell differentiation and transplantation. This review examines the use of nanotechnologies for stem cell tracking, differentiation, and transplantation. We further discuss their utility and the potential concerns regarding its cytotoxicity.
----------------------------------------------------------------------------------------------------
```

> **Try it out:**
> - Change the expansion prompt.
> - Ask for 2 or 4 alternative queries instead of 3.
> - Compare narrow expansions against broad paraphrases.

---

## 8. Evaluation

We evaluate retrieval at the **document level** because SciFact qrels refer to relevant documents.
The retriever returns chunks, so we map chunks back to parent `doc_id`s.

### Define Evaluation Metrics

```python
def all_doc_level_ids_from_results(results):
    ordered_doc_ids = []
    seen = set()
    for item in results:
        if len(item) == 3:
            _, doc, _ = item
        else:
            doc, _ = item
        doc_id = str(doc.metadata["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered_doc_ids.append(doc_id)
    return ordered_doc_ids


def doc_level_ranking_from_chunks(results, k):
    return all_doc_level_ids_from_results(results)[:k]


def recall_at_k(ranked_doc_ids, gold_doc_ids, k):
    top_k = ranked_doc_ids[:k]
    return float(any(doc_id in gold_doc_ids for doc_id in top_k))


def reciprocal_rank(ranked_doc_ids, gold_doc_ids):
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(query_ids, retriever_fn, k=5):
    rows = []
    for query_id in query_ids:
        query = queries[query_id]
        gold_doc_ids = set(qrels.get(query_id, {}).keys())
        if not gold_doc_ids:
            continue
        results = retriever_fn(query)
        all_doc_ids = all_doc_level_ids_from_results(results)
        ranked_doc_ids = doc_level_ranking_from_chunks(results, k=k)
        rows.append(
            {
                "query_id": query_id,
                "recall_at_k": recall_at_k(ranked_doc_ids, gold_doc_ids, k=k),
                "rr": reciprocal_rank(ranked_doc_ids, gold_doc_ids),
                "unique_docs_fetched": len(all_doc_ids),
                "unique_docs_kept_at_k": len(ranked_doc_ids),
            }
        )
    return pd.DataFrame(rows)
```

### Run Evaluation Comparison

```python
eval_query_ids = list(qrels.keys())[:10]

baseline_eval = evaluate_retriever(
    eval_query_ids,
    retriever_fn=lambda query: dense_retrieve(query, k=5),
    k=5,
)

expansion_eval = evaluate_retriever(
    eval_query_ids,
    retriever_fn=lambda query: retrieve_with_expansion(query, per_query_k=3),
    k=5,
)

merged_counts = baseline_eval.merge(
    expansion_eval,
    on="query_id",
    how="inner",
    suffixes=("_baseline", "_expanded"),
)
merged_counts["extra_docs_fetched"] = (
    merged_counts["unique_docs_fetched_expanded"]
    - merged_counts["unique_docs_fetched_baseline"]
)

summary = pd.DataFrame(
    [
        {
            "setup": "baseline dense",
            "queries": len(baseline_eval),
            "Recall@5": baseline_eval["recall_at_k"].mean(),
            "MRR": baseline_eval["rr"].mean(),
            "Avg unique docs fetched": baseline_eval["unique_docs_fetched"].mean(),
        },
        {
            "setup": "query expansion",
            "queries": len(expansion_eval),
            "Recall@5": expansion_eval["recall_at_k"].mean(),
            "MRR": expansion_eval["rr"].mean(),
            "Avg unique docs fetched": expansion_eval["unique_docs_fetched"].mean(),
            "Avg extra docs vs baseline": merged_counts["extra_docs_fetched"].mean(),
            "Total extra docs vs baseline": merged_counts["extra_docs_fetched"].sum(),
        },
    ]
)

summary
```

| setup | queries | Recall@5 | MRR | Avg unique docs fetched | Avg extra docs vs baseline | Total extra docs vs baseline |
|-------|---------|----------|-----|-------------------------|---------------------------|------------------------------|
| baseline dense | 10 | 0.8 | 0.658333 | 3.8 | NaN | NaN |
| query expansion | 10 | 0.9 | 0.716667 | 4.1 | 0.3 | 3.0 |

### Analyze Per-Query Impact

```python
merged_eval = baseline_eval.merge(
    expansion_eval,
    on="query_id",
    how="inner",
    suffixes=("_baseline", "_expanded"),
)

improved = merged_eval[
    merged_eval["recall_at_k_expanded"] > merged_eval["recall_at_k_baseline"]
]
worsened = merged_eval[
    merged_eval["recall_at_k_expanded"] < merged_eval["recall_at_k_baseline"]
]

print("Queries improved by expansion:", len(improved))
print("Queries worsened by expansion:", len(worsened))
```

---

## 9. Optional Answer Generation

This is intentionally short.
The main lesson is that answer quality depends on retrieval quality.
If no API key is configured, skip this section.

```python
if llm is None:
    print("No OPENAI_API_KEY found. Skipping answer generation.")
else:
    top_context = "\n\n".join(doc.page_content for doc, _ in baseline_results[:3])
    answer_prompt = f"""Answer the question using only the context below.
If the context is insufficient, say so briefly.

Question: {sample_query}

Context:
{top_context}
"""
    answer = llm.invoke(answer_prompt)
    print(answer.content)
```

**Output (when no API key):**
```
No OPENAI_API_KEY found. Skipping answer generation.
```

---

## 10. Mini Challenge

Improve retrieval for one difficult claim by changing at least two of:
- query wording
- chunking parameters in the demo
- candidate pool size
- filtering
- query expansion prompt

Report:
1. what you changed
2. what improved
3. one failure that still remains

```python
challenge_query_id = sample_query_id
challenge_query = queries[challenge_query_id]

experiment = {
    "per_query_k": 4,
    "allowed_topics": None,
    "title_keyword": None,
}

challenge_candidates = dense_retrieve(challenge_query, fetch_k=15)
challenge_filtered = filter_candidates(
    challenge_candidates,
    allowed_topics=experiment["allowed_topics"],
    title_keyword=experiment["title_keyword"],
)
challenge_reranked = rerank_candidates(challenge_query, challenge_filtered)

print("Challenge query:")
print(challenge_query)
print("\nTop reranked doc ids:")
print([row[0].metadata["doc_id"] for row in challenge_reranked[:5]])
```

**Output:**
```
Challenge query:
0-dimensional biomaterials show inductive properties.

Top reranked doc ids:
['4346436', '29638116', '6863070', '16532419', '4346436']
```

---

## Wrap-up

You now implemented the core retrieval-side RAG workflow:
- chunking
- local embeddings
- FAISS indexing
- dense retrieval
- lightweight filtering and reranking
- query expansion
- document-level evaluation with `qrels`

The main takeaway is not that one setting is always best.
The main takeaway is that **retrieval quality is highly sensitive to concrete pipeline decisions**, and those decisions are inspectable.
