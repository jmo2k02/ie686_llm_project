---
document_type: enriched-markdown
source_pdf: IE685_LA_05_RetrievalAugmentedGeneration.pdf
page_count: 60
image_count: 17
extraction_tool: enrich_pdf_local.py
---

# Retrieval Augmented Generation

### IE685 Large Language Models and Agents

University of Mannheim

---

Retrieval Augmented Generation (RAG) combines **information retrieval** and LLM-based **text generation** by first retrieving relevant passages from a corpus of documents and then using them to generate an answer.

Gao, et al.: Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997, 2023.

---

## Outline

1. Motivation for RAG

2. The RAG Pipeline

   1. Chunking

   2. Embedding / Indexing

   3. Retrieval and Re-Ranking

   4. Query Expansion / Decomposition

   5. Generation

3. RAG Workflows and RAG Agents

   1. Workflow patterns

   2. Deep research agents

4. Evaluation of RAG

   1. Evaluation Metrics

   2. Benchmarks

---

## 1. Motivation for RAG

---

### Shortcoming of LLMs: Hallucinations

- LLM might generate convincing but wrong outputs
- RAG enables connecting output and evidence

---

### Long Tail Entities

- LLMs especially hallucinate for rare, long-tail entities that were hardly covered by their pre-training data

- RAG enables the retrieval of knowledge about rare entities from the Web or intranet

Mallen, et al. ACL 2023. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories

---

### Shortcoming of RAG

- On the other hand, RAG systems might also be misled by irrelevant evidence about the wrong entity

Ori Yoran, et al.: Making retrieval-augmented language models robust to irrelevant context. In ICLR 2024 workshop on large language model agents, 2024.

---

### Limitation of LLMs: Knowledge Cutoff

- As LLMs are trained at a specific point in time, they don't know anything about events after this knowledge cutoff

---

### Two Types of Knowledge

Mallen, et al. ACL 2023. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.

---

### Widespread Adoption of RAG in Industry

- RAG enables more efficient knowledge work using the **public Web**

- More efficient discovery and usage of **company internal information**

---

## 2. The RAG Pipeline

---

### RAG Architectures

Gao, et al.: Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997, 2023.

---

### 2.1. Chunking

- Split longer documents into chunks for embedding

- Chunking strategies:

  - **Fixed-size chunks**: Fixed number of tokens/words/characters and overlap

  - **Sentence-based chunks**: Group sentences together including some overlap

  - **Paragraph-based chunks**: Using paragraph breaks as chunk boundaries

  - **Semantic chunking**: Use LLM to split document into "coherent" content blocks

---

### Sparse and Dense Retrieval

- **Task**: Find the chunks that are most relevant for the query

- **Result**: Ranked list of the top-k chunks

See: IE663 Information Retrieval and Web Search

| **Sparse Retrieval** | **Dense Retrieval** |
|---------------------|---------------------|
| BM25 | Dense Embeddings |
| Lexical Matching | Semantic Matching |

---

### Dense Retrieval

**Approach:**

1. Embed chunks and query into the same embedding space

2. Perform nearest neighbor search in the embedding space

---

### Embedding

- Transform each chunk into an embedding vector

  - Dimensionality of vectors: 384 (SBERT) to 3072 (OpenAI large)

- Many embedding models exist:

  - OpenAI text-embedding-3-small (widely used for RAG, 1532 dimensions, ~6 KB per vector, $0.02/MTok)

  - OpenAI text-embedding-3-large (high-end, 3072 dimensions, ~12 KB per vector, $0.13/MTok)

  - Open weight models: MTEB leaderboard

  - 20+ integrations in LangChain

https://huggingface.co/spaces/mteb/leaderboard

---

### Generating Embedding Vectors

DPR (Karpukhin et al., 2020)

---

### Vector Stores

- Store embeddings for retrieval

- Offer approximate nearest neighbor search (ANN)

  - Similarity metrics: cosine, dot product

- Many indexing techniques exist:

  - Product quantization, locally sensitive hashing

  - Hierarchical navigable small world graphs

- Well-known vector stores:

  - FAISS (open source package by Facebook)

  - Weaviate (open-source vector store)

  - Pinecone (commercial, cloud-based store)

- 15+ integrations in LangChain

![Figure: A conceptual icon representing a database connected to a network or data distribution system.](IE685_LA_05_RetrievalAugmentedGeneration.assets/images/page-018-img-002-xref-107.jpeg)

*Description: A conceptual icon representing a database connected to a network or data distribution system. Database cylinder (storage), network branching lines (connectivity), circular nodes (endpoints).*

---

### Indexing Methods in FAISS

https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

---

### Sparse Retrieval

**Approach:**

1. Build inverted index for **terms** in chunks

2. Use inverted index to find most similar chunks for query

---

### Similarity Computation

- Bag-of-words representation combined with Jaccard similarity

- Jaccard similarity formula:
  ```
  similarity(x, y) = |tok(x) ∩ tok(y)| / |tok(x) ∪ tok(y)|
  ```

---

### Sparse Retrievers

- Computing weighted term scores

Robertson et al. 2009. The Probabilistic Relevance Framework: BM25 and Beyond.

---

### Comparison: Dense versus Sparse Retrieval

| **Criteria** | **Sparse Retrieval** | **Dense Retrieval** |
|---|---|---|
| **Matching** | Lexical | Semantic |
| **Out-of-domain** | Robust | Can degrade |
| **Rare entities** | Strong | Weak |
| **Paraphrase** | Weak | Strong |
| **Training** | None | Required |

- Both approaches can also be used in parallel

---

### LLM-Based Query Expansion

**Purpose**: Improve retrieval recall by generating **alternative phrasings** and related queries

**Role and Task:** You are part of a Retrieval-Augmented Generation (RAG) system. Your task is to improve document retrieval by generating multiple alternative search queries that capture the same information need as the user's question.

**User query:** "What are the main causes of coral reef bleaching?"

**Instructions:**

- Generate 4 alternative search queries.
- Use different terminology, synonyms, and related concepts.
- Focus on terms likely to appear in scientific or educational texts.
- Keep each query concise (5–12 words).

1. Causes of coral bleaching in marine ecosystems
2. Environmental factors leading to coral reef bleaching
3. Why do coral reefs lose color and die
4. Ocean warming and other drivers of coral bleaching

---

### LLM-Based Query Decomposition

**Purpose**: Improve retrieval by **splitting a complex question** into multiple more focused retrieval steps

**Role and Task:** You are part of a Retrieval-Augmented Generation (RAG) system. Your task is to decompose a complex user question into several simpler search queries that can be used to retrieve relevant documents.

**User question:** "How did the invention of the printing press influence the spread of scientific knowledge in Europe?"

**Instructions:**

- Break the question into 3–5 focused sub-queries.
- Each sub-query should target one specific aspect of the question.
- Each query should be suitable for document retrieval.

1. When and where was the printing press invented
2. How printing technology changed book production in Europe
3. Role of printing in spreading scientific ideas during the Renaissance
4. Examples of scientific works widely distributed through printing

---

### Hypothetical Document Generation

**Purpose**: Reduce semantic and structural gap between query and result documents to improve precision and recall.

1. Have LLM generate a document that could be the answer

2. Search with this answer instead of the actual query

---

### 2.3 Reranking and Filtering

- The ranking of chunks is often improved after retrieval

- Irrelevant chunks are filtered to not distract the LLM

---

### Reranking using Cross-Encoder

- Query and candidate passage are encoded jointly by a transformer model like BERT which is fine-tuned to predict relevance scores

- Input format: `[CLS] query [SEP] passage [SEP]`

- **Advantages:**

  - Captures fine-grained semantic interactions

  - Handles paraphrases and contextual relevance

- **Limitations:**

  - Computationally expensive

  - Typically applied to top 20–200 retrieved chunks

---

### Additional Ranking Criteria

- **Trustworthiness of Content**

  - Rank by source: whitelist or PageRank

  - Rank by content: overlap with ground truth (knowledge-based trust)

- **Timeliness of Content**

  - Filter outdated content by timestamp

- **Duplicate Content**

  - Employ LLM to remove chunks with duplicate content

  - Employ LLM to summarize similar chunks

Dong, et al.: Knowledge-based Trust: Estimating the Trustworthiness of Web Sources. Proc. VLDB Endow. 8, 9, 938–949. 2015.

**See: IE670 Web Data Integration: Information Quality**

---

### Multi-Stage Ranking

Typically, retrieval is done by a sequence of retrieval models:

1. **Candidate generation stage(s)**: Simple but fast approaches

   - Top 200 chunks using dense and/or sparse retrieval

   - Focus on recall

2. **Ranking stage(s)**: More accurate but slower methods

   - Filter by trust / timeliness

   - Re-rank candidates using cross-encoder

   - Focus on precision

- **Result**: Top-k chunks (e.g., 10 to 20)

---

### 2.4 Generation

- The retrieved chunks are added to the prompt and the LLM generates its answer using the additional context

---

## 3. RAG Workflows and RAG Agents

- Simple RAG will have problems with more complex tasks

| **Task** | **Complexity** | **RAG Approach** |
|---|---|---|
| "Who is the current PM of UK?" | Simple | Can be easily answered using simple RAG |
| "Create a table listing all previous UK Prime Ministers, including their terms in office, political party, alma mater, and notable achievements." | Medium | May require iterative retrieval based on intermediate generation |
| "Write a Wikipedia-style article on the role of the UK Prime Minister in the British government system." | Complex | May require planning, followed by focused retrieval and section-wise refinement |

---

### RAG Architectures

Gao, et al.: Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997, 2023.

---

### RAG Workflow using Parallelization: GPT Researcher

- Early example of a RAG workflow that employs Web search to generate structured reports

  - Length: several pages

  - Including citations

- Open-source project

https://gptr.dev/

---

### RAG Workflow using Self-Reflection

Asai et al.: Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflections. ICLR 2024.

---

### Example of a Critique Step

- LLM identifies irrelevant chunks after retrieval and filters them out.

---

### Generator-Evaluator Pattern: Tavily Variant of GPT Researcher

- Includes feedback loop in each retrieval task

https://docs.tavily.com/examples/opensources/gpt-researcher

---

### Deep Research Agents

---

### Deep Research Agents apply the ReAct Pattern to Information Gathering

Li et al. 2025. WebThinker: Empowering Large Reasoning Models with Deep Research Capacity.

---

### Architecture of Deep Research Agents

Huang, et al.: Deep Research Agents: A Systematic Examination And Roadmap. arXiv:2506.18096, 2025.

Zhang, et al.: Deep Research: A Survey of Autonomous Research Agents. arXiv:2508.12752, 2025.

---

### Specialized RAG Task: Fact Checking

- Fact checking: Verification of the factual accuracy of claims (often by politicians)

  - FactCheck.org, Washington Post publish claim reviews on the Web

- Current trend: Automated fact checking using RAG pipelines

Dmonte, et al.: Claim Verification in the Age of Large Language Models: A Survey. Arxiv, 2025.

Schlichtkrull, et al.: The Automated Verification of Textual Claims (AVeriTeC) Shared Task. ACL, 2024.

---

## 4. Evaluation of RAG

- **Reference-based approaches**: Compare results to ground truth

- **Reference-free approaches**: Rely exclusively on LLM-as-a-Judge

---

### Retrieval: Reference-based Metrics

- Require **ground truth**: Which chunks are relevant?

- nDCG@10 is widely used (e.g., BEIR)

---

### Generation: Reference-based Metrics

- Require correct answers as **ground truth**

- **Scoring**:

  - Syntactic text metrics: **BLEU, ROUGE**

  - Semantic text metrics: **BERTscore, LLM-as-a-Judge**

---

### Reference-free Metrics: RAG Triad

- Employ LLM-as-a-Judge **without** ground truth

- Evaluation results may be noisy

- Widely used as cheaper and more flexible than reference-based evals

---

### Context Relevance

- Measures answer quality: **Are the retrieved chunks relevant for the query?**

  - As irrelevant chunks might lead to hallucinations

- LLM-as-a-judge prompt to assess context relevance:

**Role:** You are a judge evaluating the relevance of a retrieved context to a user question.

**Question:** {question}

**Context:** {retrieved chunk(s)}

**Instructions:** Evaluate whether the context is relevant for answering the question. Return a score from 0 to 1 where:

- 0 = completely irrelevant
- 1 = fully relevant and useful for answering the question

Explain briefly before giving the score.

---

### Groundedness / Faithfulness

- Evaluates whether the **generated answer is supported by the retrieved context** (chunks)

  - As LLM might use parametric knowledge for the answer

  - Sometimes evaluated for each sentence of answer separately

- LLM-as-a-judge prompt to assess groundedness:

**Role:** You are evaluating whether an answer is grounded in the provided context.

**Question:** {question}

**Context:** {context}

**Answer:** {answer}

**Instructions:** Determine whether the answer is supported by the context. Score from 0 to 1 where:

- 0 = answer contains claims not supported by the context
- 1 = every claim in the answer is supported by the context

---

### Answer Relevance

- Evaluates whether the **answer actually addresses the query**

  - As LLM might be distracted by the context

- LLM-as-a-judge prompt to assess answer relevance:

**Role:** You are evaluating how well an answer addresses a user question.

**Question:** {question}

**Answer:** {answer}

**Instructions:** Score the relevance of the answer to the question. Score from 0 to 1 where:

- 0 = the answer does not address the question
- 1 = the answer fully and directly answers the question

Provide reasoning followed by the score.

---

### RAG Benchmarks

1. Retrieval Benchmarks

2. End-to-End RAG Benchmarks

3. Fact Verification Benchmarks

4. Benchmarks for Deep Research Agents

---

### Retrieval Benchmark: BEIR

- Evaluates retrieval models across many datasets

  - Reference-based: Ranking of retrieved documents is compared to ground truth relevance judgements

  - Does not evaluate generation

Thakur et al. 2021. NeurIPS D&B. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.

---

### MTEB

- Embedding benchmark covering wider range of NLP tasks

- Up-to-date MTEB **leaderboard** for open-weight models

https://huggingface.co/spaces/mteb/leaderboard

---

### End-to-End RAG Benchmarks

Classic open-domain question answering (Q&A) benchmarks are widely used to evaluate RAG systems:

| **Benchmark** | **Natural Questions (NQ)** | **HotpotQA** |
|---|---|---|
| **Year** | 2019 | 2018 |
| **Source of Questions** | Real Google search queries | Crowdsourced questions |
| **Knowledge Source** | Wikipedia | Wikipedia |
| **Question Type** | Single-hop factual questions | Multi-hop reasoning questions |
| **Retrieval Requirement** | Retrieve one relevant passage/article | Retrieve multiple supporting documents |
| **Key Challenge** | Large-scale open-domain retrieval | Cross-document reasoning |
| **Typical Metrics** | Exact Match, F1 | Exact Match, F1 |
| **RAG Evaluation Role** | Tests retrieval + answer generation | Tests multi-hop retrieval + reasoning + generation |

---

### Q&A Benchmarks get harder…

- OpenAI Web RAG benchmark containing hard-to-solve but easy-to-verify questions

---

### Fact Verification Benchmarks

| **Feature** | **FEVER** | **AVERITEC** |
|---|---|---|
| **Publication** | 2018 | 2024 |
| **Main Goal** | Verify factual claims using Wikipedia evidence | Verify real-world claims using web evidence |
| **Claim Source** | Synthetic claims generated from Wikipedia sentences | Real claims from journalists, fact-checking organizations, and public discourse |
| **Knowledge Source** | Wikipedia snapshot | Open web retrieval |
| **Verdict Labels** | Supported, Refuted, Not Enough Info | Supported, Refuted, Conflicting, Not Enough Evidence |
| **Evidence Format** | One or more Wikipedia sentences | Evidence passages from multiple webpages |
| **Reasoning Complexity** | Often single or few-sentence reasoning | Multi-hop reasoning across sources |
| **Evaluation** | Label accuracy + evidence retrieval (FEVER score) | Claim verdict + correctness of evidence and QA chain |

Thorne, et al.: FEVER: A Large-scale Dataset for Fact Extraction and VERification. NAACL, 2018.

Schlichtkrull, et al.: The Automated Verification of Textual Claims (AVeriTeC) Shared Task. ACL, 2024.

---

### Benchmarks for Deep Research Agents

- Employ LLM-as-a-Judge together with **rubric-based ground truth** to evaluate the quality of long-form generated texts

- Rubrics specify evaluation dimensions and scoring guidelines, allowing evaluators (humans or LLM judges) to rate system outputs consistently

---

### Rubric-based Benchmarks

---

### Evaluating Citation Accuracy

---

## Summary

- Deep research agents are a hot topic in research and industry

- And a good topic for student projects in this course

Huang, et al.: Deep Research Agents: A Systematic Examination And Roadmap. arXiv:2506.18096, 2025.

Zhang, et al.: Deep Research: A Survey of Autonomous Research Agents. arXiv:2508.12752, 2025.

---

## See you next week!

- Next time: Context Engineering

---

## Credits

- This slide set is based on slides from:

  - Akari Asai

  - Daphne Ippolito

  - Fernando Diaz

- Many thanks to all of you!