"""Step 4.4 — RAG Minimal Demo: Search -> Store -> Retrieve -> AI Answer"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

import os

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

import chromadb
from Bio import Entrez
from openai import OpenAI

# ===== Config =====
Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

chroma_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_demo")
chroma_path = os.path.abspath(chroma_path)

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") + "/v1"
)

QUERY = "What are the main challenges of CRISPR-Cas9 in cancer treatment?"

# ===== Step 1: Search PubMed =====
print(f"[1/4] Searching PubMed: '{QUERY}'")
handle = Entrez.esearch(db="pubmed", term=QUERY, retmax=3, sort="relevance")
record = Entrez.read(handle)
handle.close()

paper_ids = record["IdList"]
print(f"  Found {record['Count']} papers, using top 3: {paper_ids}")

# Fetch abstracts
papers = []
for pid in paper_ids:
    handle = Entrez.efetch(db="pubmed", id=pid, rettype="xml", retmode="xml")
    data = Entrez.read(handle)
    handle.close()
    article = data["PubmedArticle"][0]["MedlineCitation"]["Article"]
    title = article["ArticleTitle"]
    abstract_list = article.get("Abstract", {}).get("AbstractText", ["No abstract"])
    abstract = " ".join(abstract_list) if isinstance(abstract_list, list) else str(abstract_list)
    papers.append({"id": pid, "title": title, "abstract": abstract})
    print(f"  PMID {pid}: {title[:80]}...")

# ===== Step 2: Store in ChromaDB =====
print("\n[2/4] Storing in ChromaDB...")
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection(name="demo_rag")

documents = [f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers]
ids = [p["id"] for p in papers]
collection.add(documents=documents, ids=ids)
print(f"  Stored {len(documents)} papers")

# ===== Step 3: Retrieve =====
print("\n[3/4] Retrieving relevant context...")
results = collection.query(query_texts=[QUERY], n_results=3)
context_parts = []
for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"  Result {i+1} (distance={dist:.4f})")
    context_parts.append(doc)

context = "\n\n---\n\n".join(context_parts)

# ===== Step 4: Generate Answer =====
print("\n[4/4] Generating answer via LLM...")
prompt = f"""You are a biomedical research assistant. Answer the question based ONLY on the provided paper abstracts.

Context:
{context}

Question: {QUERY}

Provide a concise answer (3-5 bullet points) with citations to PMIDs."""

response = llm.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500,
    temperature=0.3
)

answer = response.choices[0].message.content

print(f"\n{'='*60}")
print(f"Q: {QUERY}")
print(f"{'='*60}")
print(f"A:\n{answer}")
print(f"{'='*60}")
print("Sources:")
for p in papers:
    print(f"  - PMID {p['id']}: {p['title']}")

print("\n=== RAG Pipeline Complete! ===")
