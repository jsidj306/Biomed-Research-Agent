"""Step 4.2 — ChromaDB verification script"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

import os

import chromadb

chroma_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_test")
chroma_path = os.path.abspath(chroma_path)

print(f"ChromaDB data path: {chroma_path}")

# 1. Create persistent client
client = chromadb.PersistentClient(path=chroma_path)
print("[OK] PersistentClient created")

# 2. Get or create collection
collection = client.get_or_create_collection(name="test_biomed")
print(f"[OK] Collection 'test_biomed' ready (count: {collection.count()})")

# 3. Insert biomedical docs
docs = [
    "CRISPR-Cas9 is a revolutionary gene editing technology that enables precise DNA modifications",
    "CAR-T cell therapy uses engineered T cells to fight cancer by targeting specific antigens",
    "Single-cell RNA sequencing reveals cellular heterogeneity in tumor microenvironments"
]
collection.add(documents=docs, ids=["d1", "d2", "d3"])
print(f"[OK] Added {len(docs)} documents")

# 4. Query
results = collection.query(
    query_texts=["gene editing tools for cancer treatment"],
    n_results=2
)

print("\nQuery: 'gene editing tools for cancer treatment'")
print("Results:")
for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"  {i+1}. (distance={dist:.4f}) {doc[:80]}...")

print("\n=== ChromaDB is working! ===")
