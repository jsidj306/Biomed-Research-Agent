"""Step 4.3 — PubMed API verification script"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

import os

from Bio import Entrez
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

print(f"Email: {Entrez.email}")
print(f"API Key: {'[OK] set' if Entrez.api_key else '[FAIL] missing'}\n")

# 1. Search
term = "CRISPR-Cas9 cancer therapy"
print(f"Searching PubMed for: '{term}'...")
handle = Entrez.esearch(db="pubmed", term=term, retmax=5, sort="relevance")
record = Entrez.read(handle)
handle.close()

print(f"[OK] Found {record['Count']} papers")
print(f"Top 5 IDs: {record['IdList']}\n")

# 2. Fetch first paper details
first_id = record["IdList"][0]
print(f"Fetching details for PMID: {first_id}...")
handle = Entrez.efetch(db="pubmed", id=first_id, rettype="xml", retmode="xml")
papers = Entrez.read(handle)
handle.close()

article = papers["PubmedArticle"][0]["MedlineCitation"]["Article"]
title = article["ArticleTitle"]
abstract_list = article.get("Abstract", {}).get("AbstractText", ["No abstract available"])
abstract = " ".join(abstract_list) if isinstance(abstract_list, list) else str(abstract_list)

print(f"\nTitle: {title}")
print(f"Abstract: {abstract[:300]}...\n")

print("=== PubMed API is working! ===")
