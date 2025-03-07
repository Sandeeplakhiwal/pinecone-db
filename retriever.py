from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.llms import openai
# from langchain_community.vectorstores import Pinecone
from langchain.chains import retrieval_qa
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
import json
from md import get_texts
import re
from datetime import datetime


load_dotenv()


embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


FALLBACK_DATE = "01/01/1900"

def extract_reconciliation_date(text):
    match = re.search(r"Reconciliation Date:\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    if match:
        date_str = match.group(1)
        try:
            datetime.strptime(date_str, "%m/%d/%Y")
            return date_str
        except ValueError:
            pass
    return None


def get_chunks_with_metadata():
    # Load text
    loader = TextLoader("data/test.txt")
    data = loader.load()
    full_text = "\n\n".join([doc.page_content for doc in data])

    # Extract the date
    reconciliation_date = FALLBACK_DATE

    # Markdown splitting
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Heading1"), ("##", "Heading2"), ("###", "Heading3"), ("####", "Heading4")])
    text_chunks = header_splitter.split_text(full_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Tune as needed
        chunk_overlap=200
    )

    small_chunks = []

    for chunk in text_chunks:
        splits = recursive_splitter.split_text(chunk.page_content)
        small_chunks.extend(splits)

    # Further character splitting
    final_chunks = []
    last_known_date = FALLBACK_DATE

    for chunk in small_chunks:
        current_date = extract_reconciliation_date(chunk)
        if current_date:
            last_known_date = current_date  # Update with the latest found date
            print("Current Date/Last known date", current_date)
        
        final_chunks.append({
            "text": chunk,
            "metadata": {"reconciliation_date": last_known_date}
        })
            
    print("Chunks created.")
    return final_chunks


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def pc_db(index_name: str):
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            deletion_protection="disabled",
            tags={
                "environment": "development"
            }
        )
        print("PC created.")

    # Connect to the index
    index = pc.Index(index_name)
    return index

dense_index = pc_db("bank-rec")


def upsert_to_db(chunks):
    vectors = []
    
    for i, chunk in enumerate(chunks):
        embedding = embed_model.embed_query(chunk["text"])
        
        vectors.append({
            "id": f"text-{i}",
            "values": embedding,  # single embedding per chunk
            "metadata": {
                "reconciliation_date": chunk["metadata"]["reconciliation_date"],
                "text": chunk["text"][:1000]  # Optional: store first 1000 chars of text
            }
        })

    print(f"{len(vectors)} vectors prepared. Upserting...")

    batch_size = 96
    for i in range(0, len(vectors), batch_size):
        dense_index.upsert(vectors=vectors[i:i+batch_size])
        print(f"Upserted batch {i // batch_size + 1}")

    print("All batches upserted successfully.")
   

def do_integrated_query(query: str):
        results = dense_index.search_records(
            namespace="bank-reconcilation", 
            query={
                "inputs": {"text": query}, 
                "top_k": 10
            }       
        )
        return results

def do_query(query, reconciliation_date=None):
    query_embedding = embed_model.embed_query(query)

    filter_conditions = {}

    if reconciliation_date:
        filter_conditions["reconciliation_date"] = reconciliation_date

    results = dense_index.query(  vector=query_embedding, top_k=5, include_metadata=True, filter={
        "reconciliation_date": {"$eq": reconciliation_date}
    })

    return results



# upsert_to_db(get_chunks_with_metadata())

response = do_query(
    query="reconcilation date 6/30/2023",
    reconciliation_date="6/30/2023"
)

print("Response", response)