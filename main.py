from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def extract_summary_data(text):
    data = {}

    # Extract Reconciliation Date
    match = re.search(r'\*\*Reconciliation Date:\*\* ([\d/]+)', text)
    if match:
        data["Reconciliation Date"] = match.group(1)

    # Extract Status
    match = re.search(r'\*\*Status:\*\* ([\w]+)', text)
    if match:
        data["Status"] = match.group(1)

    # Extract table rows
    table_pattern = r'\| (.*?) \| (.*?) \|'
    rows = re.findall(table_pattern, text)
    data["Table"] = [{"Description": desc.strip(), "Amount": amt.strip()} for desc, amt in rows]

    return data


load_dotenv()

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def pc_db(index_name: str):
#     # Check if the index already exists, otherwise create it
#     if not pc.has_index(index_name):
#         pc.create_index(
#             name=index_name,
#             dimension=384,
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud="aws",
#                 region="us-east-1"
#             ) 
#         )
#     # Connect to the index
#     index = pc.Index(index_name)
#     return index

# def pc_db(index_name: str):
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":"multilingual-e5-large",
                "field_map":{"text": "chunk_text"}
            }
        )

    # Connect to the index
    index = pc.Index(index_name)

    return index


# loader = PyPDFDirectoryLoader("pdfs")
loader = TextLoader("data/test.txt")

# print("Loader", loader)

# print("PDFs", loader.load())

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000,
    chunk_overlap=500,
    length_function=len,
    is_separator_regex=False,
)

# print("Data", data)

text_chunks = text_splitter.split_documents(data)


# print("Text chunks0:", text_chunks[0].page_content)
# print("\n\n\n\n\n\nText chunks1:", text_chunks[1].page_content)
# print("\n\n",len(text_chunks))
# print("\n\n\n\n\n\nText chunks2:", text_chunks[2].page_content)

texts = [chunk.page_content for chunk in text_chunks]

dense_index = pc_db("bank-rec")


# embeddings = [embed_model.embed_query(text) for text in texts]

# print("\n\n\nEmbeddings", embeddings)

# doc_search = dense_index.upsert(embeddings)

# print("Upserted successfully")

# print(dense_index.describe_index_stats())


def upsert_to_db(texts):

    embeddings = [embed_model.embed_query(text) for text in texts]

    print("Embeddings are ready.")

    vectors = [
        {
            "id": f"text-{i}",
            "values": embedding,
            "metadata": {"text": text}
        }
        for i, (embedding, text) in enumerate(zip(embeddings, texts))
    ]

    dense_index.upsert(vectors)

    print("Upserted successfully.")

    # print(dense_index.describe_index_stats())


def upsert_to_integrated_db(texts):
    data_to_upsert = [
        {
            "_id": f"text-{i}",
            "chunk_text": text,
        }
        for i, text in enumerate(texts)
    ]
    
    batch_size = 96
    for i in range(0, len(data_to_upsert), batch_size):
        batch = data_to_upsert[i:i+batch_size]
        dense_index.upsert_records("bank-reconcilation", batch)
        print(f"Upserted batch {i // batch_size + 1}")

    print("All batches upserted successfully.")

# upsert_to_db(texts)

upsert_to_integrated_db(get_texts())

def do_query(query):
    query_embedding = embed_model.embed_query(query)

    results = dense_index.query( vector=query_embedding, top_k=5, include_metadata=True)

    return results

def do_integrated_query(query: str):
        results = dense_index.search_records(
            namespace="bank-reconcilation", 
            query={
                "inputs": {"text": query}, 
                "top_k": 10
            }       
        )
        return results



# response = do_query("reconcilation date 06/30/2023")
# response = do_integrated_query("reconcilation date 06/30/2023")

# for hit in response['result']['hits']:
#     chunk_text = hit['fields']['chunk_text']
#     extracted_data = extract_summary_data(chunk_text)
#     print(extracted_data)

# print("Response", response)


