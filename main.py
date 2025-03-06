from langchain_community.document_loaders import PyPDFDirectoryLoader
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

load_dotenv()

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def pc_db(index_name: str):

    # Check if the index already exists, otherwise create it
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            # cloud="aws",
            # region="us-east-1",
            # embed={
            #     "model": "llama-text-embed-v2",
            #     "field_map": {"text": "chunk_text"}
            # }
            # embed=embed_model
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

    # Connect to the index
    index = pc.Index(index_name)

    return index


loader = PyPDFDirectoryLoader("pdfs")

# print("PDFs", loader.load())

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

print("Data", data)

text_chunks = text_splitter.split_documents(data)


print("Text chunks0:", text_chunks[0].page_content)
print("\n\n\n\n\n\nText chunks1:", text_chunks[1].page_content)
print("\n\n",len(text_chunks))
print("\n\n\n\n\n\nText chunks2:", text_chunks[2].page_content)

texts = [chunk.page_content for chunk in text_chunks]

dense_index = pc_db("sandeep")


# embeddings = [embed_model.embed_query(text) for text in texts]

# print("\n\n\nEmbeddings", embeddings)

# doc_search = dense_index.upsert(embeddings)

# print("Upserted successfully")

# print(dense_index.describe_index_stats())


def upsert_to_db(texts):

    embeddings = [embed_model.embed_query(text) for text in texts]

    print("Embeddings are ready.")

    dense_index.upsert(embeddings)

    print("Upserted successfully.")

    print(dense_index.describe_index_stats())


upsert_to_db(texts)

