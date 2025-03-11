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
import re
from datetime import datetime
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings,VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from uuid import uuid4
load_dotenv()


embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

Settings.embed_model = embed_model

FALLBACK_DATE = None

def extract_reconciliation_date(text):
    match = re.search(r"\**Reconciliation Date:\**\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    if match:
        date_str = match.group(1)
        try:
            datetime.strptime(date_str, "%m/%d/%Y")
            return date_str
        except ValueError:
            pass
    return None







# # 
# MARKDOWN_SEPARATORS = [
#     "\n#{1,6} ",
#     "```\n",
#     "\n\\*\\*\\*+\n",
#     "\n---+\n",
#     "\n___+\n",
#     "\n\n",
#     "\n",
#     " ",
#     "",
# ]



def get_chunks_with_metadata():
    # Load text
    # loader = TextLoader("test.txt")
    # data = loader.load()

    with open("data/test.txt" , "r") as f:
        data = f.read()
    # print(data, "dat")
    # full_text = "\n\n".join([doc.page_content for doc in data])

    # Markdown splitting
    # header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "Heading1")])
    # text_chunks = header_splitter.split_text(full_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200,
        is_separator_regex=False,
        separators="\n\n"
        # separators=MARKDOWN_SEPARATORS
        # separators="|"
    )

    # small_chunks = []

    # for chunk in data:
    #     splits = recursive_splitter.split_text(chunk.page_content)
    #     small_chunks.extend(splits)

    # Further character splitting

    small_chunks = recursive_splitter.create_documents([data])
    final_chunks = []
    last_known_date = FALLBACK_DATE

    # print("--------",small_chunks[0], "pringint ", "--------END-------")
    value_text = "None"
    for chunk in small_chunks:
        unclread_pattern = "Outstanding Checks/Vouchers"
        clread_pattern = "Cleared Checks/Vouchers"


        matches = re.findall(unclread_pattern, chunk.page_content)
        matches1 = re.findall(clread_pattern, chunk.page_content)

        # value_text = "checks/vouchers" if len(matches) > 0 else "cleared" if len(matches1) else "none"

        if(len(matches) > 0):
            value_text = "checks/vouchers"
        
        if(len(matches1) > 0):
            value_text = "cleared"

        # print(matches, "outstaind ")
        current_date = extract_reconciliation_date(chunk.page_content)
        if current_date:
            last_known_date = current_date  
            # print("Current Date/Last known date", current_date)
        
        final_chunks.append(Document(
            _doc_id =  uuid4(),
            metadata = {"reconciliation_date": last_known_date, "uncleared": value_text},
            text = chunk.page_content,
        #     {
        #     "doc_id": uuid4(),
        #     "text": chunk,
        #     "metadata": 
        # }
        ))
            
    # print("Chunks created.")
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
                "text": chunk["text"][:1000]  
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

# response = do_query(
#     query="reconcilation date 6/30/2023",
#     reconciliation_date="6/30/2023"
# )

# print("Response", response)
# dense_index.
system_prompt = '''
you are a Financial intelligent agent. You have a Financial data in the structured format. So you are Giving the answer as per the user query. Bases of query findout the relevant answer from the data and provide the similar response in a json format 
'''

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, MetadataInfo,FilterOperator
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
def generate_response(query, reconcilation_date):
    # results = do_query(query, reconcilation_date)
    
    # print("Query Results:", len(results["matches"]), results)  # Debugging line
    # print(results)
    # if not results.get("matches"):  # Check if results exist
    #     print("No matches found.")
    #     return "No relevant data found."

    # context = "\n\n".join([r["metadata"]["text"] for r in results["matches"]])

    # print("Context is:", context)

    # prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # response = ollama.chat(model="deepseek-r1:latest", messages=[{"role": "user", "content": prompt}])
    chuk = get_chunks_with_metadata()

    # print(len(chuk))
    llm = Ollama(model="llama2:latest", temperature=0.0, context_window=1000)
    Settings.llm = llm

    # vec_store = PineconeVectorStore(pinecone_index=dense_index)
    index = VectorStoreIndex.from_documents(documents=chuk)
    filter = MetadataFilters(
        filters=[
            MetadataFilter(
                key="reconciliation_date", value=reconcilation_date
            ), MetadataFilter(key="uncleared", value="checks/vouchers")
        ]
    )

    # print(chuk[70].text)
    # print(chuk[70].metadata['reconciliation_date'])
    # print

    # for ch in chuk:
    #     if(ch.metadata['reconciliation_date'] == reconcilation_date
    #        ):
    #         print(ch.text)
    #         print(ch.metadata)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=20, filters=filter)
    nodes = retriever.retrieve(query)

    
   
    qs_eng = RetrieverQueryEngine(retriever=retriever)
    _q = qs_eng.query(query)
    # print(retriever.retrieve("outstanding checks/vouchers"), "nodes")
    # qs_eng = index.as_query_engine(similarity_top_k=5, filter=filter, llm=llm)
    # _query = qs_eng.query(query)
    print(_q)
    return "response"



generated_response = generate_response("extract all outstanding checks/vouchers in json format", "6/30/2023")

# print("Generated response: ", generated_response)
# print(len(get_chunks_with_metadata()))
# for ch in get_chunks_with_metadata():
#     # print(ch.text)
#     # print(ch.metadata)
        
#         if(ch.metadata['reconciliation_date'] == "6/30/2023"
#            and ch.metadata['uncleared'] == "checks/vouchers"
#            ):
#             print(ch.text)
#             print(ch.metadata)