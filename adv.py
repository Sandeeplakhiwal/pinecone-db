from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import re
import json
from datetime import datetime
from uuid import uuid4
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, MetadataInfo,FilterOperator
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain_huggingface import HuggingFaceEmbeddings
import time


# Add this before creating the index
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
Settings.embed_model = embed_model



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



def filter_relevant_entries(text):
    """Extract only lines containing 'Outstanding Checks/Vouchers' or 'Cleared Checks/Vouchers'."""
    relevant_lines = []
    for line in text.split("\n"):
        if "Outstanding Checks/Vouchers" in line or "Cleared Checks/Vouchers" in line:
            relevant_lines.append(line)
    return "\n".join(relevant_lines)

FALLBACK_DATE = None

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





def generate_resp(query, reconcilation_date):
    chuk = get_chunks_with_metadata()

    # print("Chuk", chuk)

    # print(len(chuk))
    llm = Ollama(model="llama2:latest", temperature=0.0, context_window=1000, request_timeout=360)
    Settings.llm = llm

    index = VectorStoreIndex.from_documents(documents=chuk)
    filter = MetadataFilters(
        filters=[
            MetadataFilter(
                key="reconciliation_date", value=reconcilation_date
            ), MetadataFilter(key="uncleared", value="checks/vouchers")
        ]
    )

    # retriever = index.as_retriever()

    # retriever = VectorIndexRetriever(index=index, similarity_top_k=20, filters=filter)

    # nodes = retriever.retrieve(query)

    # print("Nodes", nodes)
   
    # qs_eng = RetrieverQueryEngine(retriever=retriever)
    qs_eng = index.as_query_engine()

    print("Came to queryEngine")

    _q = qs_eng.query(query)

    # chat_engine = index.as_chat_engine(llm=llm)

    # chat_res = chat_engine.chat("What is the capital of India?")

    print("Response is:", _q)
    return "response"

generate_resp("What is the reconcilation id?", "6/30/2023")

def chat_with(query):
    start_time = time.time()
    llm = Ollama(model="llama2:latest", temperature=0.0, context_window=1000, request_timeout=360)
    Settings.llm = llm

    res = llm.chat([ChatMessage(role="user", content=query)])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Response Time: {elapsed_time:.2f} seconds")
    print("Response", res)

