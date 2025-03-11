from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone,ServerlessSpec
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

load_dotenv()

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text = extract_text("./pdfs/hess201.pdf")
print("text extracted...")

def create_text_chunks(text, chunk_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,  chunk_overlap=overlap,  separators=["\n\n", "\n", " "])
    chunks = text_splitter.split_text(text)
    print("chunks created...")
    return chunks


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


dense_index = pc_db("textbook")

def upsert_to_db(chunks):
    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = embed_model.embed_query(chunk)

        vectors.append({
            "id": f"text-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk
            }
        })

    print(f"{len(vectors)} vectors prepared. Upserting...")

    batch_size = 96

    for i in range(0, len(vectors), batch_size):
        dense_index.upsert(vectors=vectors[i:i+batch_size])
        print(f"Upserted batch {i // batch_size + 1}")

    print("All batches upserted successfully.")

        
# upsert_to_db(create_text_chunks(text))

def do_query(query):
    query_embedding  = embed_model.embed_query(query)

    results = dense_index.query(vector=query_embedding, top_k=10, include_metadata=True)

    return results

# print("Response: ",do_query("What is the name of this chapter?"))

def generate_response(query):
    results = do_query(query)

    print("Query vectors found...")

    if not results.get("matches"):
        print("No matches found.")
        return "No relevant data found"
    
    context = "\n\n".join([r["metadata"]["text"] for r in results["matches"]])

    # print("Context done...", context)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])

    return response


print("Generated response: ", generate_response("What is the name of this chapter?"))