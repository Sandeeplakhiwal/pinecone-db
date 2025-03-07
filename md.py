from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def get_texts():
    # Load your text file
    loader = TextLoader("data/test.txt")
    data = loader.load()

    # Extract the full text from documents
    full_text = "\n\n".join([doc.page_content for doc in data])

    # Setup MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Heading1")
    ]

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Split the text into chunks keeping the markdown structure
    text_chunks = text_splitter.split_text(full_text)

    final_chunks = []

    character_splitter = RecursiveCharacterTextSplitter(chunk_size=3000,
    chunk_overlap=200)

    for chunk in text_chunks:
        smaller_chunks = character_splitter.split_text(chunk.page_content)
        final_chunks.extend(smaller_chunks)

    for i, chunk in enumerate(final_chunks):
        print(f"Chunk {i} size: {len(chunk.encode('utf-8'))} bytes")

    # For Pinecone upserts
    return final_chunks

