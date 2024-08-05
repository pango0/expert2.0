import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nextfire/paraphrase-multilingual-minilm")
    return embeddings

def get_text_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter

def clear_database(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def load_documents(path):
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

def add_to_chroma(chunks, db_path):
    db = Chroma(
        persist_directory=db_path, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in {db_path}: {len(existing_ids)}")
    
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_file_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        current_file_id = source
        if current_file_id == last_file_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_file_id}:{current_chunk_index}"
        last_file_id = current_file_id
        chunk.metadata["id"] = chunk_id
    return chunks

def create_database(data_path, data_db_path, chunk_size, chunk_overlap, flag=""):
    if flag == "--reset":
        print("Clearing database...")
        clear_database(data_db_path)
    documents = load_documents(data_path)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    add_to_chroma(chunks, data_db_path)

if __name__ == "__main__":
    create_database("summaries", "summary_db", 500, 150)