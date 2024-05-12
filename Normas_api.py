from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os
import shutil
import pdfplumber
from dotenv import load_dotenv

load_dotenv() 
os.environ["OPENAI_API_KEY"] 
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    
    print("Splitting documents into chunks...")
    chunks = split_text(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    print("Saving chunks to Chroma...")
    save_to_chroma(chunks)
    print("Chunks saved to Chroma.")

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(DATA_PATH, filename)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    # Aquí asumimos que el nombre del archivo es la fuente.
                    # El número de página se incrementa para cada página del documento.
                    documents.append(Document(text, metadata={"Fuente": filename, "pagina": i+1}))
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Print debug information for one document
    document = chunks[10]
    print("Sample document content:", document.page_content)
    print("Sample document metadata:", document.metadata)

    return chunks

def calculate_chunk_ids(chunks: list[Document]):
    print("Calculando IDs de los fragmentos...")
    last_page_id = None  
    current_chunk_index = 0  

    for chunk in chunks:  
        source = chunk.metadata.get("Fuente")  
        page = chunk.metadata.get("pagina")  

        current_page_id = f"{source}:{page}"  

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id  

        chunk.metadata["id"] = chunk_id

    print("IDs de los fragmentos calculados.")
    return chunks  

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks_with_ids, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks_with_ids)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
