import argparse
import os
import sys
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma



def main():
    # Crea un analizador de argumentos
    parser = argparse.ArgumentParser()
    # Añade el argumento --reset al analizador
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    # Analiza los argumentos de la línea de comandos
    args = parser.parse_args()
    # Si se proporcionó la bandera --reset, borra la base de datos
    if args.reset:
        confirm = input("¿Estás seguro de que quieres borrar la base de datos? (s/n): ")
        if confirm.lower() == 's':
            print("✨ Eliminando Database")
            clear_database()
        else:
            print("Operación cancelada, la base de datos no se borrará.")
            sys.exit(0)

    # Carga los documentos
    documents = load_documents()
    # Divide los documentos en fragmentos
    chunks = split_documents(documents)
    # Añade los fragmentos a Chroma
    add_to_chroma(chunks)

def load_documents():
    document_loader=PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# divide  los documentos en fragmentos o “chunks” más pequeños y los almacena en la variable chunks^^

def split_documents(documents: list[Document]):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,

    )
    return text_splitter.split_documents(documents)

##################################################
################--- ---------#####################

def add_to_chroma(chunks: list[Document]):
    # Crea una instancia de Chroma que representa la base de datos
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calcula los ID de los documentos
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Obtiene los documentos existentes
    existing_items = db.get(include=[])  # Los ID siempre se incluyen por defecto
    existing_ids = set(existing_items["ids"])
    print(f"Numero de Documentos existentes en la Base de Datos: {len(existing_ids)}")

    




def calculate_chunk_ids(chunks):
    # Esta función calcula y asigna un ID único a cada "chunk" (fragmento de documento) en la lista de chunks.

    # Esto creará IDs como "data/Norma_Api.pdf:1:2"    Fuente, NumeroDePagina, ChunkNumber
    # Fuente de la Página : Número de Página : Índice del Chunk

    last_page_id = None  # Almacena el ID de la última página procesada
    current_chunk_index = 0  # Almacena el índice del chunk actual en la página actual

    for chunk in chunks:  # Itera sobre cada chunk en la lista de chunks
        source = chunk.metadata.get("Fuente")  # Obtiene la fuente del chunk (por ejemplo, el nombre del archivo)
        page = chunk.metadata.get("pagina")  # Obtiene el número de página del chunk

        current_page_id = f"{source}:{page}"  # Crea el ID de la página actual

        # Si el ID de la página actual es el mismo que el del último, incrementa el índice.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            # Si el ID de la página actual es diferente al del último, reinicia el índice del chunk.
            current_chunk_index = 0

        # Calcula el ID del chunk.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id  # Actualiza el último ID de página

        # Añade el ID del chunk a los metadatos del chunk.
        chunk.metadata["id"] = chunk_id

    return chunks  # Devuelve la lista de chunks con sus IDs asignados



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()