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



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()