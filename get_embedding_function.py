from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from dotenv import load_dotenv

load_dotenv() 
os.environ["OPENAI_API_KEY"] 
def get_embedding_function ():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OpenAIEmbeddings()
    return embeddings