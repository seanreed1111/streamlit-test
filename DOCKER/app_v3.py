import os, sys
from pathlib import Path
import openai
import streamlit as st
import tiktoken
import numpy as np
import pickle
from openai.embeddings_utils import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import AzureOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from loguru import logger
# Load environment variables
load_dotenv('deployment.env')

# model variables
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = 0

# embedding variables
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
EMBEDDING_API_TYPE = os.getenv('EMBEDDING_TYPE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_VERSION')

embeddings = OpenAIEmbeddings(
                openai_api_key=EMBEDDING_API_KEY,
                openai_api_base=EMBEDDING_API_BASE,
                openai_api_type=EMBEDDING_API_TYPE,
                openai_api_version = EMBEDDING_API_VERSION,
                deployment=EMBEDDING_DEPLOYMENT_NAME,

                show_progress_bar=True
)

llm = AzureChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE,
                openai_api_type=OPENAI_API_TYPE,
                openai_api_version=OPENAI_API_VERSION,
                model_name=MODEL_NAME,
                deployment=DEPLOYMENT_NAME,
                temperature=TEMPERATURE,
                request_timeout=20
)

logger.add(sys.stderr, format="{time} {level} {message}", level="DEBUG")
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", level="DEBUG")

cwd = Path.cwd()
data_path = cwd / "data"
data_path.mkdir(exist_ok=True)
pickle_file_path = data_path / "documents.pkl"

if pickle_file_path.is_file:
    with open(pickle_file_path, "rb") as f:
            documents = pickle.load(f)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, embeddings=embeddings):
    return embeddings.embed_query(text)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def run_prompt(prompt, llm):
    response = ""
    return response['choices'][0]['text']


