import os, sys
from pathlib import Path
import json
import tiktoken
import openai
import numpy as np
import pickle
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt
from loguru import logger

# Load environment variables
load_dotenv("embedding.env")

# Configure Azure OpenAI Service API
openai.api_key=os.getenv('EMBEDDING_API_KEY')
openai.api_base=os.getenv('EMBEDDING__API_BASE')
openai.api_type = os.getenv('EMBEDDING_TYPE')
openai.api_version = os.getenv('EMBEDDING_VERSION')
EMBEDDING_DEPLOYMENT_NAME=os.getenv('EMBEDDING_DEPLOYMENT_NAME')
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CHUNK_SIZE = 1000

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(deployment=EMBEDDING_DEPLOYMENT_NAME)

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
logger.debug(openai.api_type, openai.api_version, openai.api_key, openai.api_base)


test_text = "This is a test"
query_result = embeddings.embed_query(test_text)
doc_result = embeddings.embed_documents([test_text])
logger.info(json.dumps(query_result))
logger.info(json.dumps(doc_result))