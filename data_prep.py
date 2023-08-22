import os
from pathlib import Path
import json
import tiktoken
import openai
import numpy as np
import pickle
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load environment variables
load_dotenv("embedding.env")

# Configure Azure OpenAI Service API
openai.api_type = os.getenv('EMBEDDING_TYPE')
openai.api_version = os.getenv('EMBEDDING_VERSION')
# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CHUNK_SIZE = 1000
openai.api_key=os.getenv('EMBEDDING_KEY')
openai.api_base=os.getenv('EMBEDDING_BASE')
EMBEDDING_DEPLOYMENT_NAME=os.getenv('EMBEDDING_DEPLOYMENT_NAME')
# initialize tiktoken for encoding text
encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)

print(openai.api_type, openai.api_version, openai.api_key, openai.api_base)

# list all files in the data
cwd = Path.cwd()
data_dir = cwd / "data"
files = os.listdir(data_dir)
print(files)
# read content from each file and append it to documents

SIZE = 20000 ################## ONLY THE FIRST SIZE CHARACTERS
documents = []
for file in files:
    with open(os.path.join(data_dir, file), "r") as f:
        # read the content from the txt file
        content = (f.read())[:SIZE]
        documents.append({
            "filename": file,
            "content": content,
        })

# print some stats about the documents
print(f"Loaded {len(documents)} documents")
for doc in documents:
    num_tokens = len(encoding.encode(doc['content']))
    print(f"Filename: {doc['filename']} Content: {doc['content'][:80]}... \n---> Tokens: {num_tokens}\n")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text):
    # remove newlines and double spaces
    text = text.replace("\n", " ").replace("  ", " ")
    return openai.Embedding.create(input=text, engine=EMBEDDING_DEPLOYMENT_NAME)["data"][0]["embedding"]

# Create embeddings for all docs
for doc in documents:
    doc['embedding'] = get_embedding(doc['content'])
    print(f"Created embedding for {doc['filename']}")

# Save documents to disk
with open(os.path.join(data_dir, "documents.pkl"), "wb") as f:
    pickle.dump(documents, f)
