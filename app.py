import os
from pathlib import Path
import openai
import streamlit as st
import tiktoken
import numpy as np
import pickle
from openai.embeddings_utils import cosine_similarity
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load environment variables
load_dotenv('deployment.env')

# model variables
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
# embedding variables
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")


cwd = Path.cwd()
pickle_file_path = cwd /"data" / "documents.pkl"
print(cwd, pickle_file_path)
with open(pickle_file_path, "rb") as f:
        documents = pickle.load(f)

print("loaded documents!")
print(openai.api_type, openai.api_version, openai.api_base, openai.api_key, DEPLOYMENT_NAME)
print(EMBEDDING_DEPLOYMENT_NAME, EMBEDDING_API_BASE, EMBEDDING_API_KEY)
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, key=EMBEDDING_API_KEY, base=EMBEDDING_API_BASE, engine=EMBEDDING_DEPLOYMENT_NAME):
    # remove newlines and double spaces
    text = text.replace("\n", " ").replace("  ", " ")
    openai.api_key, openai.api_base = key, base
    return openai.Embedding.create(input=text, engine=engine)["data"][0]["embedding"]

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def run_prompt(prompt, max_tokens=1000, key=openai.api_key,base=openai.api_base, engine=DEPLOYMENT_NAME):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens
    )
    return response['choices'][0]['text']

# configure UI elements with Streamlit

st.title('Demo app')
question = st.text_input('Question')
answer_button = st.button('Generate answer')

if answer_button:
    # first extract the actual search query from the question
    question_prompt = f"""You extract search queries from prompts and remove all styling options or other things (e.g., the formatting the user asks for). You do not answer the question.
Prompt: {question}\n
Query:"""
    search_query = run_prompt(question_prompt, max_tokens=1000)
    # then get the embedding and compare it to all documents
    qe = get_embedding(search_query)
    similarities = [cosine_similarity(qe, doc['embedding']) for doc in documents]
    max_i = np.argmax(similarities)

    st.write(f"**Searching for:** {search_query}\n\n**Found answer in:** {documents[max_i]['filename']}")

    # finally generate the answer
    prompt = f"""
    Content:
    {documents[max_i]['content']}
    Please answer the question below using only the content from above. If you don't know the answer or can't find it, say "I couldn't find the answer".
    Question: {question}
    Answer:"""
    answer = run_prompt(prompt)

    st.write(f"**Answer**:\n\n{answer}")
