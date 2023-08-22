import os
import openai
import streamlit as st
import tiktoken
import numpy as np
import pickle
from openai.embeddings_utils import cosine_similarity
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load environment variables
# load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

COMPLETION_MODEL = 'gpt-35-turbo'
EMBEDDING_MODEL = 'text-embedding-ada-002'

documents = pickle.load(open("documents.pkl", "rb"))

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text):
    # remove newlines and double spaces
    text = text.replace("\n", " ").replace("  ", " ")
    return openai.Embedding.create(input=text, engine=EMBEDDING_MODEL)["data"][0]["embedding"]

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def run_prompt(prompt, max_tokens=1000):
    response = openai.Completion.create(
        engine=COMPLETION_MODEL,
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
    search_query = run_prompt(question_prompt, max_tokens=100)
    
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