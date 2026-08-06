"""
Skincare Recommendation Chatbot — Streamlit app.

This mirrors the pipeline built in Skincare_ChatBot.ipynb:
  1. Load the pre-built embeddings (Ingredients_embedding.npy) and the
     cleaned ingredient dataframe (ingredient_data.pkl).
  2. Embed the user's query and retrieve the top matching ingredients.
  3. If nothing matches well enough, refuse ("not my area of expertise").
  4. Otherwise, pass the retrieved rows to Groq to generate a grounded answer.

Run with:
    uv run streamlit run app.py

Requires GROQ_API_KEY set in a .env file in the same folder.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---------------------------------------------------------------

load_dotenv()

EMBEDDINGS_PATH = "Ingredients_embedding.npy"
DATA_PATH = "ingredient_data.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.35
GENERATION_TEMPERATURE = 0.2

OFF_TOPIC_MESSAGE = (
    "This is not my area of expertise — I can help with skincare concerns "
    "and product recommendations."
)

SYSTEM_PROMPT = """You are a skincare recommendation assistant. You only answer
questions about skincare concerns, ingredients, and product recommendations,
using ONLY the data provided below. Do not use outside knowledge. Cite the
ingredient/product name for every claim. If the provided data doesn't clearly
answer the question, say you don't have enough information.
If the user's question is unrelated to skincare, respond exactly with:
"This is not my area of expertise — I can help with skincare concerns and product recommendations."
"""

# --- Cached resources (loaded once per session, not per message) ----------


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_index():
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(DATA_PATH):
        st.error(
            f"Could not find '{EMBEDDINGS_PATH}' and/or '{DATA_PATH}'. "
            "Run the notebook's embedding-generation cell first so these "
            "files exist in the same folder as app.py."
        )
        st.stop()
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_pickle(DATA_PATH)
    return embeddings, df


@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it to a .env file in this folder.")
        st.stop()
    return Groq(api_key=api_key)


# --- Core pipeline ----------------------------------------------------------


def retrieve(query, model, embeddings, df, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings)[0]

    top_idx = sims.argsort()[::-1][:top_k]
    max_sim = float(sims[top_idx[0]])

    if max_sim < threshold:
        return None, max_sim

    results = df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results, max_sim


def generate_response(client, query, retrieved_df):
    context = "\n".join(
        f"- {row['name']}: {row['short_description']} (benefits: {row['what_does_it_do']})"
        for _, row in retrieved_df.iterrows()
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User question: {query}\n\nRelevant data:\n{context}"},
    ]
    completion = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=GENERATION_TEMPERATURE,
    )
    return completion.choices[0].message.content


def chatbot_reply(query, model, embeddings, df, client):
    retrieved, max_sim = retrieve(query, model, embeddings, df)
    if retrieved is None:
        return OFF_TOPIC_MESSAGE, None
    reply = generate_response(client, query, retrieved)
    return reply, retrieved


# --- Streamlit UI -----------------------------------------------------------

st.set_page_config(page_title="Skincare Recommendation Chatbot", page_icon="🧴")
st.title("🧴 Skincare Recommendation Chatbot")
st.caption(
    "Ask about a skin concern and get ingredient recommendations grounded "
    "in the INCI ingredients dataset."
)

model = load_embedding_model()
embeddings, df = load_index()
client = load_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("e.g. What's good for oily, acne-prone skin?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, retrieved = chatbot_reply(prompt, model, embeddings, df, client)
        st.write(reply)

        if retrieved is not None:
            with st.expander("Show matched ingredients"):
                st.dataframe(
                    retrieved[["name", "short_description", "similarity"]].reset_index(drop=True)
                )

    st.session_state.messages.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot only answers questions it can ground in the ingredient "
        "dataset. Off-topic questions get a fixed refusal instead of a guess."
    )
    st.metric("Ingredients in database", len(df))
    st.caption(f"Similarity threshold: {SIMILARITY_THRESHOLD}")