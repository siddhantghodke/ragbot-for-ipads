import streamlit as st
import os
from dotenv import load_dotenv
from main import ask_query, load_retriever

# Load environment variables early
load_dotenv()

st.title("iPad Chatbot")

# Minimal chat-like styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache the retriever to avoid reloading each time
@st.cache_resource
def get_cached_retriever():
    return load_retriever()

retriever = get_cached_retriever()

# Get API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")

# chat history list
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# Render chat messages
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Chat input at bottom
prompt = st.chat_input("Ask about iPads…")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        for chunk in ask_query(prompt, retriever, st.session_state.chat_history):
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.chat_history.append((prompt, full_response))

