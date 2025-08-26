import os
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv() 

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY environment variable not found! Please set it in your environment.")

def load_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key
                    )
    vector_store = FAISS.load_local("vectorstore.db", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def ask_query(query, retriever, chat_history=[]):
    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash", 
                        temperature=0.3, 
                        max_tokens=1000,
                        google_api_key=api_key,
                        streaming=True
                    )

    # Get relevant documents
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format chat history
    chat_history_str = str(chat_history[-3:]) if chat_history else "No previous conversation"
    
    system_prompt = """You are an expert on Apple iPads with comprehensive knowledge from Wikipedia and structured specifications.
Use the following pieces of retrieved context to answer the question accurately.
If you don't know the answer based on the provided context, say that you don't know.
Provide detailed, accurate information about iPad models, specifications, release dates, and features.
Use clear, concise language and cite specific information when possible.
Do not provide negative statements after a concise answer like doesn't have, doesn't support, etc.
Provide source in the form of link at the end.
if the question is about ipad pro provide link to wikipedias ipad pro page 
if the question is about ipad mini provide the link to ipad mini wikipedia at the end of the response 

When answering questions about specific iPad models, use the structured information provided:
- iPad Pro (M4): Latest model with M4 chip, Ultra Retina XDR display, price 99,990 INR
- iPad Air (M3): Latest model with M3 chip, Liquid Retina display, price 79,990 INR
- iPad (11th gen): Latest model with A16 Bionic chip, Liquid Retina display, price 69,990 INR
- iPad mini (A17 Pro): Latest model with A17 Pro chip, compact 8.3-inch design, price 59,990 INR

Always mention the latest models and their key specifications when relevant.

Also access the chat history to provide contextually relevant and coherent responses. If the user is asking follow-up questions, reference previous parts of the conversation appropriately."""

    # Create messages for the chat model using proper message objects
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context: {context}\n\nChat History: {chat_history_str}\n\nQuestion: {query}")
    ]

    # Stream the response directly from the LLM
    return llm.stream(messages)
