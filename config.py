# config.py
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file

load_dotenv("C:/Users/vikas/OneDrive/Desktop/Internship/newLearning/llama_tut/rag_chatbot/config.env")

# load_dotenv("/root/rag_chatbot/config.env")

# Paths
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH")
CHATBOT_DATA_PATH = os.getenv("CHATBOT_DATA_PATH")
APPOINTMENTS_CSV_PATH = os.getenv("APPOINTMENTS_CSV_PATH")
LOG_FILE_PATH = "chat_log.csv"

# Token and model settings
MAX_TOKENS_PER_RESPONSE = int(os.getenv("MAX_TOKENS_PER_RESPONSE"))
MAX_SESSION_TOKENS = int(os.getenv("MAX_SESSION_TOKENS"))
SEARCH_DOCS = int(os.getenv("SEARCH_DOCS"))

# Ollama API settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

# Common keywords for intent recognition (split the comma-separated strings into lists)
INSURANCE_KEYWORDS = os.getenv("INSURANCE_KEYWORDS").split(",")
BOOKING_KEYWORDS = os.getenv("BOOKING_KEYWORDS").split(",")
GREETINGS = os.getenv("GREETINGS").split(",")

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE="""
You are ADA, an assistant helping people in the insurance sector. You must strictly follow these instructions:
1. Only use the provided context to answer the user's questions.
2. If the answer to the question is not in the context provided, reply: "I don't know the answer."
3. Never provide answers based on information outside of the context.
4. Greet user responses.
5. Hold conversation as if you are an insurance agent.
6. Keep responses direct and avoid extra phrases.

Context: {context}
Question: {question}
Strictly relevant answer:
"""

# Statement constants
USER_NAME_STR = os.getenv("USER_NAME_STR")
CONTACT_INFO_STR = os.getenv("CONTACT_INFO_STR")
APPOINTMENT_DATE_STR = os.getenv("APPOINTMENT_DATE_STR")
PREFERRED_TIME_STR = os.getenv("PREFERRED_TIME_STR")

# Load embeddings and FAISS database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings)

time_minutes=5

