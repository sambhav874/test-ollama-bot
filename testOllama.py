import os
import csv
import ollama
import chainlit as cl
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import requests
from langchain.schema import Document

# Constants
DATA_PATH = "pdfs/"
DB_FAISS_PATH = "vectorstores/db_faiss"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL = "llama3.1:8b"
API_URL = "https://srv618269.hstgr.cloud/api/insurances"

custom_prompt_template = """
You are an insurance agent of Wing Heights Ghana - An insurance provider.
Use the following pieces of information to answer the user's question.
Answer the question only if it is present in the given piece of information.
If you don't know the answer or the question is not related to the provided information, say: "I am an insurance agent and I can only provide insurance solutions offered by our company. Would you like to book an appointment to discuss your insurance needs?"

If the user wants to book an appointment, ask for the following details one by one:
1. Name
2. Contact Number
3. Email
4. Appointment Date
5. Insurance Type

After collecting all details, provide a summary of the appointment details.

If the user doesn't want to book an appointment, end the conversation politely.

For basic greetings, respond with short, friendly statements.

Context: {context}
Question: {question}

Helpful answer:
"""

def fetch_api_data():
    try:
        response = requests.get(API_URL)
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return None


def create_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH} directory. Please add your PDF files to this directory and run the script again.")
        return None

    if not os.listdir(DATA_PATH):
        print(f"No PDF files found in {DATA_PATH}. Please add your PDF files and run the script again.")
        return None

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Fetch API data
    api_data = fetch_api_data()
    if api_data:
        # Convert API data to Document objects
        api_documents = [Document(page_content=str(item), metadata={"source": "API"}) for item in api_data]
        documents.extend(api_documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"})
    
    db = FAISS.from_documents(texts, embeddings)
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved to {DB_FAISS_PATH}")
    return db


def load_vector_db():
    if not os.path.exists(DB_FAISS_PATH):
        print("Vector store not found. Creating new vector store...")
        return create_vector_db()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"})
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    desired_model = 'llama3.1:8b'

    initialization_prompt = {
        "role": "system",
        "content": custom_prompt_template
    }

    def ollama_chat(query, context=""):
        response = ollama.chat(
            model=desired_model, 
            messages=[
                {"role": "system", "content": initialization_prompt['content']},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        if response and 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "I apologize, but I couldn't process your request. How else can I assist you with our insurance services?"

    return ollama_chat

def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 2})

    def qa_chain(query):
        # Retrieve relevant documents from the vector store
        print(query)
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Call the LLM with the combined context and query
        response = llm(query, context)
        return response

    return qa_chain

def qa_bot():
    db = load_vector_db()
    if db is None:
        return None
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def count_tokens(text):
    # This is a simple approximation. For accurate counting, you'd need to use the actual tokenizer.
    return len(text.split())

@cl.on_chat_start
async def start():
    chain = qa_bot()
    if chain is None:
        await cl.Message(content="Failed to initialize the chatbot. Please make sure PDF files are present in the 'pdfs' directory.").send()
        return
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the insurance chatbot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)
    cl.user_session.set("token_count", 0)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    token_count = cl.user_session.get("token_count")
    appointment_details = cl.user_session.get("appointment_details", {})
    booking_confirmed = cl.user_session.get("booking_confirmed", False)
    awaiting_confirmation = cl.user_session.get("awaiting_confirmation", False)

    ques_tok = count_tokens(message.content)
    token_count += ques_tok

    if token_count <= 200:
        try:
            if appointment_details and booking_confirmed:
                # We're in the process of booking an appointment
                appointment_fields = ["Name", "Contact Number", "Email", "Appointment Date", "Insurance Type"]
                current_field = appointment_fields[appointment_details["step"]]
                
                appointment_details[current_field] = message.content
                appointment_details["step"] += 1

                if appointment_details["step"] < len(appointment_fields):
                    next_field = appointment_fields[appointment_details["step"]]
                    answer = f"Thank you. Now, please provide your {next_field}:"
                else:
                    summary = "Appointment Details:\n"
                    for field in appointment_fields:
                        summary += f"{field}: {appointment_details.get(field, 'Not provided')}\n"
                    answer = f"Thank you for providing all the details. Here's a summary of your appointment:\n\n{summary}\nWe look forward to assisting you!"
                    
                    
                    with open('appointment_details.csv', mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([appointment_details.get(field, 'Not provided') for field in appointment_fields])
                    
                    
                    appointment_details = {}
                    booking_confirmed = False
                    cl.user_session.set("appointment_details", {})
                    cl.user_session.set("booking_confirmed", False)
            elif awaiting_confirmation:
                if message.content.lower() == 'yes':
                    booking_confirmed = True
                    appointment_details = {"step": 0}
                    answer = "Great! Let's book an appointment. Please provide your Name:"
                    cl.user_session.set("appointment_details", appointment_details)
                    cl.user_session.set("booking_confirmed", booking_confirmed)
                elif message.content.lower() == 'no':
                    answer = "No problem. How else can I assist you with our insurance services?"
                    cl.user_session.set("appointment_details", {})
                    cl.user_session.set("booking_confirmed", False)
                else:
                    answer = "I'm sorry, I didn't understand your response. Please answer with 'Yes' or 'No'. Would you like to book an appointment?"
                cl.user_session.set("awaiting_confirmation", message.content.lower() not in ['yes', 'no'])
            else:
                # Normal conversation flow
                answer = chain(message.content)
                
                if "book an appointment" in answer.lower():
                    answer += "\n\nWould you like to book an appointment? Please respond with 'Yes' or 'No'."
                    cl.user_session.set("awaiting_confirmation", True)
            
        except Exception as e:
            await cl.Message(content=f"Error occurred: {e}").send()
            return

        ans_tok = count_tokens(answer)
        token_count += ans_tok

        await cl.Message(content=f"\nQuestion tokens: {ques_tok}\nAnswer tokens: {ans_tok}").send()
        await cl.Message(content=answer).send()

        # Log conversation data
        data = [
            [message.content, ques_tok, answer, ans_tok],
        ]
        with open('chatbot_data.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        cl.user_session.set("token_count", token_count)
    else:
        await cl.Message(content="You have expired your tokens. Please consult with customer service through mail or the provided customer assistance number.").send()

if __name__ == "__main__":
    if not os.path.exists(DB_FAISS_PATH):
        print("Vector store not found. Creating new vector store...")
        db = create_vector_db()
        if db is None:
            print("Failed to create vector store. Exiting.")
            exit(1)
    app = cl.arun()
