
# utils.py
from config import *
import csv
import json
import requests
from datetime import datetime
from langchain.prompts import PromptTemplate

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from textblob import TextBlob
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from nltk.sentiment import SentimentIntensityAnalyzer





model = SentenceTransformer('all-MiniLM-L6-v2')

# sia = SentimentIntensityAnalyzer()

def calculate_bleu(reference, response):
    # Smoothing function to handle cases with low overlap
    smooth = SmoothingFunction().method1
    
    # Tokenizing the reference and response
    reference_tokens = [reference.split()]
    response_tokens = response.split()
    
    # Calculate BLEU score using corpus_bleu
    bleu_score = corpus_bleu([reference_tokens], [response_tokens], smoothing_function=smooth)
    
    return bleu_score


def calculate_rouge(reference, response):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, response)
    return scores['rougeL'].fmeasure

def calculate_cosine_similarity(reference, response):
    ref_embedding = model.encode([reference])
    res_embedding = model.encode([response])
    return cosine_similarity(ref_embedding, res_embedding)[0][0]

def calculate_meteor(reference, response):
    # Ensure both reference and response are tokenized (i.e., as lists of words)
    reference_tokens = reference.split()  # Tokenize the reference
    response_tokens = response.split()    # Tokenize the response
    return meteor_score([reference_tokens], response_tokens)

# Calculate F1 score
def calculate_f1(reference, response):
    # Tokenize to get word lists
    reference_tokens = set(reference.split())
    response_tokens = set(response.split())
    # Precision and Recall
    common_tokens = reference_tokens.intersection(response_tokens)
    if len(response_tokens) == 0 or len(reference_tokens) == 0:
        return 0  # Avoid division by zero
    precision = len(common_tokens) / len(response_tokens)
    recall = len(common_tokens) / len(reference_tokens)
    # F1 Calculation
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Calculate sentiment score
# def calculate_sentiment(response):
#     sentiment = sia.polarity_scores(response)
#     return sentiment['compound']  # Compound score for overall polarity

# Log interaction data with scores
def save_interaction(timestamp, query, ques_tok, validated_answer, ans_tok, total_tokens, 
                     bleu_score, rouge_score, cosine_score, meteor_score, f1_score):
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "response": validated_answer,
        "question_tokens": ques_tok,
        "answer_tokens": ans_tok,
        "total_tokens": total_tokens,
        "bleu_score": bleu_score,
        "rouge_score": rouge_score,
        "cosine_similarity": cosine_score,
        "meteor_score": meteor_score,
        "f1_score": f1_score,
    }

    # Append the entry to the CSV file
    with open("chat_log.csv", mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())
        if file.tell() == 0:  # Write header if file is empty
            writer.writeheader()
        writer.writerow(log_entry)

# Token counting function
def count_tokens(text):
    return len(text.split())

# Load the custom prompt template
def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['context', 'question'])

# Check if the query is a greeting
def is_greeting(query):
    return any(greet in query.lower().strip().split() for greet in GREETINGS)

# Check if the query is insurance-related
def is_insurance_related(query):
    return any(keyword in query.lower().strip().split() for keyword in INSURANCE_KEYWORDS)

# Check if the query indicates booking intent
def is_booking_intent(query):
    return any(keyword in query.lower().split() for keyword in BOOKING_KEYWORDS)

# Book an appointment and save details
def appointment_booking():
    print("ADA: Let's book an appointment for you. I will need some details.")
    user_name = input(USER_NAME_STR)
    contact_info = input(CONTACT_INFO_STR)
    preferred_date = input(APPOINTMENT_DATE_STR)
    preferred_time = input(PREFERRED_TIME_STR)
    appointment_datetime = f"{preferred_date} {preferred_time}"
    
    # Save the appointment information to CSV
    with open(APPOINTMENTS_CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_name, contact_info, appointment_datetime])
    
    print("ADA: Thank you! Your appointment has been booked.")
    return "Your appointment has been successfully scheduled."

# Validate response against the context
def validate_response(response, context):
    return response if any(word in response for word in context.split()) else "I don't know the answer."

# Send a prompt to the Ollama API and get response
def get_ollama_response(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3",
        "prompt": prompt,
        "temperature": 0.2,
        "top_k": 10,
        "top_p": 0.9
    }
    response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
    responses = response.text.strip().split('\n')
    return ''.join([json.loads(res)["response"] for res in responses if json.loads(res).get("response")]).strip()

def get_relevant_context(query, db):
    """
    Retrieve relevant context from the FAISS database based on the query.
    """
    retriever = db.as_retriever(search_kwargs={'k': SEARCH_DOCS})  # Adjust 'k' as needed
    search_results = retriever.get_relevant_documents(query)
    # Combine the content of search results into a single string as context
    context = " ".join([doc.page_content for doc in search_results])
    return context

def generate_and_validate_response(query, db, prompt_template):
    """
    Generate and validate response for a given query.
    """
    # db = FAISS.load_local("/root/rag_chatbot/vectorstores/db_faiss", embeddings, allow_dangerous_deserialization=True)
    # Retrieve relevant context
    context = get_relevant_context(query, db)
    
    # Format prompt with the retrieved context
    prompt = prompt_template.format(context=context, question=query)
    
    # Generate the response
    answer = get_ollama_response(prompt)
    
    # Validate response
    validated_answer = validate_response(answer, context)
    
    return validated_answer


































































# # utils.py
# from config import *
# import csv
# import json
# import requests
# from datetime import datetime
# from langchain.prompts import PromptTemplate

# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import numpy as np

# model = SentenceTransformer('all-MiniLM-L6-v2')

# def calculate_bleu(reference, response):
#     reference_tokens = [reference.split()]
#     response_tokens = response.split()
#     return sentence_bleu(reference_tokens, response_tokens)

# def calculate_rouge(reference, response):
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = scorer.score(reference, response)
#     return scores['rougeL'].fmeasure

# def calculate_cosine_similarity(reference, response):
#     ref_embedding = model.encode([reference])
#     res_embedding = model.encode([response])
#     return cosine_similarity(ref_embedding, res_embedding)[0][0]

# import csv

# # Log interaction data with scores
# def save_interaction(timestamp, query, ques_tok, validated_answer, ans_tok, total_tokens, bleu_score, rouge_score, cosine_score):
#     log_entry = {
#         "timestamp": timestamp,
#         "query": query,
#         "response": validated_answer,
#         "question_tokens": ques_tok,
#         "answer_tokens": ans_tok,
#         "total_tokens": total_tokens,
#         "bleu_score": bleu_score,
#         "rouge_score": rouge_score,
#         "cosine_similarity": cosine_score
#     }

#     # Append the entry to the CSV file
#     with open("chat_log.csv", mode="a", newline="") as file:
#         writer = csv.DictWriter(file, fieldnames=log_entry.keys())
#         if file.tell() == 0:  # Write header if file is empty
#             writer.writeheader()
#         writer.writerow(log_entry)

# # Token counting function
# def count_tokens(text):
#     return len(text.split())

# # Load the custom prompt template
# def set_custom_prompt():
#     return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['context', 'question'])

# # Check if the query is a greeting
# def is_greeting(query):
#     return any(greet in query.lower().strip().split() for greet in GREETINGS)

# # Check if the query is insurance-related
# def is_insurance_related(query):
#     return any(keyword in query.lower().strip().split() for keyword in INSURANCE_KEYWORDS)

# # Check if the query indicates booking intent
# def is_booking_intent(query):
#     return any(keyword in query.lower().split() for keyword in BOOKING_KEYWORDS)

# # Book an appointment and save details
# def appointment_booking():
#     print("ADA: Let's book an appointment for you. I will need some details.")
#     user_name = input(USER_NAME_STR)
#     contact_info = input(CONTACT_INFO_STR)
#     preferred_date = input(APPOINTMENT_DATE_STR)
#     preferred_time = input(PREFERRED_TIME_STR)
#     appointment_datetime = f"{preferred_date} {preferred_time}"
    
#     # Save the appointment information to CSV
#     with open(APPOINTMENTS_CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow([user_name, contact_info, appointment_datetime])
    
#     print("ADA: Thank you! Your appointment has been booked.")
#     return "Your appointment has been successfully scheduled."

# # Validate response against the context
# def validate_response(response, context):
#     return response if any(word in response for word in context.split()) else "I don't know the answer."



# # Send a prompt to the Ollama API and get response
# def get_ollama_response(prompt):
#     headers = {"Content-Type": "application/json"}
#     data = {
#         "model": "llama3",
#         "prompt": prompt,
#         "temperature": 0.2,
#         "top_k": 10,
#         "top_p": 0.9
#     }
#     response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
#     responses = response.text.strip().split('\n')
#     return ''.join([json.loads(res)["response"] for res in responses if json.loads(res).get("response")]).strip()


# def get_relevant_context(query, db):
#     """
#     Retrieve relevant context from the FAISS database based on the query.
#     """
#     retriever = db.as_retriever(search_kwargs={'k': SEARCH_DOCS})  # Adjust 'k' as needed
#     search_results = retriever.get_relevant_documents(query)
#     # Combine the content of search results into a single string as context
#     context = " ".join([doc.page_content for doc in search_results])
#     return context

# def generate_and_validate_response(query, db, prompt_template):
#     """
#     Generate and validate response for a given query.
#     """
#     db=FAISS.load_local("/root/rag_chatbot/vectorstores/db_faiss", embeddings, allow_dangerous_deserialization=True)
#     # Retrieve relevant context
#     context = get_relevant_context(query, db)
    
#     # Format prompt with the retrieved context
#     prompt = prompt_template.format(context=context, question=query)
    
#     # Generate the response
#     answer = get_ollama_response(prompt)
    
#     # Validate response
#     validated_answer = validate_response(answer, context)
    
#     return validated_answer
