
# utils2.py
from config import *
import csv
import json
import requests
from datetime import datetime
from langchain.prompts import PromptTemplate
from config import (GREETINGS, INSURANCE_KEYWORDS, BOOKING_KEYWORDS, OLLAMA_API_URL, CUSTOM_PROMPT_TEMPLATE, 
                    APPOINTMENTS_CSV_PATH, CHATBOT_DATA_PATH)

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

# Save interaction to CSV file
def save_interaction(timestamp, query, ques_tok, answer, ans_tok, tokens_count):
    with open(CHATBOT_DATA_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, query, ques_tok, answer, ans_tok, tokens_count])

# Process each response safely
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
    
    # Process responses line by line
    responses = response.text.strip().split('\n')
    result = []
    
    for res in responses:
        try:
            # Deserialize the JSON and check for the "response" key
            parsed = json.loads(res)
            if "response" in parsed:
                result.append(parsed["response"])
        except json.JSONDecodeError:
            # Handle any deserialization errors
            print("Warning: Failed to deserialize a response segment.")
            continue
    
    # Return the combined and stripped result
    return ''.join(result).strip()





## Send a prompt to the Ollama API and get response
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
    if is_greeting(query):
            greet_response = "ADA: Hello! How can I assist you with insurance-related questions today?"
            print(f"{greet_response}")
            tokens_count += count_tokens(greet_response)
            return 

    # Handle appointment booking
    if is_booking_intent(query):
        appointment_booking()
        return 
    
    if not is_insurance_related(query):
            print("ADA: I'm here to help with insurance-related questions only.")
            return 
    # Retrieve relevant context
    context = get_relevant_context(query, db)
    
    # Format prompt with the retrieved context
    prompt = prompt_template.format(context=context, question=query)
    
    # Generate the response
    answer = get_ollama_response(prompt)
    
    # Validate response
    validated_answer = validate_response(answer, context)
    
    return validated_answer