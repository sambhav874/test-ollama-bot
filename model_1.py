import csv
import requests
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
import json
from datetime import datetime

# Define paths
DB_FAISS_PATH = "vectorstores/db_faiss"

# Token counting function
def count_tokens(text):
    return len(text.split())

# Custom prompt template with stricter instructions
custom_prompt_template = """
You are ADA, an assistant helping people in the insurance sector. You must strictly follow these instructions:
1. Only use the provided context to answer the user's questions.
2. If the answer to the question is not in the context provided, reply: "I don't know the answer."
3. Never provide answers based on information outside of the context.
4. Greet user responses.
5. Hold conversation as if you are an insurance agent.
6. Keep responses direct and avoid extra phrases.

Context: {context}
Question: {question}

#   adding new prompt to direct chatbot for using context, question and rules
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

MAX_TOKENS_PER_RESPONSE = 100

# Custom LLM class to interface with Ollama API
class OllamaLLM(LLM):
    def _call(self, prompt: str, stop=None):
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3",
            "prompt": prompt,
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.9
        }
        
        response = requests.post(url, headers=headers, json=data)
        try:
            responses = response.text.strip().split('\n')
            collected_responses = []
            for response_str in responses:
                parsed_response = json.loads(response_str)
                if parsed_response.get("response"):
                    collected_responses.append(parsed_response["response"])
            final_output = ''.join(collected_responses).strip()
            
            # Cap response to the token limit
            final_token_count = count_tokens(final_output)
            if final_token_count > MAX_TOKENS_PER_RESPONSE:
                truncated_response = ' '.join(final_output.split()[:MAX_TOKENS_PER_RESPONSE])
                return truncated_response.strip()
            else:
                return final_output

        except ValueError as ve:
            print(f"JSON decoding error: {ve}")
            raise Exception(f"Ollama API call failed with invalid JSON. Response: {response.text}")

    @property
    def _identifying_params(self):
        return {"name_of_model": "ollama_custom"}

    @property
    def _llm_type(self):
        return "ollama_custom"

# Define a list of common greetings
greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]

# Function to handle greeting responses
def is_greeting(query):
    query_arr = query.lower().strip().split(" ")
    return any(el in query_arr for el in greetings)

# Validate response against the context
def validate_response(response, context):
    if any(word in response for word in context.split()):
        return response.strip()
    return "I don't know the answer."

# Intent classification to filter non-insurance-related questions
def is_insurance_related(query):
    insurance_keywords = ["insurance", "policy", "claim", "premium", "coverage", "deductible", "beneficiary", "health", "life", "vehicle"]
    return any(keyword in query.lower() for keyword in insurance_keywords)

# Check if user intent is appointment booking
def is_booking_intent(query):
    booking_keywords = ["appointment", "book", "schedule", "meeting"]
    query_arr = query.lower().split(" ")
    return any(el in query_arr for el in booking_keywords)

# Function to handle appointment booking
def appointment_booking():
    print("ADA: Let's book an appointment for you. I will need some details.")
    user_name = input("Please enter your full name: ")
    contact_info = input("Please provide your contact information (phone/email): ")
    preferred_date = input("Preferred date for the appointment (YYYY-MM-DD): ")
    preferred_time = input("Preferred time for the appointment (HH:MM): ")
    appointment_datetime = f"{preferred_date} {preferred_time}"
    
    # Save the appointment information to CSV
    with open('appointments.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_name, contact_info, appointment_datetime])
    
    print("ADA: Thank you! Your appointment has been booked.")
    return "Your appointment has been successfully scheduled."

# Setup the retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    
    llm = OllamaLLM()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    
    return qa

def final_result(query):
    if is_booking_intent(query):
        return appointment_booking()
    
    if not is_insurance_related(query):
        return "I'm here to help with insurance-related questions only."
    

    qa_chain = qa_bot()
    result = qa_chain({"query": query})
    
    if "result" in result:
        answer = result["result"]
    else:
        answer = "I don't know the answer based on the information I have."
    
    if "source_documents" in result:
        context = " ".join([doc.page_content for doc in result["source_documents"]])
    else:
        context = ""
    
    validated_answer = validate_response(answer, context)
    return validated_answer

# Main execution loop
if __name__ == "__main__":
    tokens_count = 0
    while tokens_count <= 2000:
        query = input("YOU: ")
        ques_tok = count_tokens(query)
        
        tokens_count += ques_tok
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if is_greeting(query):
            greet_response = "ADA: Hello! How can I assist you with insurance-related questions today?"
            print(f"{greet_response} [Timestamp: {timestamp}]")
            tokens_count += count_tokens(greet_response)
            continue  # Skip processing further if it's a greeting
        
        try:
            # Retrieve and validate the response based on context
            answer = final_result(query)
            ans_tok = count_tokens(answer)
            tokens_count += ans_tok
            
            print(f"ADA: {answer} [Timestamp: {timestamp}]")
            print(f"Tokens used: Question - {ques_tok}, Answer - {ans_tok}, Total - {tokens_count}")
            
            # Save the interaction to a CSV file with timestamp and token count
            data = [[timestamp, query, ques_tok, answer, ans_tok, tokens_count]]
            with open('chatbot_data.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(data)
        except Exception as e:
            print(f"Error occurred: {e}")
            break
    
    # Print final timestamp
    final_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"ADA: Thank you for chatting! The conversation ended at {final_timestamp}.")
    print("ADA: You have expired your tokens! Please speak to customer care!")

