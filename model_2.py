# automated chatbot


import csv
import requests
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
import json

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

Context: {context}
Question: {question}
Strictly relevant answer:
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
    query_arr=query.lower().strip().split(" ")
    for el in query_arr:
        if el in greetings:
            return True
    return False


# Validate response against the context
def validate_response(response, context):
    if any(word in response for word in context.split()):
        return response
    else:
        return "I don't know the answer based on the provided information."

# Intent classification to filter non-insurance-related questions
def is_insurance_related(query):
    insurance_keywords = ["insurance", "policy", "claim", "premium", "coverage", "deductible", "beneficiary", "health", "life", "vehicle"]
    return any(keyword in query.lower() for keyword in insurance_keywords)

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
    initial_query = input("YOU: ")  
    query = initial_query  
    while tokens_count <= 1000:
        ques_tok = count_tokens(query)
        tokens_count += ques_tok
        
        if is_greeting(query):
            greet_response="ADA: Hello! How can I assist you with insurance-related questions today?"
            print(greet_response)
            tokens_count+=count_tokens(greet_response)
            continue  # Skip processing further if it's a greeting
        

        try:
            prompt = f"Question: {query}"
            answer = final_result(prompt)  
            ans_tok = count_tokens(answer)
            tokens_count += ans_tok
            
            print(f"ADA: {answer}")
            print(f"Tokens used: Question - {ques_tok}, Answer - {ans_tok}")
            
            # Save the conversation to CSV
            data = [[query, ques_tok, answer, ans_tok]]
            with open('automated_data.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(data)
            
            
            query = answer
            print("--------------------------------------------------------------------------------")

        except Exception as e:
            print(f"Error occurred: {e}")
            break
    
    print("ADA: You have expired your tokens! Please speak to customer care!")








