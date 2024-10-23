# chatbot without automation




import csv
import requests
import sentencepiece as spm
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import json

# Define paths
DB_FAISS_PATH = "vectorstores/db_faiss"

def count_tokens(text):
    return len(text.split())  

custom_prompt_template = """
You are ADA, who is helping people in insurance sector, so perform conversation and ask questions accordingly
Use the following pieces of information to answer the user's question.
Answer the question only if it is present in the given piece of information.
If you don't know the answer, please just say that you don't know the answer.
But do answer basic greeting messages with a short statement.
Converse as if you are an insurance agent and ask questions accordingly.

Context: {context}
Question: {question}
Helpful answer:
"""

# Function to set the custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def query_ollama_v2(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3",  # Use the appropriate model identifier
        "prompt": prompt
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        
        responses = response.text.strip().split('\n')

        collected_responses = []

        for response_str in responses:
            try:
                parsed_response = json.loads(response_str)
                if parsed_response.get("response"):
                    collected_responses.append(parsed_response["response"])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

        final_output = ''.join(collected_responses)


        if final_output:
            return final_output.strip()
        else:
            return "I'm sorry, I don't know the answer."

    except ValueError as ve:
        print(f"JSON decoding error: {ve}")
        raise Exception(f"Ollama API call failed with invalid JSON. Response: {response.text}")


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Create a function to interact with the Ollama API
    llm = query_ollama_v2

    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    
    return qa


def final_result(query):
    qa_result = qa_bot()
    prompt = f"Question: {query}"
    answer = query_ollama_v2(prompt)  
    return answer

# Main execution loop 
if __name__ == "__main__":
    tokens_count = 0
    while tokens_count <= 2000:
        query = input("YOU: ")
        ques_tok = count_tokens(query)
        tokens_count += ques_tok
        
        try:
            
            prompt = f"Question: {query}"
            answer = query_ollama_v2(prompt)  
            ans_tok = count_tokens(answer)
            tokens_count += ans_tok
            
            print(f"ADA: {answer}")
            print(f"Tokens used: Question - {ques_tok}, Answer - {ans_tok}")
            
            data = [[query, ques_tok, answer, ans_tok]]
            with open('chatbot_data.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(data)
        except Exception as e:
            print(f"Error occurred: {e}")
            break
    
    print("ADA: You have expired your tokens! Please speak to the customer care!")
    
    