from flask import Flask, request, jsonify
from utils2 import *  # Import your chatbot function
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config import *

app = Flask(__name__)
prompt_template = set_custom_prompt()

# db2=FAISS.load_local("/root/rag_chatbot/vectorstores/db_faiss", embeddings, allow_dangerous_deserialization=True)

# Define the chatbot endpoint
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    query = data.get("query")
    
    # Ensure the query is provided
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Generate the response from the chatbot function
        prompt_template = set_custom_prompt()
        response = generate_and_validate_response(query, db, prompt_template)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Error generating response: {e}"}), 500

# Additional endpoints can go here if needed

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
