from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
import os

app = Flask(__name__)

# Load or generate the FAISS index when the server starts
index_dir = "car_faiss"
faiss_index = None

def create_faiss_index():
    global faiss_index
    # URL of the Wikipedia page on cars
    url = "https://en.wikipedia.org/wiki/Car"

    # Send a GET request to fetch the raw HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the main content of the page
    content = soup.find('div', {'class': 'mw-parser-output'})

    # Extract all paragraphs
    paragraphs = content.find_all('p')

    # Combine paragraphs into a single text
    car_info = "\n".join([para.get_text() for para in paragraphs])

    # Split the car_info into chunks (e.g., sentences)
    car_info_chunks = car_info.split('\n')

    # Wrap the model with LangChain's SentenceTransformerEmbedding
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Convert the text chunks into LangChain Document objects
    documents = [Document(page_content=chunk) for chunk in car_info_chunks]

    # Create a FAISS Vectorstore using LangChain's FAISS wrapper
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    faiss_index = FAISS.from_documents(documents, embedding_model)
    faiss_index.save_local(index_dir)
    print("FAISS index created and saved.")

# Call this function once when the app starts
create_faiss_index()

# Initialize the LLM for QA
model_path = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=2048,
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    verbose=True,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_index.as_retriever(),
    chain_type="stuff"
)

# API endpoint to handle queries
# Serve the HTML page on root route
@app.route("/")
def home():
    return render_template("index.html")  # This serves the index.html file

@app.route("/query", methods=["POST"])
def query_knowledge_base():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Run the QA chain to get the answer
    response = qa_chain.run(query)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
