from flask import Flask, jsonify, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pypdf
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
import sqlite3
import glob

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HUGGING_FACE_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")  


app = Flask(__name__)

## Load PDFs from DATA folder
pdf_loader = []
for file_path in glob.glob("DATA/*.pdf"):
    try:
        loader = PyPDFLoader(file_path)
        pdf_loader.append(loader)
    except pypdf.errors.PdfReadError as e:
        print(f"Error loading {file_path}: {e}")
        continue

## Load web pages from multiple URLs
web_loaders = []
urls = ["https://www.thehindu.com/",
         "https://www.nytimes.com/", 
         "https://www.bbc.com/"]
for url in urls:
    web_loader = WebBaseLoader(url)
    web_loaders.append(web_loader)

web_documents = []
for loader in web_loaders:
    web_documents.extend(loader.load())    

## Combine PDF and web documents
documents = []
for loader in pdf_loader:
    documents.extend(loader.load())
documents.extend(web_documents)

## Create a text splitter
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=200)

## Split the documents into chunks
chunks = text_splitter.split_documents(documents)

## Create a LLm
llm = Ollama(model="llama2")

## Create a vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})
vectorstore = FAISS.from_documents(chunks, embeddings)

## Create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries based on the provided documents. If you don't know the answer, say 'I don't know.'"),
        ("user", "Question:{question}")
    ]
)

## Create an output parser
output_parser = StrOutputParser()

## Create a chain
chain = prompt | llm | output_parser

def get_similar_documents(question):
    # Perform similarity search
    similar_documents = vectorstore.similarity_search(question)
    response = ""
    for doc in similar_documents:
        response += doc.page_content
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form.get('input_text')
    if input_text:
        question = input_text
        # Get similar documents based on the user's question
        documents = get_similar_documents(question)
        # Invoke the chain with the user's question and relevant documents
        response = chain.invoke({"question": question, "documents": documents})
    else:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                input_text = r.recognize_google(audio, language="en-IN")
                question = input_text
                # Get similar documents based on the user's question
                documents = get_similar_documents(question)
                # Invoke the chain with the user's question and relevant documents
                response = chain.invoke({"question": question, "documents": documents})
            except sr.UnknownValueError:
                response = "Sorry, I didn't understand what you said."
            except sr.RequestError as e:
                response = "Error: " + str(e)
    save_chat_history(input_text, response)
    return jsonify({'response': response})

def save_chat_history(input_text, response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (input_text TEXT, response TEXT)''')
    c.execute("INSERT INTO chat_history VALUES (?,?)", (input_text, response))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)
