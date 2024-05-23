import datetime
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pypdf
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import glob

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HUGGING_FACE_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

app = Flask(__name__)
CORS(app)

# Load PDFs from DATA folder
pdf_loader = []
for file_path in glob.glob("DATA/*.pdf"):
    try:
        loader = PyPDFLoader(file_path)
        pdf_loader.append(loader)
    except pypdf.errors.PdfReadError as e:
        print(f"Error loading {file_path}: {e}")
        continue

# Combine PDF documents
documents = []
for loader in pdf_loader:
    documents.extend(loader.load())

# Create a text splitter
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=200)

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# Create a LLM
llm = Ollama(model="llama2")

# Create a vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create a prompt template
prompt_template = (
    "You are a helpful assistant. Answer the question based on the following context only:\n"
    "{context}\n"
    "If the context does not provide sufficient information, respond with 'The information is not available in the provided documents.'\n"
    "Question: {question}\n"
    "Answer:"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template)
    ]
)

# Create an output parser
output_parser = StrOutputParser()

# Create a chain
chain = prompt | llm | output_parser

def get_similar_documents(question):
    # Perform similarity search
    similar_documents = vectorstore.similarity_search(question)
    context = ""
    for doc in similar_documents:
        context += doc.page_content + "\n"
    return context

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form.get('input_text')
    if input_text:
        question = input_text
        # Get similar documents based on the user's question
        context = get_similar_documents(question)
        if not context.strip():
            response = "The information is not available in the provided documents."
        else:
            # Prepare the input for the chain
            input_data = {"context": context, "question": question}
            # Invoke the chain with the user's question and relevant documents
            response = chain.invoke(input_data)
    else:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                input_text = r.recognize_google(audio, language="en-IN")
                question = input_text
                # Get similar documents based on the user's question
                context = get_similar_documents(question)
                if not context.strip():
                    response = "The information is not available in the provided documents."
                else:
                    # Prepare the input for the chain
                    input_data = {"context": context, "question": question}
                    # Invoke the chain with the user's question and relevant documents
                    response = chain.invoke(input_data)
            except sr.UnknownValueError:
                response = "Sorry, I didn't understand what you said."
            except sr.RequestError as e:
                response = "Error: " + str(e)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
