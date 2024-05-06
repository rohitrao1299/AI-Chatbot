# app.py
import sqlite3
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests  # Add this import statement

from bs4 import BeautifulSoup

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

# Language Model Setup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries. If you don't know the answer, say don't know "),
        ("user", "Question:{question}")
    ]
)

llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Utility Functions
def summarize_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            summary = meta_description['content']
        else:
            summary = title
        return summary
    except Exception as e:
        return f"Error summarizing URL: {str(e)}"

def save_chat_history(input_text, response, db_conn):
    c = db_conn.cursor()
    c.execute("INSERT INTO chat_history (input_text, response) VALUES (?,?)", (input_text, response))
    db_conn.commit()

# Flask App Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form.get('input_text')
    if not input_text:
        return jsonify({'response': "No input provided"})

    db_conn = sqlite3.connect('chat_history.db')
    try:
        if input_text.startswith('http'):  # Check if input is a URL
            response = summarize_url(input_text)
        else:
            response = chain.invoke({"question": input_text})
        save_chat_history(input_text, response, db_conn)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"})
    finally:
        db_conn.close()

if __name__ == '__main__':
    app.run(debug=True)