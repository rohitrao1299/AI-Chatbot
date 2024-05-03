from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import sqlite3
import speech_recognition as sr

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries. If you don't know the answer, say don't know "),
        ("user","Question:{question}")
    ]
)

## LLAMA2 LLm
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form.get('input_text')
    if input_text:
        response = chain.invoke({"question":input_text})
    else:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                input_text = r.recognize_google(audio, language="en-IN")
                response = chain.invoke({"question":input_text})
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