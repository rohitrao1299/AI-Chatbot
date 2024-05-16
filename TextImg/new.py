import sqlite3
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import random
import PyPDF2
import io
import speech_recognition as sr
from werkzeug.utils import secure_filename
import PIL
from PIL import Image, ImageEnhance, ImageFilter

# Flask App Initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

# Load environment variables
load_dotenv()

# Language Model Setup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from diffusers import DiffusionPipeline, StableDiffusionPipeline


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template for Language Model
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries. Give response from the uploaded pdf also if ask question from their own . If you don't know the answer, say don't know "),
        ("user", "Question:{question}")
    ]
)

# LLM Model
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Utility Functions
import json
# Function to summarize URL content
def summarize_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = soup.find('title').text

        # Extract meta description
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            summary = meta_description['content']
        else:
            summary = title

        # Filter out unwanted elements (prices and ratings)
        for unwanted in soup.select('.price, .rating'):
            unwanted.extract()

        # Extract text content from paragraphs
        paragraphs = soup.find_all('p')
        text_content = ' '.join([p.text.strip() for p in paragraphs])

        # Combine title, meta description, and text content to form the summary
        full_summary = {
            'Title': title,
            'Summary': summary,
            'Content': text_content
        }

        return full_summary
    except Exception as e:
        return {"error": f"Error summarizing URL: {str(e)}"}




# Define the upload directory and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER    

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDF
def extract_text_from_pdf(filename):
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Flask App Routes

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = request.form.get('input_text')
    if not input_text:
        return jsonify({'response': "No input provided"})
    try:
        if input_text.startswith('http'):  # Check if input is a URL
            response = summarize_url(input_text)
        else:
            response = "I can only summarize links. Please provide a URL."
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"})






@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form.get('input_text')
    if input_text:
        response = chain.invoke({"question": input_text})
    else:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                input_text = r.recognize_google(audio, language="en-IN")
                response = chain.invoke({"question": input_text})
            except sr.UnknownValueError:
                response = "Sorry, I didn't understand what you said."
            except sr.RequestError as e:
                response = "Error: " + str(e)
    return jsonify({'response': response})


# Define the prompts for image generation
PROMPTS = [
    "A group of officers in a meeting room discussing military strategies for an upcoming mission.",
    "A team of soldiers navigating an obstacle course, with a rugged terrain in the background.",
    "An army officer leading a team of soldiers in a combat situation, with a cityscape in the background.",
    "A group of candidates working together to overcome a challenging situation.",
    "A leader comforting a team member who is feeling overwhelmed, with a cityscape in the background.",
    "A group of soldiers planning a surprise attack on an enemy base, with a desert landscape in the background.",
    "A team of firefighters working together to extinguish a large fire in a commercial building.",
    "A group of doctors and nurses working together in a hospital to save a patient's life.",
    "A team of astronauts conducting a spacewalk to repair a damaged satellite.",
    "A group of soldiers climbing a steep mountain, with a snowy landscape in the background.",
    "A team of scientists conducting an experiment in a high-tech laboratory.",
    "A group of police officers working together to solve a complex case.",
    "A team of engineers building a bridge over a wide river.",
    "A group of artists collaborating on a large-scale mural painting.",
    "A team of chefs working together to prepare a gourmet meal in a busy restaurant kitchen.",
    "A team of soldiers on a trek, crossing a river and overcoming natural obstacles.",
    "A group of students studying in a classroom and teacher writing on the blackboard.",
    "A group of passengers travelling in a bus or train and overcoming natural obstacles.",
    "A team of players celebrating that won the championship game, with a packed stadium and cheering fans in the background.",
    "A young boy and girl working together to build a sandcastle on the beach, with a sunny sky and ocean waves in the background.",
    "A boy trying to molest a girl in a park and people are walking around them.",
    "A group of adventurous friends hiking through a dense forest, with tall trees, colorful leaves, and a clear blue sky in the background. They are navigating the trail, pointing out interesting plants and animals, and sharing stories and laughter.",
    "A beautiful outdoor wedding ceremony, with a bride and groom exchanging vows under a floral archway, surrounded by their loved ones. The sun is shining, and there are colorful flowers and greenery everywhere.",
    "An exciting science exhibition showcasing technology and innovative projects, with students and researchers presenting their work to a curious and engaged audience. There are robots, 3D printers, virtual reality headsets, and other high-tech gadgets on display.",
    "A bustling fair with a strong police presence, ensuring the safety and security of all visitors. The police officers are patrolling the area, interacting with the public, and providing guidance as needed. There are games, rides, food stalls, and entertainment.",
    "A peaceful farming scene, with a farmer tending to his crops in a lush green field, surrounded by rolling hills and a clear blue sky. There are rows of healthy plants, irrigation systems, and farm equipment in the background.",
    "A vibrant and bustling Indian village, with narrow streets, colorful houses, and a lively marketplace. There are street vendors selling spices, textiles, and handicrafts, as well as a temple, a school, and a community center.",
    "A meeting of the local Indian panchayat, with community leaders and elders gathered to discuss important issues and make decisions that affect the village. There are men and women of all ages, dressed in traditional Indian clothing, sitting in a circle and engaging in respectful and productive dialogue.",
    "A traditional and sustainable Indian village agricultural operation, with small fields of crops, manual irrigation systems, and ox-driven plows. There are farmers working in the fields, planting and harvesting crops by hand, and a community of villagers gathering to share knowledge and resources.",

    # Add more prompts as needed
]

# Function to generate AI-based images using Stable Diffusion
def generate_image_using_stable_diffusion(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    image = pipe(prompt).images[0]

    # Apply transformations to generate black & white and blur features
    bw_layer = image.convert('L')  # Convert to black & white
    # blur_layer = image.filter(ImageFilter.GaussianBlur(radius=5))  # Apply Gaussian blur
    # sketch_layer = image.filter(ImageFilter.FIND_EDGES)  # Apply edge detection (sketch effect)


    # Create a new image with the same size and mode as the original image
    new_image = Image.new(image.mode, image.size)

    # Combine the two layers into the new image
    new_image.paste(bw_layer, (0, 0))
    # new_image.paste(blur_layer, (0, 0))
    # new_image.paste(sketch_layer, (0, 0))

    return new_image

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Select a random prompt from the list
        prompt = random.choice(PROMPTS)
        image_output = generate_image_using_stable_diffusion(prompt)

        # Convert the image to JPEG format and set the quality
        image_output = image_output.convert('RGB')
        img_io = io.BytesIO()
        image_output.save(img_io, format='JPEG', quality=85)
        img_io.seek(0)

        # Return the generated image as a response
        # return send_file(img_io, mimetype='image/jpeg')
        return img_io.getvalue(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files.get('file')  # Use request.files.get('file') to get the file object
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'response': "PDF file uploaded successfully. You can now start asking questions."})
    return jsonify({'error': 'Invalid file format'})


@app.route('/ask_question', methods=['POST'])
def ask_question():
    input_text = request.form.get('input_text')
    if input_text:
        response = chain.invoke({"question": input_text})
    else:
        response = "Please provide a question."   
    
    return jsonify({'response': response})

# Define the prompts for generating multiple-choice questions
mcq_prompt = ChatPromptTemplate.from_messages([
    ("user", "Generate multiple-choice questions:\n\n{text}")
])

# Define Ollama model for MCQ generation
ollama_mcq_model = Ollama(model="llama2")
output_parser_mcq = StrOutputParser()
mcq_chain = mcq_prompt | ollama_mcq_model | output_parser_mcq

@app.route('/generate_mcqs', methods=['POST'])
def generate_mcqs():
    input_text = request.json.get('input_text')  # Access JSON data instead of form data
    if input_text:
        # Generate MCQs using Ollama (LLama2) model
        mcq_response = mcq_chain.invoke({"text": input_text})
        mcqs = mcq_response.strip().split('\n')
        return jsonify({'mcqs': mcqs})
    else:
        return jsonify({'error': 'Please provide input text'})
    
@app.route('/books',methods=['GET'])
def books():
    return "Everything is in books"    



if __name__ == '__main__':
    app.run(debug=True)