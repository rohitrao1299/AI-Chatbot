import base64
from concurrent.futures import ThreadPoolExecutor
import io
import time
from datetime import datetime, timedelta
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
import torch
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
    "A cyberpunk hacker with a bright mohawk hairstyle, leather jacket, and cybernetic implants, typing code on a holographic computer",
  "An astronaut in a sleek white spacesuit planting the American flag on Mars, red dust and rocks in background, Earth visible in sky",
  "A beautiful female android with long silver hair, glowing blue eyes, pale skin, standing in slick futuristic nightclub",
  "A scientist in a lab coat developing nanobots, high tech microscopes and equipment in background",
  "A pilot with an oxygen mask flying a stealth spaceship through dense asteroid field, firing lasers, cockpit visible",
  "A combat cyborg with black bionic arms playing an electric guitar during an apocalyptic thunderstorm, surrounded by rubble",
  "A classic robot butler in suit neatly serving tea to elegant lady on hovering chairs in pristine white room with view of cityscape",
  "An elf ranger with bow aimed, long hair, cloak, armor, standing amid mystical ruins overgrown with vines",
  "A fierce post-apocalyptic warrior battling mutated beasts in abandoned city streets filled with rubble and overgrown plants",
  "An engineer building a teleportation device, lab filled with computers, cables, and glowing portal",
  "A wizard with staff casting a magic spell, mystical runes surrounding, ancient ruins in misty mountains behind",
  "A pale vampire with fangs at futuristic nightclub filled with bright neon lights, dancing crowd wearing stylish outfits",
  "A soldier in futuristic armor aiming a laser rifle, muzzle flash visible, battlefield filled with smoke behind",
  "A young hacker programming holographic computer displaying complex code, warehouse with graffiti and high tech equipment around",
  "A samurai with robotic armor and laser katana fighting an enormous, fire-breathing dragon roaring",
  "A muscular mutant superhero with chainsawed arms rescuing humans from the rubble of a collapsed building",
  "A steampunk inventor wearing goggles tinkering on a mechanical robot as gears turn, pipes vent steam in workshop",
  "A Jedi Knight wielding a lightsaber leaving a trail of light, facing off against an enemy with a red lightsaber in nighttime forest",
  "A pilot controlling a large battle mech, engaging enemy robots and tanks on city streets filled with explosions",
  "A cowboy with facemask riding a galloping robotic horse with rocket boosters through the desert near pyramids and a spaceship crash site",
  "A beautiful cyborg DJ at flashy neon rave wearing LED glasses and arm implants, deadmau5-style helmet, operating glowing holographic turntables",
  "A smiling girl hugging her tall, lean four-legged robot dog with screens for eyes in a sunny green meadow dotted with flowers",
  "A classic square robot with an artist's beret carefully painting an abstract picture on canvas in art studio with paint tubes and brushes",
  "A ninja assassin clad in black with masks leaping between high tech skyscrapers at night, throwing knives at armored target",
  "An old wizard with long white beard wearing robes and pointed hat riding a sleek, colorful hoverboard through mysterious ruins",
  "A hacker jacked into virtual reality, coding on screens displaying cyberspace environment with digital rain and architectural structures",
  "A female elf ranger with bow crouching in vivid lush forest with sunlight streaming through the trees behind her",
  "A neural-linked gamer wearing a headset controlling a humanoid robot fighting in global tournament arena filled with drones and holographic crowds",
  "A diverse crew including humans, cyborgs, androids on the bridge of an exploratory starship venturing into deep space about to engage warp speed, captain in center seat",
  "An android with exposed machinery playing speed chess with a human opponent in slick room with bookshelves and trophy case",
  "A girl befriending a cute fluffy alien with large dark eyes sitting together among grass and alien mushrooms glowing under beautiful night sky",
  "A scientist injecting himself with experimental nanomachines, body merging with mechanical parts with blue glow visible through seams in skin",
  "A shiny humanoid robot with tracks instead of legs exploring Mars desert, red rocks and cliffs visible under orange sky",
  "A futuristic spaceship with glowing engines engaging warp speed, leaving stretched star trails behind approaching massive gas giant planet with rings",
  "A young hacker programming next-gen AI displaying code on multiple holographic screens inside warehouse with graffiti",
  "A battlefield depicting the aftermath of a violent robot uprising against humans, damaged machinery leaking oil and sparks among rubble",
  "A sleek cyborg assassin levitating off the ground using built-in antigravity tech, surrounded by floating holographic data streams in dark room",
  "An astronaut in advanced spacesuit taking first step onto surface of inhabitable exoplanet, expansive vista of alien landscape and sky visible ",
  "A biohacker in makeshift basement lab injecting neon glowing DNA vials, genetic engineering equipment with blinking screens around",
  "A smiling girl engaging in friendly conversation with a hovering orb-shaped AI displaying animated expressions - cityscape visible through windows behind",
  "A classic robot butler in suit neatly serving tea to masked elites seated on hovering chairs in pristine penthouse suite with expansive city views",
  "A futuristic battleship leading a small fleet against a larger mothership over earth, visible in background through cockpit window, firing bright weapons leaving glowing trails",
  "An android with exposed machinery thoroughly engrossed in playing immersive sci-fi virtual reality game, cables linking body to computer servers",
  "A hoverbike gang riding sleek colorful bikes leaving neon trails through nighttime city streets lined with holographic billboards and signs",
  "A hacker jacked into the virtual architecture of a sinister megacorporation consisting of ominous towering structures, coding on screens showing cyberspace",
  "A scientist developing neural implants and prosthetic limbs for radical new human augmentation procedures in lab filled with computers, robotic arms, and vials",
  "A fighter pilot targeting enemy attack drones during intense aerial dogfight, firing missiles while performing evasive maneuvers, wings of advanced jet visible in cockpit",
  "A shapeshifting assassin android impersonating a human to infiltrate high-security gala, preparing to draw handgun concealed under elegant evening gown",
  "A damaged explorer robot examining mysterious glowing hieroglyphs inside ancient towering alien ruins, strange structures and technologies overgrown with vines behind",
  "The crew of a sleek exploratory starship entering cryogenic pods before faster-than-light journey, automated systems visible maintaining ship operations",
  "A hooded hacker decrypting data stolen from crash-landed alien scout ship, strings of symbols on computer screens inside warehouse with graffiti and exposed brick",
  "A smiling girl affectionately hugging cute short robot companion with blinking monitor for a face, grass and blue sky in background",
  "An astronaut triumphantly planting American flag on surface of Mars, bootprints in red soil with Earth visible in orange sky above",
  "A scientist carefully monitoring array of computer servers, cables linking equipment to jar with synthetic brain inside glowing yellow liquid",
  "A female android wearing 1950s style uniform smiling while efficiently serving milkshakes to laughing customers at neon-lit diner with spaceship visible through window",
  "A pilot in advanced spacesuit ejecting from cockpit of experimental aircraft, firing built-in rocket boosters, flames and smoke trails visible behind",
  "A hacker stealing corporate secrets in virtual reality representation of office tower, coding on screens showing cyberspace architecture",
  "A menacing cyborg with half its head replaced by glowing machinery playing chess against an unbeatable AI supercomputer in sleek room",
  "A genetically engineered dinosaur with spiked tail kept on chain leash by soldier in futuristic armor, abandoned buildings visible behind rubble",
  "A classic robot butler managing smart home, efficient serving dinner to family while monitoring various devices and appliances simultaneously",
  "The crew of a Federation starship including humans, aliens, androids gathering for first contact ritual, bridge filled with slick interfaces, planet visible through window",
  "An assassin android impersonating doctor to infiltrate prison, readying lethal injection hidden in coat",
  "An affectionate girl gently embracing her dog-sized robot companion with tank treads and a single blinking eye for a head",
  "A muscular cyborg prize fighter covered in tattoos and cybernetics battling a menacing opponent in caged arena surrounded by cheering audience",
  "A wizard experimenting with combining futuristic tech like forcefields, levitation, and advanced genetics with magical essences, artifacts, and spellbooks",
  "The crew of a exploratory vessel looking shocked as their ship is flung into the outer solar system after narrowly surviving an experimental faster than light engine misjump",
  "A hacker manipulating global stock prices with orbital quantum computer running advanced AI stock trading algorithms, screens displaying dizzying data dashboards",
  "A musician directing an orchestra made up of various aliens including tall, elegant bird-like creatures playing lyrical stringed instruments",
  "A barely functioning explorer robot caked in red dust examining strange slimy liquid oozing from orifices of biomechanical structures half-buried on hellish volcanic Venusian surface",
  "A massive battlecruiser firing all weapons against reptilian capital ship in orbit over gas giant planet with swirling eyes and many moons while several smaller craft are destroyed around it",
  "An astronaut spacewalking outside slowly rotating space station near luminous nebula, performing repairs on damaged solar panel torn open by micrometeorite",
  "A scientist overseeing construction of a microscopic nanobot swarm designed for programmable biological editing, petri dishes filled with the bots visible under microscopes",
  "An alien ambassador with large head, slender body, big glossy eyes signing treaty documents as cameras flash at gathering of various press and leaders in front of United Nations building with flags out front",
  "A smiling blonde female android wearing 1950's uniform roller skating around flashing neon diner taking orders from seated customers which include aliens and robots",
  "The crew of the starship Galileo looking stunned by large creature on viewscreen rumbling deep incomprehensible noises during tense first contact event on planet with pink sky visible from hull",
  "A battered, dusty six legged scavenger robot examining wreckage of ancient human skyscrapers being reclaimed by desert sand with a storm brewing in red distance",
  "Interior view of experimental warp fighter cockpit with softly glowing controls and HUD helmet as test pilot prepares to push craft past light speed barrier",
  "A stealthy assassin android with night vision eyes infiltrating top secret research base, silently taking out armored guards with knife and garrote wire",
  "A hacker remotely infiltrating systems of Arasaka Corporation tower through virtual cyberspace of swirling data, coding on terminal displaying access nodes",
  "A ragged crew narrowly escaping from the massive beak of giant space-dwelling star kraken amid debris of destroyed ships with laser blasts visible ",
  "A futuristic armored soldier psyching up cleverly disguised, bio-engineered alternate dinosaur before releasing it onto battlefield filled with chaos",
  "An astronaut haunted by frightening tentacled entities only he can perceive, futilely struggling as they drag him toward dark pit on icy surface of moon Europa",
  "An android with perfected human mannerisms eagerly answering trivia questions on stage surrounded by cheering audience and swirling lights",
  "A neon gang member racing law enforcement hover drones through decaying skyscrapers on supersonic hover bike leaving glowing particle trails",
  "A wizard adept in both mystic arts and illegal tech spare parts channeling lightning while battling mercenaries and gun turrets",
  "Crew of experimental warpship regaining stunned senses as disabled craft tumbles out of control with streaking stars visible through cockpit window",
  "A scientist in biohazard suit carefully dissecting elongated skull and tentacled extremities of an alien corpse on operating table in underground bunker",
  "A desperate rebel freedom fighter racing across battlefield strewn with titanic mechs and wreckage toward extraction ship waiting open in the smoke-filled skies",
    
    # Add more prompts as needed
]


# # Function to generate AI-based images using Stable Diffusion
# def generate_image_using_stable_diffusion(prompt):
#     pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
#     image = pipe(prompt,num_inference_steps=12, guidance_scale=7.5).images[0]

#     # Apply transformations to generate black & white and blur features
#     bw_layer = image.convert('L')  # Convert to black & white
#     # blur_layer = image.filter(ImageFilter.GaussianBlur(radius=5))  # Apply Gaussian blur
#     # sketch_layer = image.filter(ImageFilter.FIND_EDGES)  # Apply edge detection (sketch effect)


#     # Create a new image with the same size and mode as the original image
#     new_image = Image.new(image.mode, image.size)

#     # Combine the two layers into the new image
#     new_image.paste(bw_layer, (0, 0))
#     # new_image.paste(blur_layer, (0, 0))
#     # new_image.paste(sketch_layer, (0, 0))

#     return new_image

# @app.route('/generate', methods=['POST'])
# def generate():
#     if request.method == 'POST':
#         # Select a random prompt from the list
#         prompt = random.choice(PROMPTS)
#         image_output = generate_image_using_stable_diffusion(prompt)

#         # Convert the image to JPEG format and set the quality
#         image_output = image_output.convert('RGB')
#         img_io = io.BytesIO()
#         image_output.save(img_io, format='JPEG', quality=85)
#         img_io.seek(0)

#         # Return the generated image as a response
#         # return send_file(img_io, mimetype='image/jpeg')
#         return img_io.getvalue(), 200, {'Content-Type': 'image/jpeg'}

# In-memory cache to store recently used prompts
recent_prompts = {}
CACHE_DURATION = timedelta(hours=1)  # Cache duration

# Clean up old entries from the cache
def clean_cache():
    now = datetime.now()
    keys_to_delete = [key for key, timestamp in recent_prompts.items() if now - timestamp > CACHE_DURATION]
    for key in keys_to_delete:
        del recent_prompts[key]

def generate_image_using_stable_diffusion(prompt):
     pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
     image = pipe(prompt,num_inference_steps=12, guidance_scale=7.5).images[0]
     bw_layer = image.convert('L')  # Convert to black & white
     img = Image.new(image.mode, image.size)
     img.paste(bw_layer, (0, 0))
     return img
   

@app.route('/generate', methods=['GET'])
def generate():
    if request.method == 'GET':
        clean_cache()  # Clean the cache before selecting a new prompt
        
        # Filter out recently used prompts
        available_prompts = [prompt for prompt in PROMPTS if prompt not in recent_prompts]
        
        if not available_prompts:
            return jsonify({"error": "All prompts have been used recently. Please try again later."})
        
        # Select a random prompt from the available prompts
        selected_prompt = random.choice(available_prompts)
        
        # Update the cache with the selected prompt
        recent_prompts[selected_prompt] = datetime.now()
        
        image_output = generate_image_using_stable_diffusion(selected_prompt)

        # Convert the image to JPEG format and set the quality
        image_output = image_output.convert('RGB')
        img_io = io.BytesIO()
        image_output.save(img_io, format='JPEG', quality=85)
        img_io.seek(0)

        # Encode the image to base64
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # Create the data URL
        image_url = f"data:image/jpeg;base64,{img_base64}"

        # Return the data URL as a JSON response
        return jsonify({"data": image_url})

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
    app.run(debug=True, host="0.0.0.0")