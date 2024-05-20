import random
from flask import Flask, flash, render_template, request, send_file, send_from_directory
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import io
import os
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

# Define the prompt here
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
    image = pipe(prompt,num_inference_steps=12, guidance_scale=7.5).images[0]

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

# Function to handle image generation and serve the generated image
@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Select a random prompt from the list
        prompt = random.choice(PROMPTS)
        image_output = generate_image_using_stable_diffusion(prompt)

        # Convert the image to JPEG format and set the quality
        image_output = image_output.convert('RGB')
        img_io = io.BytesIO()
        image_output.save(img_io, format='JPEG', quality=95)
        img_io.seek(0)

        # Return the generated image as a response
        return send_file(img_io, mimetype='image/jpeg')

# Function to handle story submission
@app.route('/submit_story', methods=['POST'])
def submit_story():
    if request.method == 'POST':
        story = request.form.get('story', '')
        if not story:
            error = 'Please fill out this form'
            return render_template('home.html', error=error)
        return 'Story submitted successfully!'

# Define the home page route
@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)