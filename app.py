from flask import Flask, request, jsonify
import logging
import os
from transformers import pipeline
from PIL import Image
import re
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Access the API key from environment variables
api_key = os.getenv('API_KEY')

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.aimlapi.com",  # Ensure this is the correct base URL for your API
)

# Load the CLIP image classification pipeline
image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# Define the function for image classification to get the text
def classify_image(image, labels):
    results = image_classifier(images=image, candidate_labels=labels)
    predicted_label = results[0]['label']
    return predicted_label

@app.route('/classify-image', methods=['POST'])
def classify_image_route():
    data = request.json
    language = data.get('language')  # Use get() to safely access the key
    option = data.get('option')      # Use get() to safely access the key

    # Load the image directly from a file (since it's static)
    image = Image.open(r'C:/Users/musta/backend_project/colorcircle.png') 

    # Define lists of colors and shapes in different languages
    list_of_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'black', 'white', 'purple', 'pink']
    list_of_shapes = ['circle', 'square', 'rectangle', 'triangle', 'diamond', 'oval']
    list_of_colors_arabic = ['أحمر', 'أزرق', 'أخضر', 'أصفر', 'برتقالي', 'بني', 'أسود', 'أبيض', 'بنفسجي', 'وردي']
    list_of_shapes_arabic = ['دائرة', 'مربع', 'مستطيل', 'مثلث', 'معين', 'بيضوي']
    list_of_shapes_french = ['cercle', 'carré', 'rectangle', 'triangle', 'losange', 'ovale']
    list_of_colors_french = ['rouge', 'bleu', 'vert', 'jaune', 'orange', 'marron', 'noir', 'blanc', 'violet', 'rose']

    # Determine labels based on the option selected
    if option == 'Colors':
        labels = list_of_colors if language == 'English' else list_of_colors_arabic if language == 'Arabic' else list_of_colors_french
    else:
        labels = list_of_shapes if language == 'English' else list_of_shapes_arabic if language == 'Arabic' else list_of_shapes_french

    predicted_label = classify_image(image, labels)

    if predicted_label:
        return jsonify({
            "predicted_label": predicted_label
        })
    else:
        return jsonify({"error": "No valid color or shape detected."}), 400


@app.route('/speak', methods=['GET'])
def speak():
    logger.info("Received a request to /speak endpoint")
    response_message = "API works! Text-to-speech functionality will be added here."
    logger.info(f"Responding with: {response_message}")

    return jsonify({
        "status": "success",
        "message": response_message
    })

if __name__ == '__main__':
    app.run(debug=True)
