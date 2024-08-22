from flask import Flask, request, jsonify
import logging
import os
from transformers import pipeline
from PIL import Image
import io  # For handling byte streams
import openai  # Import openai directly
from flask_cors import CORS
from dotenv import load_dotenv
import time  # For handling retries

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
openai.api_key = api_key  # Set the API key for the openai module

# Load the CLIP image classification pipeline
image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# Define the function for image classification to get the text
def classify_image(image, labels):
    results = image_classifier(images=image, candidate_labels=labels)
    predicted_label = results[0]['label']
    return predicted_label

# Function to generate an image based on the incorrect classification
def generate_image(prompt):
    response = openai.Image.create(
        prompt=f"view a {prompt}",
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

@app.route('/classify-image', methods=['POST'])
def classify_image_route():
    # Check if an image file is included in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    language = request.form.get('language', 'English')
    option = request.form.get('option', 'Colors')

    # Read the image file from the request
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))

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
        # Check if the classification matches the expected input (modify 'expected_color_or_shape' as needed)
        if predicted_label.lower() == "expected_color_or_shape":  # Replace with the actual expected label
            return jsonify({
                "predicted_label": predicted_label,
                "message": "Correct classification"
            })
        else:
            # Generate a new image if classification is incorrect
            generated_image_url = generate_image(predicted_label)
            return jsonify({
                "predicted_label": predicted_label,
                "message": "Incorrect classification, generating a new image",
                "generated_image_url": generated_image_url
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
