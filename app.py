from flask import Flask, request, jsonify
import logging
import os
from transformers import pipeline
from PIL import Image
import openai  # Import openai directly
from flask_cors import CORS
from dotenv import load_dotenv
import time
from werkzeug.utils import secure_filename
import sqlite3 
from pathlib import Path
from gtts import gTTS
from flask import send_file





# Define lists of colors and shapes in different languages
list_of_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'black', 'white', 'purple', 'pink']
list_of_colors_arabic = ['أحمر', 'أزرق', 'أخضر', 'أصفر', 'برتقالي', 'بني', 'أسود', 'أبيض', 'بنفسجي', 'وردي']
list_of_colors_french = ['rouge', 'bleu', 'vert', 'jaune', 'orange', 'marron', 'noir', 'blanc', 'violet', 'rose']

list_of_shapes = ['circle', 'square', 'rectangle', 'triangle', 'diamond', 'oval']
list_of_shapes_arabic = ['دائرة', 'مربع', 'مستطيل', 'مثلث', 'معين', 'بيضوي']
list_of_shapes_french = ['cercle', 'carré', 'rectangle', 'triangle', 'losange', 'ovale']



# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_llama_client = openai

# Access the API key from environment variables
api_key = os.getenv('API_KEY')

API_KEY1 =os.getenv('API_KEY1')

openai_llama_client.api_key = API_KEY1



# Initialize OpenAI client
openai.api_key = api_key  # Set the API key for the openai module



app = Flask(__name__)


# Define the path for storing speech files
def get_speech_file_path(filename="speech.mp3"):
    return Path(__file__).parent / filename

# Function to generate speech from text
def generate_speech(text, filename="speech.mp3"):
    tts = gTTS(text=text, lang='en')
    speech_file_path = get_speech_file_path(filename)
    tts.save(speech_file_path)
    return speech_file_path

# Function to transcribe audio using OpenAI Whisper model
def transcribe_audio(file):
    try:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=file,
        )
        return response['text']
    except Exception as e:
        return str(e), 400

# Function to save transcription to a text file
def save_transcription_to_file(transcription_text):
    with open('transcriptions.txt', 'a') as f:
        f.write(transcription_text + '\n')

# Load the CLIP image classification pipeline
image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# Define the function for image classification to get the text
def classify_image(image, labels):
    results = image_classifier(images=image, candidate_labels=labels)
    predicted_label = results[0]['label']
    return predicted_label

# Function to generate an image based on the incorrect classification
def generate_image(prompt):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=f"view a {prompt}",
        n=1,
        size="1024x1024"
    )
    return response.data[0].url

#Image classification api

# Image classification API
@app.route('/classify-image', methods=['POST'])
def classify_image_route():
    data = request.json
    language = data.get('language')
    option = data.get('option')
    image_data = data.get('image_data')  # Image data sent as base64

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the Base64 image data
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

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
        # Check if the classification matches the input
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


@app.route('/get-colors', methods=['GET'])
def get_colors():
    language = request.args.get('language', 'English')  # Default to English if no language is provided
    print(f"Received language: {language}")

    if language == 'English':
        colors = list_of_colors
    elif language == 'Arabic':
        colors = list_of_colors_arabic
    elif language == 'French':
        colors = list_of_colors_french
    else:
        return jsonify({"error": "Language not supported"}), 400
    
    return jsonify({"colors": colors})

@app.route('/get-shapes', methods=['GET'])
def get_shapes():
    language = request.args.get('language', 'English')  # Default to English if no language is provided
    
    if language == 'English':
        shapes = list_of_shapes
    elif language == 'Arabic':
        shapes = list_of_shapes_arabic
    elif language == 'French':
        shapes = list_of_shapes_french
    else:
        return jsonify({"error": "Language not supported"}), 400
    
    return jsonify({"shapes": shapes})

def save_transcription_to_file(transcription_text):
    with open('transcriptions.txt', 'a') as f:
        f.write(transcription_text + '\n')

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            transcription_text = transcribe_audio(file)
            
            if isinstance(transcription_text, tuple):
                return jsonify({"error": transcription_text[0]}), transcription_text[1]
            
            # Save the transcription to the file instead of the database
            save_transcription_to_file(transcription_text)
            
            return jsonify({"transcription": transcription_text})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File upload failed"}), 400


@app.route('/generate-speech', methods=['POST'])
def generate_speech_route():
    data = request.get_json()

    # Validate input
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']

    try:
        speech_file_path = generate_speech(text)
        return send_file(speech_file_path, mimetype='audio/mp3')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
if __name__ == '__main__':
    app.run(debug=True)
