from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from ultralytics import YOLO
import requests
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Configuration for Edamam API
APP_ID = '6408aaaf'
APP_KEY = '38b3802b487f80df555360b87a6e6fac'

# Load your YOLOv8 model
class VegetableClassifier:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image):
        # Convert PIL image to NumPy array
        image_np = np.array(image)
        results = self.model(image_np)
        
        # Initialize a list to store extracted information
        detections = []
        
        # Extract class names and bounding boxes
        for result in results:
            for detection in result.boxes:
                # Convert Tensor to list
                class_id = int(detection.cls.item())
                class_name = self.model.names[class_id]  # Get class name from ID
                
                # Extract bounding box coordinates
                bbox = detection.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Append the detection information
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(detection.conf.item()),  # Convert Tensor to float
                    'bounding_box': bbox
                })
        
        # Prepare the response with detected vegetables
        vegetable_counts = {}
        for detection in detections:
            name = detection['class_name']
            if name in vegetable_counts:
                vegetable_counts[name] += 1
            else:
                vegetable_counts[name] = 1

        return {
            'detections': detections,
            'vegetable_counts': vegetable_counts
        }

# Initialize your vegetable classifier
vegetable_classifier = VegetableClassifier(model_path="best.pt")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Load the image
        image = Image.open(file).convert("RGB")
        
        # Classify the image
        prediction_results = vegetable_classifier.predict(image)

        # Extract vegetable names for Edamam API query
        vegetable_list = list(prediction_results['vegetable_counts'].keys())
        query = ','.join(vegetable_list)

        # URL for Edamam Recipe API
        VEGETABLES_API_URL = f'https://api.edamam.com/search?q={query}&app_id={APP_ID}&app_key={APP_KEY}'

        # Make a GET request to the API
        response = requests.get(VEGETABLES_API_URL, timeout=10)
        data = response.json()
        
        # Prepare the response with recipes
        hits = data.get('hits', [])
        recipes = []

        for hit in hits:
            recipe = hit['recipe']
            recipes.append({
                'label': recipe['label'],
                'image': recipe['image'],
                'source': recipe['source'],
                'calories': round(recipe['calories'], 2),
                'dietLabels': recipe['dietLabels'],
                'url': recipe['url']
            })

        # Prepare final response
        response = {
            'vegetable_counts': prediction_results['vegetable_counts'],
            'detections': prediction_results['detections'],
            'recipes': recipes
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/popular-recipes', methods=['GET'])
def get_popular_recipes():
    url = f'https://api.edamam.com/search?q=vegetable&app_id={APP_ID}&app_key={APP_KEY}&from=0&to=5&sort=popularity'
    
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
       
        data = response.json()
        return jsonify(data['hits']), 200
    else:
        return jsonify({'error': 'Failed to fetch popular recipes'}), 500

    
if __name__ == '__main__':
    app.run(debug=True)
