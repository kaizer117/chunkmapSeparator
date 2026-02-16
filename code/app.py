from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute path - go up one level from backend/ to find web-app/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # This is your-project/
WEB_APP_PATH = os.path.join(PROJECT_ROOT, 'web-app')

# Sample SVG object templates
SVG_TEMPLATES = {
    'rectangles': [
        {'type': 'rect', 'x': 100, 'y': 100, 'width': 80, 'height': 60, 'fill': 'rgba(255, 0, 0, 0.3)', 'stroke': 'red'},
        {'type': 'rect', 'x': 300, 'y': 200, 'width': 120, 'height': 80, 'fill': 'rgba(0, 255, 0, 0.3)', 'stroke': 'green'},
        {'type': 'rect', 'x': 500, 'y': 150, 'width': 100, 'height': 100, 'fill': 'rgba(0, 0, 255, 0.3)', 'stroke': 'blue'},
    ],
    'circles': [
        {'type': 'circle', 'cx': 200, 'cy': 300, 'r': 40, 'fill': 'rgba(255, 255, 0, 0.3)', 'stroke': 'orange'},
        {'type': 'circle', 'cx': 400, 'cy': 400, 'r': 50, 'fill': 'rgba(255, 0, 255, 0.3)', 'stroke': 'purple'},
        {'type': 'circle', 'cx': 600, 'cy': 250, 'r': 35, 'fill': 'rgba(0, 255, 255, 0.3)', 'stroke': 'teal'},
    ],
    'paths': [
        {'type': 'path', 'd': 'M50 150 L150 150 L100 250 Z', 'fill': 'rgba(255, 128, 0, 0.3)', 'stroke': 'brown'},
        {'type': 'path', 'd': 'M250 350 L350 350 L300 450 L250 350', 'fill': 'rgba(128, 0, 128, 0.3)', 'stroke': 'magenta'},
    ]
}

@app.route('/')
def index():
    """Serve the main index.html"""
    return send_from_directory(WEB_APP_PATH, 'index.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    """Serve all static files from web-app directory"""
    # Security check
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    
    return send_from_directory(WEB_APP_PATH, filename)

@app.route('/api/get-svg-objects', methods=['GET'])
def get_svg_objects():
    """
    Endpoint to get SVG objects
    In a real application, this would do some processing
    Returns a random combination of SVG objects
    """
    print("Received request for SVG objects")
    
    # Simulate some backend processing
    # Randomly select objects from templates
    num_objects = random.randint(2, 5)
    selected_objects = []
    
    for i in range(num_objects):
        # Randomly choose object type
        category = random.choice(['rectangles', 'circles', 'paths'])
        template = random.choice(SVG_TEMPLATES[category])
        
        # Create a copy and add unique ID
        obj = template.copy()
        obj['id'] = f"{category[:-1]}-{i}-{random.randint(1000, 9999)}"
        
        # Add some random variation
        if 'x' in obj:
            obj['x'] += random.randint(-20, 20)
        if 'y' in obj:
            obj['y'] += random.randint(-20, 20)
        if 'cx' in obj:
            obj['cx'] += random.randint(-20, 20)
        if 'cy' in obj:
            obj['cy'] += random.randint(-20, 20)
            
        selected_objects.append(obj)
    
    return jsonify({
        'status': 'success',
        'svgObjects': selected_objects,
        'count': len(selected_objects)
    })

@app.route('/api/get-new-svg-objects', methods=['GET'])
def get_new_svg_objects():
    """
    Endpoint to get new SVG objects (different from previous)
    Returns a different set of SVG objects
    """
    print("Received request for new SVG objects")
    
    # Simulate different backend processing
    # Return a different combination
    num_objects = random.randint(3, 6)
    selected_objects = []
    
    for i in range(num_objects):
        category = random.choice(['rectangles', 'circles', 'paths'])
        template = random.choice(SVG_TEMPLATES[category])
        
        obj = template.copy()
        obj['id'] = f"{category[:-1]}-new-{i}-{random.randint(1000, 9999)}"
        
        # Add more variation
        if 'width' in obj:
            obj['width'] += random.randint(-20, 40)
        if 'height' in obj:
            obj['height'] += random.randint(-20, 40)
        if 'r' in obj:
            obj['r'] += random.randint(-10, 20)
            
        selected_objects.append(obj)
    
    return jsonify({
        'status': 'success',
        'svgObjects': selected_objects,
        'count': len(selected_objects),
        'message': 'New SVG objects generated'
    })

@app.route('/api/select-object', methods=['POST'])
def select_object():
    """
    Endpoint to handle object selection (optional)
    """
    data = request.get_json()
    object_id = data.get('objectId')
    
    print(f"Object selected: {object_id}")
    
    return jsonify({
        'status': 'success',
        'message': f'Object {object_id} selected',
        'objectId': object_id
    })

if __name__ == '__main__':

    print(f"Base dir: {BASE_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Web app path: {WEB_APP_PATH}")
    print(f"Web app exists: {os.path.exists(WEB_APP_PATH)}")
    app.run(debug=True, port=4000)