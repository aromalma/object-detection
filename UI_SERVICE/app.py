
from flask import Flask, request, render_template,  Response,jsonify,send_file,render_template_string
import requests
import numpy as np
import os


app = Flask(__name__)
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://mlservice:8011")
@app.route('/')
def upload_file():
    """
    first page for uploading
    """
    # upload page
    return render_template('upload.html')
    
@app.route('/upload', methods=['POST']) 
def detect():
    """
    call the mlservice
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Get the uploaded file
    image_file = request.files['image']

    # Prepare the files payload for forwarding to mlservr
    files = {'image': (image_file.filename, image_file.stream, image_file.mimetype)}

    try:
        # Forward the request to the ml server
        response = requests.post(ML_SERVICE_URL, files=files)

        html_template = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Processed Image</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 600px;
                    margin: 50px auto;
                    text-align: center;
                    color: #333;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ccc;
                }
                h1 {
                    margin-top: 20px;
                    font-size: 1.5em;
                }
            </style>
        </head>
        <body>
            <img src="data:image/png;base64,{{ image }}" alt="Processed Image">
            <h1>{{ text }}</h1>
        </body>
        </html>
        '''
     
        resp = response.json()
        text = resp["json"]
        image = resp["image"]

        #Dispaly results
        return render_template_string(html_template, text=text, image=image)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to connect to target server", "details": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0",port =8010,debug=True)

