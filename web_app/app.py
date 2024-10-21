import os
import random
import torch
from flask import Flask, render_template, request
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from web_app.models.resnet_rnn import ModelExtractor
from PIL import Image
import random


app = Flask(__name__)
rnnModelExtractor = ModelExtractor()

# Path to your image dataset (inside web_app/static/images now)
# image_folder = "web_app/static/images/test"  # Relative to the web_app folder
# image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

# Single route for displaying and generating captions
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # Retrieve the index of the selected image
        idx = random.randint(0,4000)        
        
        # generate caption
        caption_resnet_rnn, img_name = rnnModelExtractor.retrieve_model(idx)
        print(img_name)
        
        # Render the index page with the image and the generated caption
        return render_template("index.html", image_name=img_name, caption=caption_resnet_rnn)


    # For GET requests, just render the page without captions
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
