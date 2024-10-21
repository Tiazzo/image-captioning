import os
import random
import torch
from flask import Flask, render_template, request
from models.resnet_rnn import retrieve_model as retrieve_model_resnet_rnn
from PIL import Image
import random


app = Flask(__name__)

# Path to your image dataset (inside web_app/static/images now)
image_folder = "web_app/static/images/train"  # Relative to the web_app folder
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

# Single route for displaying and generating captions
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Randomly pick an image from the dataset
        image_path = random.choice(image_paths)
        image_name = os.path.basename(image_path)

        # Retrieve the index of the selected image
        idx = random.randint(0,4000)        
        
        # Call the retrieve_model function to generate a caption
        caption_resnet_rnn = retrieve_model_resnet_rnn(idx)  # Generate caption from model
        
        # Render the index page with the image and the generated caption
        return render_template("index.html", image_name=image_name, caption=caption_resnet_rnn)


    # For GET requests, just render the page without captions
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
