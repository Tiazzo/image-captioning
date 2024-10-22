import os
import random
import torch
from flask import Flask, render_template, request
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from web_app.models.resnet_rnn import ModelExtractorRnn
from web_app.models.resnet_rnn_att import ModelExtractorRnnAtt
from web_app.models.resnet_vit_rnn import ModelExtactorVitRnn
from web_app.models.git import GitModelExtractor


app = Flask(__name__)
rnnModelExtractor = ModelExtractorRnn()
rnnAttModelExtractor = ModelExtractorRnnAtt()
vitRnnModelExtractor = ModelExtactorVitRnn()
GitModelExtractor = GitModelExtractor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Single route for displaying and generating captions
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # Retrieve the index of the selected image
        idx = random.randint(0, 4000)

        # generate caption
        caption_resnet_rnn, img_name = rnnModelExtractor.generate_caption(idx)
        try: caption_resnet_rnn = caption_resnet_rnn.replace('.', '')
        except: pass

        caption_resnet_rnn_att, _ = rnnAttModelExtractor.generate_caption(idx)
        try: caption_resnet_rnn_att = caption_resnet_rnn_att.replace('.', '')
        except: pass

        caption_resnet_vit, _ = vitRnnModelExtractor.generate_caption(idx)
        try: caption_resnet_vit = caption_resnet_vit.replace('.', '')
        except: pass

        caption_git = GitModelExtractor.generate_caption(idx)
        try: caption_git = caption_git.replace('.', '')
        except: pass

        # Render the index page with the image and the generated captions
        return render_template(
            "index.html",
            image_name=img_name,
            caption_rnn=caption_resnet_rnn,
            caption_rnn_att=caption_resnet_rnn_att,
            caption_vit=caption_resnet_vit,
            caption_git=caption_git
        )

    # For GET requests, just render the page without captions
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
