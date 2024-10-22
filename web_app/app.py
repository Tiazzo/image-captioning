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
        caption_resnet_rnn, img_name, refs = rnnModelExtractor.generate_caption(idx)
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


        captions = [caption_resnet_rnn, caption_resnet_rnn_att, caption_resnet_vit, caption_git]

        scores = compute_scores(captions, refs)
        print(scores)
        return render_template(
            "index.html",
            image_name=img_name,
            captions=captions,
            scores=scores
        )

    # For GET requests, just render the page without captions
    return render_template("index.html", captions=['', '', '', ''])

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
smoothing_function = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_scores(captions: list, reference_captions: list):
    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores = [], [], [], []
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference_captions]

    print(len(captions))
    for cap in captions:

        generated_tokens = nltk.word_tokenize(cap.lower())
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)

        rouge_scores = [rouge.score(ref, cap) for ref in reference_captions]
        best_rouge1 = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
        best_rouge2 = np.mean([score["rouge2"].fmeasure for score in rouge_scores])
        best_rougeL = np.mean([score["rougeL"].fmeasure for score in rouge_scores])

        rouge1_scores.append(best_rouge1)
        rouge2_scores.append(best_rouge2)
        rougeL_scores.append(best_rougeL)

    return bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores

if __name__ == "__main__":
    app.run(debug=True)
