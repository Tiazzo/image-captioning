from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from web_app.models.git_classes import ImageCaptionDataset


class GitModelExtractor():

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = torch.load("web_app/ckpt/git.ckpt", map_location=device)
        
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
        self.model.load_state_dict(self.ckpt["model_state_dict"])

        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.test_dataset = ImageCaptionDataset("web_app/static/images/test", "web_app/static/images/test/captions_test.txt", self.processor)

    
    def generate_caption(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoding, image = self.test_dataset[idx]

        inputs = self.processor(images=image, return_tensors='pt').to(device)
        pixel_values = inputs.pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_caption



