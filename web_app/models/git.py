from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from web_app.models.git_classes import ImageCaptionDataset


class GitModelExtractor():

    def __init__(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.ckpt = torch.load("web_app/ckpt/git.ckpt", map_location=device)
        
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
        self.model.load_state_dict(self.ckpt["model_state_dict"])

        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.test_dataset = ImageCaptionDataset("web_app/static/images/test", "web_app/static/images/test/captions_test.txt", self.processor)

    
    def generate_caption(self, idx):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        encoding, image = self.test_dataset[idx]

        # Make sure encoding is moved to the right device
        encoding = {k: v.unsqueeze(0).to(device) for k, v in encoding.items() if k in ['input_ids', 'pixel_values']}
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(**encoding, max_length=513)
        
        # Decode generated caption
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return caption



