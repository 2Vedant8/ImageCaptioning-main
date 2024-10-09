from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class HuggingFaceCaptioningModel:
    def __init__(self):
        # Load pre-trained BLIP processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    def generate_caption(self, image_path, conditional_text=None):
        # Load and process image
        raw_image = Image.open(image_path).convert('RGB')

        if conditional_text:
            # Conditional image captioning
            inputs = self.processor(raw_image, conditional_text, return_tensors="pt")
        else:
            # Unconditional image captioning
            inputs = self.processor(raw_image, return_tensors="pt")

        # Generate caption
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption
