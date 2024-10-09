import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from PIL import Image
import pyttsx3
import threading  # For parallel TTS
from model import HuggingFaceCaptioningModel

# Function to perform Text-to-Speech in a separate thread
def speak_caption(caption):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speech rate
    engine.setProperty('volume', 0.9)  # Set volume (0.0 to 1.0)
    engine.say(caption)
    engine.runAndWait()

# Initialize Hugging Face captioning model
captioning_model = HuggingFaceCaptioningModel()

# Image for testing
filename = 'data/f8k/images/image.png'

# Generate caption using Hugging Face model
generated_caption = captioning_model.generate_caption(filename)

# Start speaking the generated caption in a separate thread
thread = threading.Thread(target=speak_caption, args=(generated_caption,))
thread.start()

# Ground truth caption
ground_truth_caption = "two kid with watermelon helmet sitting in a basket with wheels"

# Tokenize the generated and ground truth captions
generated_tokens = generated_caption.lower().split()
ground_truth_tokens = ground_truth_caption.lower().split()

# Calculate precision and accuracy
min_length = min(len(generated_tokens), len(ground_truth_tokens))
generated_tokens = generated_tokens[:min_length]
ground_truth_tokens = ground_truth_tokens[:min_length]

precision = precision_score(ground_truth_tokens, generated_tokens, average='micro', zero_division=1)
accuracy = accuracy_score(ground_truth_tokens, generated_tokens)

# Calculate BLEU score (unigram)
bleu_1 = sentence_bleu([ground_truth_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)

# Calculate ROUGE-L score
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_scores = scorer.score(ground_truth_caption, generated_caption)
rouge_l = rouge_scores['rougeL'].fmeasure

# Display results
print(f"Image: {filename}")
print(f"Generated Caption: {generated_caption}")
print(f"Ground Truth Caption: {ground_truth_caption}")
print(f"BLEU-1 Score (Unigram): {bleu_1}")
print(f"ROUGE-L Score: {rouge_l}")

# Show image
image = Image.open(filename)
plt.imshow(np.asarray(image))
plt.axis('off')  # Hide axis for a cleaner image display
plt.show()
print("Audio caption has been played.")
