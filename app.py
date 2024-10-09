import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from model import HuggingFaceCaptioningModel
import pyttsx3
import threading
import sacrebleu
from rouge_score import rouge_scorer
import nltk  # Keep this for METEOR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize the captioning model
captioning_model = HuggingFaceCaptioningModel()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# Function to perform TTS and save as mp3
def speak_caption(caption, audio_file_path):
    tts_engine.save_to_file(caption, audio_file_path)
    tts_engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    ground_truth_caption = request.form['ground_truth']
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate the caption
        generated_caption = captioning_model.generate_caption(filepath)
        
        # Save the audio caption
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'caption_audio.mp3')
        thread = threading.Thread(target=speak_caption, args=(generated_caption, audio_path))
        thread.start()
        thread.join()

        # Tokenize captions
        generated_tokens = generated_caption.lower().split()
        ground_truth_tokens = ground_truth_caption.lower().split()

        # Calculate BLEU score
        bleu_1 = sacrebleu.sentence_bleu(generated_caption, [ground_truth_caption]).score

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(ground_truth_caption, generated_caption)
        rouge_l = rouge_scores['rougeL'].fmeasure

        # Remove METEOR calculation, as it's no longer included

        return render_template('result.html', 
                               filename=filename, 
                               generated_caption=generated_caption,
                               audio_url=url_for('static', filename='uploads/caption_audio.mp3'),
                               bleu_1=bleu_1,
                               rouge_l=rouge_l)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/history')
def upload_history():
    # Placeholder for upload history logic
    return render_template('upload_history.html')

if __name__ == '__main__':
    app.run(debug=True)
