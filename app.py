from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import unicodedata
import pyidaungsu as pds
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('hate_speech_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    word2vec_model = pickle.load(f)

# Preprocessing function
def preprocess_text(text, word2vec_model, max_length=50, embedding_dim=300):
    text = unicodedata.normalize('NFKC', text)
    segments = text.split('_')
    tokens = [token for seg in segments if seg for token in pds.tokenize(seg)]
    if hasattr(word2vec_model, 'wv'):
        word_vectors = [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(embedding_dim) 
                        for word in tokens]
    else:
        word_vectors = [word2vec_model[word] if word in word2vec_model else np.zeros(embedding_dim) 
                        for word in tokens]
    if not word_vectors:
        word_vectors = [np.zeros(embedding_dim)]
    padded = pad_sequences([word_vectors], maxlen=max_length, padding='post', dtype='float32')
    return padded

# Prediction function
def predict_text(text):
    processed_text = preprocess_text(text, word2vec_model, max_length=50, embedding_dim=300)
    prediction = model.predict(processed_text, verbose=0)
    return prediction[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    prediction_score = predict_text(text)
    predicted_class = "Hate Speech" if prediction_score > 0.5 else "Normal Speech"
    
    return jsonify({
        'prediction_score': float(prediction_score),
        'predicted_class': predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
