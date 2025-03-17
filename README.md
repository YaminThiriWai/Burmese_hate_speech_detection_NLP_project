# Burmese Hate Speech Detection

## Description
This project aims to detect hate speech in Burmese text using a deep learning model. It utilizes TensorFlow and a Convolutional Neural Network (CNN) for text classification. The model predicts whether a given Burmese text contains hate speech or is of normal speech.

## Dataset
The dataset used in this project is provided by [Simbolo](https://huggingface.co/datasets/simbolo-ai/encrypted-burmese-hate-speech), specifically designed to train models for detecting hate speech in Burmese language texts.

## Files in the Repository
- `app.py`: Flask application to deploy the model as a web service.
- `train_model.py`: Python script used to train the hate speech detection model.
- `dataset_prep.ipynb`: Jupyter notebook that demonstrates how the dataset was prepared and preprocessed.
- `hate_speech_model.h5`: The trained model saved in HDF5 format.
- `tokenizer.pkl`: The tokenizer used to convert text to sequences, saved as a pickle file.
- `index.html`: Frontend interface for interacting with the model via a web browser.
- `requirements.txt`: A list of Python packages required to run the project.

## Installation
Clone this repository to your local machine and install the required dependencies:
```bash
git clone https://github.com/YaminThiriWai/Burmese_hate_speech_detection_NLP_project.git
cd Burmese_hate_speech_detection_NLP_project
pip install -r requirements.txt
```

## Running the Application
To start the Flask server, execute:
```bash
python app.py
```
This command starts the server on localhost, making the web application accessible via `http://localhost:8000`.

## Usage
1. Open a web browser and go to `http://localhost:8000`.
2. Use the text box provided on the web page to input Burmese text.
3. Click the "Predict" button to classify the text and see if it is classified as hate speech or normal speech.


## Credits
- Dataset and initial preprocessing tools provided by [Simbolo](https://huggingface.co/datasets/simbolo-ai/encrypted-burmese-hate-speech).
- Tokenization potentially supported by `cc.my.300.bin.gz`, depending on project configuration.
