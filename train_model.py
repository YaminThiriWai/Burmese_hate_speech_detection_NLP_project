import pandas as pd
import unicodedata
import pyidaungsu as pds
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load dataset
data = pd.read_csv("/Users/mac/Desktop/NLP Project/combined_dataset.csv")
texts = data['text'].tolist()
labels = data['label'].tolist()

# Print label distribution
print(pd.Series(labels).value_counts())

# Text cleaning and tokenization
tokenized_texts = []
for text in texts:
    text = unicodedata.normalize('NFKC', text)
    segments = text.split('_')
    tokens = [token for seg in segments if seg for token in pds.tokenize(seg)]
    tokenized_texts.append(' '.join(tokens))

# Verify tokenized output
print("First 5 tokenized texts:", tokenized_texts[:5])

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(tokenized_texts, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train Word2Vec model
tokenized_sentences = [text.split() for text in tokenized_texts]
embedding_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save the Word2Vec model as tokenizer.pkl
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(embedding_model, f)
print("Tokenizer saved as tokenizer.pkl")

# Convert texts to embeddings
embedding_dim = 100  
X_train_embedded = [[embedding_model.wv[word] for word in text.split() if word in embedding_model.wv] for text in X_train]
X_val_embedded = [[embedding_model.wv[word] for word in text.split() if word in embedding_model.wv] for text in X_val]
X_test_embedded = [[embedding_model.wv[word] for word in text.split() if word in embedding_model.wv] for text in X_test]

# Pad sequences
max_length = 50
X_train_padded = pad_sequences(X_train_embedded, maxlen=max_length, padding='post', dtype='float32')
X_val_padded = pad_sequences(X_val_embedded, maxlen=max_length, padding='post', dtype='float32')
X_test_padded = pad_sequences(X_test_embedded, maxlen=max_length, padding='post', dtype='float32')

# model
model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(max_length, embedding_dim)))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train_padded, np.array(y_train),
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val_padded, np.array(y_val)),
                    callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, np.array(y_test))
print(f"Test Accuracy: {accuracy:.4f}")

# Save the trained model
model.save('hate_speech_model.h5')
print("Model saved as hate_speech_model.h5")

#Predict on a sample text
def preprocess_text(text, word2vec_model, max_length=50, embedding_dim=100):
    text = unicodedata.normalize('NFKC', text)
    segments = text.split('_')
    tokens = [token for seg in segments if seg for token in pds.tokenize(seg)]
    word_vectors = [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(embedding_dim) 
                    for word in tokens]
    if not word_vectors:
        word_vectors = [np.zeros(embedding_dim)]
    padded = pad_sequences([word_vectors], maxlen=max_length, padding='post', dtype='float32')
    return padded

test_text = "လီးသံကြီးနဲ့မအိမ်လုံး"
processed_text = preprocess_text(test_text, embedding_model, max_length, embedding_dim)
prediction = model.predict(processed_text)
print("Prediction Score:", prediction)
predicted_class = (prediction > 0.7).astype("int32")
print("Predicted Class:", "Hate Speech" if predicted_class[0][0] == 1 else "Normal Speech")