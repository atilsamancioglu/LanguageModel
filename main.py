import tensorflow as tf
from model import SimpleTransformer
import numpy as np
import PyPDF2
import re
import pickle
import os

def read_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special characters
    return text.strip()

# Sample text preprocessing
def preprocess_text(text, vocab_size=1000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, oov_token="<OOV>"
    )
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])[0]
    return sequences, tokenizer

def create_training_data(sequences, seq_length=50):
    input_sequences = []
    for i in range(len(sequences) - seq_length):
        input_sequences.append(sequences[i:i + seq_length + 1])
    
    x = np.array([seq[:-1] for seq in input_sequences])
    y = np.array([seq[1:] for seq in input_sequences])
    return x, y

def main():
    # Create a directory for saving models if it doesn't exist
    save_dir = "saved_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Read PDF file
    pdf_path = "mobydickpdf.pdf"
    try:
        text = read_pdf(pdf_path)
        print(f"Successfully loaded PDF. Text length: {len(text)} characters")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return
    
    # Model parameters
    vocab_size = 5000
    seq_length = 50
    d_model = 64
    num_heads = 2
    num_layers = 2
    
    # Preprocess data
    sequences, tokenizer = preprocess_text(text, vocab_size)
    x_train, y_train = create_training_data(sequences, seq_length)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    # Create and compile model
    model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers)
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Add ModelCheckpoint callback to save the best model during training
    checkpoint_path = os.path.join(save_dir, "model_checkpoint.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    # Train model
    history = model.fit(
        x_train, 
        y_train, 
        epochs=10, 
        batch_size=32,
        validation_split=0.1,
        callbacks=[checkpoint_callback]
    )

    # Save the final model
    final_model_path = os.path.join(save_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Save the tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer.pickle")
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Save training history
    history_path = os.path.join(save_dir, "training_history.pickle")
    with open(history_path, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()
