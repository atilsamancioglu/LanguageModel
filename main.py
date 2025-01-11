import tensorflow as tf
from model import SimpleTransformer
import numpy as np
import PyPDF2
import re
import pickle
import os

def read_pdf(pdf_path):
    """
    Extracts and processes text from a PDF file.
    
    How it works:
    1. Opens a PDF file using PyPDF2 library
    2. Reads through each page of the PDF
    3. Extracts raw text content
    4. Cleans the text by:
       - Removing extra whitespace
       - Removing special characters
       - Standardizing spacing
    
    Args:
        pdf_path (str): Path to the PDF file to read
        
    Returns:
        str: Cleaned text content from the PDF
    """
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
    """
    Converts raw text into a format suitable for the language model.
    
    How it works:
    1. Creates a tokenizer that will:
       - Convert words to numbers (e.g., "whale" -> 42)
       - Handle unknown words with <OOV> (Out Of Vocabulary) token
       - Limit vocabulary to most common words
    2. Processes the text by:
       - Converting to lowercase
       - Splitting into individual words
       - Creating a dictionary of word->number mappings
       - Converting the text into sequences of numbers
    
    Args:
        text (str): Raw text to process
        vocab_size (int): Maximum number of unique words to keep
        
    Returns:
        tuple: (sequences of numbers, tokenizer object)
    """
    # Add special tokens
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    # Split text into words
    words = text.lower().split()
    # Fit tokenizer on words
    tokenizer.fit_on_texts([' '.join(words)])
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences([' '.join(words)])[0]
    return sequences, tokenizer

def create_training_data(sequences, seq_length=50):
    """
    Creates input-output pairs for training the language model.
    
    How it works:
    1. Takes a sequence of numbers (representing words)
    2. Creates sliding windows of text, where:
       - Input: Words 1-50
       - Target: Words 2-51
       This teaches the model to predict the next word
    
    Example:
    Text: "the cat sat on mat"
    Becomes:
    Input: "the cat sat on"
    Target: "cat sat on mat"
    
    Args:
        sequences (list): List of numbers representing words
        seq_length (int): Length of each training sequence
        
    Returns:
        tuple: (input sequences, target sequences) as numpy arrays
    """
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
    vocab_size = 10000  # Increased vocabulary size
    seq_length = 50
    d_model = 128      # Increased model dimension
    num_heads = 4      # Increased number of attention heads
    num_layers = 3     # Increased number of transformer layers
    
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
