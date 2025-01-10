import tensorflow as tf
from model import SimpleTransformer
import numpy as np
import PyPDF2
import re

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
    # Read PDF file (replace with your PDF path)
    pdf_path = "path/to/your/book.pdf"
    try:
        text = read_pdf(pdf_path)
        print(f"Successfully loaded PDF. Text length: {len(text)} characters")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return
    
    # Model parameters
    vocab_size = 5000  # Increased for better vocabulary coverage
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
    
    # Train model
    model.fit(
        x_train, 
        y_train, 
        epochs=10, 
        batch_size=32,
        validation_split=0.1
    )

if __name__ == "__main__":
    main()
