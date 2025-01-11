import tensorflow as tf
import pickle
import os
import numpy as np

def load_model_and_tokenizer():
    save_dir = "saved_model"
    
    # Load the model
    model_path = os.path.join(save_dir, "final_model.keras")
    model = tf.keras.models.load_model(model_path)
    
    # Load the tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer.pickle")
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer

def generate_text(model, tokenizer, seed_text, num_words=50, temperature=0.7):
    """
    Generates new text based on a seed phrase.
    
    How it works:
    1. Takes a starting phrase (seed text)
    2. For each new word to generate:
       - Converts current text to numbers using tokenizer
       - Feeds numbers into model to get predictions
       - Applies temperature to control randomness:
         * Low temperature (0.2-0.3): More focused, repetitive text
         * High temperature (0.7-1.0): More creative, diverse text
       - Randomly selects next word based on predictions
       - Adds new word to generated text
       - Repeats until desired length
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer object for converting words to/from numbers
        seed_text (str): Starting phrase for generation
        num_words (int): Number of words to generate
        temperature (float): Controls randomness (0.2-1.0)
        
    Returns:
        str: Generated text including seed text
    """
    # Convert seed text to lowercase to match training data
    seed_text = seed_text.lower()
    
    # Convert seed text to sequence
    current_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    generated_text = seed_text
    
    # Create reverse word index for converting numbers back to words
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    for _ in range(num_words):
        # Prepare the sequence for prediction
        padded_sequence = current_sequence[-50:]  # Use the last 50 tokens
        if len(padded_sequence) < 50:
            padded_sequence = [0] * (50 - len(padded_sequence)) + padded_sequence
            
        # Convert to numpy array and reshape for model input
        padded_sequence = np.array(padded_sequence)[np.newaxis, :]
            
        # Get model's predictions for next word
        predictions = model.predict(padded_sequence, verbose=0)[0]
        
        # Get the last prediction (for the next word)
        next_word_logits = predictions[-1]
        
        # Apply temperature scaling to control randomness
        scaled_logits = next_word_logits / temperature
        # Subtract max for numerical stability
        scaled_logits = scaled_logits - np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits)
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Ensure valid probability distribution
        probabilities = np.clip(probabilities, 1e-7, 1.0)
        probabilities = probabilities / np.sum(probabilities)
        
        try:
            # Sample from the distribution to get next word
            predicted_id = np.random.choice(len(probabilities), p=probabilities)
            
            # Skip if predicted token is OOV or padding
            if predicted_id == 0 or predicted_id == tokenizer.word_index.get('<OOV>', 0):
                continue
                
            # Convert number back to word and add to generated text
            predicted_word = reverse_word_index.get(predicted_id, '')
            if predicted_word:
                generated_text += ' ' + predicted_word
                current_sequence.append(predicted_id)
                
        except ValueError as e:
            print(f"Error in sampling: {e}")
            continue
    
    return generated_text

if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Generate some text with different seeds
    seed_texts = [
        "The white whale",
        "Call me Ishmael",
        "The sea was",
    ]
    
    # Try different temperatures
    temperatures = [0.2, 0.3, 0.5]
    
    for temp in temperatures:
        print(f"\nGenerating with temperature {temp}:")
        print("-" * 50)
        for seed_text in seed_texts:
            print(f"\nSeed: {seed_text}")
            generated_text = generate_text(
                model, 
                tokenizer, 
                seed_text, 
                num_words=30, 
                temperature=temp
            )
            print(f"Generated: {generated_text}\n")
            print("-" * 50) 