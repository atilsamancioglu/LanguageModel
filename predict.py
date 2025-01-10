import tensorflow as tf
import pickle
import os

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

def generate_text(model, tokenizer, seed_text, num_words=50):
    # Convert seed text to sequence
    current_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    generated_text = seed_text
    
    for _ in range(num_words):
        # Prepare the sequence for prediction
        padded_sequence = current_sequence[-50:]  # Use the last 50 tokens
        if len(padded_sequence) < 50:
            padded_sequence = [0] * (50 - len(padded_sequence)) + padded_sequence
            
        # Predict next word
        predictions = model.predict([padded_sequence], verbose=0)[0]
        predicted_id = tf.argmax(predictions[-1]).numpy()
        
        # Convert the predicted ID to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                generated_text += " " + word
                current_sequence.append(predicted_id)
                break
    
    return generated_text

if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Generate some text
    seed_text = "The white whale"
    generated_text = generate_text(model, tokenizer, seed_text)
    print(f"Generated text:\n{generated_text}") 