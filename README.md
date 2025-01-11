# Simple Language Model Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Components in Detail](#components-in-detail)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

## Introduction

This project creates a simple language model that can learn from text and generate new content. It's built using modern deep learning techniques, specifically a simplified version of the Transformer architecture (similar to what powers ChatGPT, but on a smaller scale).

### Key Features
- Learns from PDF documents
- Generates text based on learned patterns
- Uses Transformer architecture
- Configurable text generation parameters
- Suitable for educational purposes

## Installation

1. Clone the repository:

```bash
git clone https://github.com/atilsamancioglu/LanguageModel.git
cd LanguageModel
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

Required dependencies:
- tensorflow==2.16.1
- numpy>=1.24.0
- PyPDF2>=3.0.0

## Project Structure
```
project/
├── main.py # Training script
├── model.py # Model architecture
├── predict.py # Text generation
├── documentation.md # This documentation
└── requirements.txt # Dependencies
```

## How It Works

### The Process Flow

1. **Data Loading**: 
   - Reads text from a PDF file
   - Extracts and cleans the content
   - Prepares the text for processing

2. **Preprocessing**:
   - Converts text to lowercase
   - Tokenizes text into words
   - Creates a vocabulary dictionary
   - Converts words to numerical sequences

3. **Training**:
   - Splits text into input-output pairs
   - Feeds data through the Transformer model
   - Updates model weights based on predictions
   - Saves the trained model for later use

4. **Generation**:
   - Takes a seed text as input
   - Predicts next words one by one
   - Uses temperature parameter to control creativity
   - Produces human-readable output

### Behind the Scenes

Think of the process like teaching someone a new language:
- First, they need to read examples (PDF reading)
- Then, they learn vocabulary and patterns (preprocessing)
- Finally, they can create their own sentences (generation)

## Components in Detail

### 1. PDF Reading and Text Processing

#### PDF Reader Function

```python
def read_pdf(pdf_path):
    """Extract text from a PDF file."""
```

This function:
- Opens your PDF document
- Extracts text from each page
- Cleans the text by:
  - Removing extra spaces
  - Removing special characters
  - Standardizing formatting
- Returns a clean text string ready for processing

Real-world example:
Input PDF text: "The whale\n\nswam in\tthe ocean!"
Cleaned output: "the whale swam in the ocean"

#### Text Preprocessor

This function:
- Converts all text to lowercase
- Breaks text into individual words
- Creates a vocabulary of most common words
- Converts words to numerical indices
- Handles unknown words with special <OOV> token

Example transformation:
Input text: "The whale swam in the ocean"
Processed: [4, 7, 12] # Numbers representing words in vocabulary

### 2. Model Architecture

#### SimpleTransformer Class
The heart of our language model is the Transformer architecture, which consists of several key components:

##### Embedding Layer
```python
self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
- Converts word indices into rich vector representations
- Each word gets a unique numerical representation
- Example: "whale" → [0.2, -0.5, 0.7, ...] (vector of numbers)
- Helps model understand word relationships
```

##### Positional Encoding
```python
self.pos_encoding = self.positional_encoding(vocab_size, d_model)
- Adds information about word position in sequence
- Helps model understand word order
- Uses mathematical functions (sine and cosine) to encode positions
- Essential for understanding sentence structure
```

##### Transformer Blocks
Each transformer block contains:
1. **Multi-Head Attention**
   - Allows model to focus on different parts of text
   - Multiple attention heads capture different relationships
   - Like reading a sentence while focusing on different aspects

2. **Feed-Forward Network**
   - Processes attended information
   - Two dense layers with ReLU activation
   - Learns complex patterns in the text

3. **Layer Normalization**
   - Stabilizes training
   - Helps model learn more effectively

### 3. Training Process

#### Data Preparation
```python
def create_training_data(sequences, seq_length=50):
What it does:
- Creates sliding windows of text
- Each window is 50 words long
- Input: Words 1-49
- Target: Words 2-50

Example:
Text: "the white whale swam in the ocean"
Input: "the white whale swam"
Target: "white whale swam in"
```

#### Training Configuration
Current settings:
```python
vocab_size = 10000 # Maximum number of unique words
d_model = 128 # Dimension of word embeddings
num_heads = 4 # Number of attention heads
num_layers = 3 # Number of transformer blocks
batch_size = 32 # Samples processed at once
epochs = 10 # Complete passes through data
```

#### Model Training
The training process:
1. **Initialization**
   - Creates model with specified parameters
   - Compiles with Adam optimizer
   - Uses cross-entropy loss function

2. **Training Loop**
   - Processes batches of text
   - Updates model weights
   - Monitors loss and accuracy
   - Saves best model checkpoint

3. **Model Saving**
   - Saves final trained model
   - Saves tokenizer for later use
   - Saves training history

### 4. Text Generation Process

#### Generation Function

```python
def generate_text(model, tokenizer, seed_text, num_words=50, temperature=0.7):


This function handles the text generation process:

1. **Input Processing**
   - Takes a seed text (e.g., "The white whale")
   - Converts to lowercase to match training
   - Tokenizes into numerical sequence

2. **Temperature Control**
   - Temperature parameter controls randomness
   - Lower values (0.2-0.3): More focused, predictable text
   - Higher values (0.5-1.0): More creative, diverse text
   - Helps balance between coherence and creativity

3. **Word Generation Loop**
   - Predicts next word probabilities
   - Applies temperature scaling
   - Samples next word from distribution
   - Adds word to generated sequence
   - Repeats until desired length
```

## Usage Guide

### Training a New Model

1. **Prepare Your Data**
   - Place your PDF file in the project directory
   - Update the pdf_path in main.py:
   ```python
   pdf_path = "your_book.pdf"
   ```

2. **Start Training**
   ```bash
   python main.py
   ```

3. **Monitor Training**
   - Watch for progress updates
   - Check loss and accuracy values
   - Training saves checkpoints automatically

### Generating Text

1. **Basic Usage**
   ```bash
   python predict.py
   ```

2. **Output Example**
   ```
   Generating with temperature 0.3:
   --------------------------------------------------
   Seed: "The white whale"
   Generated: "the white whale moved through the dark waters 
   of the sea while the crew watched in silence..."
   ```

3. **Customizing Generation**
   - Modify seed texts in predict.py
   - Adjust temperature values
   - Change number of words generated

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Symptom: Out of memory during training
   - Solutions:
     ```python
     vocab_size = 5000  # Reduce from 10000
     seq_length = 30    # Reduce from 50
     batch_size = 16    # Reduce from 32
     ```

2. **Poor Generation Quality**
   - Symptom: Nonsensical or repetitive text
   - Solutions:
     - Increase training epochs
     - Adjust temperature
     - Use larger training text
     ```python
     epochs = 20        # Increase from 10
     temperature = 0.3  # Try different values
     ```

3. **PDF Reading Errors**
   - Symptom: "Error loading PDF"
   - Solutions:
     - Check file exists
     - Verify PDF is not corrupted
     - Try different PDF format

## Advanced Topics

### Model Customization

#### Increasing Model Capacity
For better performance on larger texts:
```python
# In main.py
d_model = 256        # Increase from 128
num_heads = 8        # Increase from 4
num_layers = 6       # Increase from 3
vocab_size = 15000   # Increase vocabulary
```

#### Training Optimization
Fine-tune training parameters:
```python
# In main.py
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

### Advanced Generation Settings

#### Custom Temperature Scheduling
Implement dynamic temperature:
```python
def dynamic_temperature(epoch):
    """Adjust temperature based on generation progress"""
    return 0.5 - (epoch * 0.01)  # Gradually decrease temperature
```

#### Beam Search
For more structured text generation:
```python
def beam_search_generate(model, seed, beam_width=3):
    """Generate text using beam search instead of random sampling"""
    candidates = [(seed, 0.0)]
    # Implementation details...
```

### Performance Optimization

1. **GPU Utilization**
   ```python
   # Check GPU availability
   if tf.test.is_gpu_available():
       print("Training on GPU")
   ```

2. **Memory Management**
   ```python
   # Use gradient accumulation
   gradient_accumulation_steps = 4
   ```

3. **Data Pipeline**
   ```python
   # Use tf.data for efficient data loading
   dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
   dataset = dataset.cache().shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
   ```

### Project Extensions

1. **Multiple PDF Support**
```python
def process_multiple_pdfs(pdf_directory):
    """Process all PDFs in a directory"""
    combined_text = ""
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith('.pdf'):
            text = read_pdf(os.path.join(pdf_directory, pdf_file))
            combined_text += text + "\n"
    return combined_text
```

2. **Model Evaluation**
```python
def evaluate_model(model, test_data, metrics=['perplexity', 'accuracy']):
    """Comprehensive model evaluation"""
    results = {}
    # Implementation details...
    return results
```

3. **Interactive Generation**
```python
def interactive_generation():
    """Interactive text generation interface"""
    model, tokenizer = load_model_and_tokenizer()
    while True:
        seed = input("Enter seed text (or 'quit' to exit): ")
        if seed.lower() == 'quit':
            break
        temperature = float(input("Enter temperature (0.2-1.0): "))
        generated = generate_text(model, tokenizer, seed, temperature=temperature)
        print(f"\nGenerated text:\n{generated}\n")
```

## Future Improvements

1. **Model Architecture**
   - Add attention visualization
   - Implement different attention mechanisms
   - Add dropout layers for better regularization

2. **Training Process**
   - Implement learning rate scheduling
   - Add early stopping
   - Implement cross-validation

3. **Text Generation**
   - Add diverse decoding strategies
   - Implement context conditioning
   - Add generation constraints

---
