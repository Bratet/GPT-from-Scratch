# GPT-from-scratch

This repository contains a basic implementation of the GPT (Generative Pre-trained Transformer) architecture from scratch using PyTorch. The repository contains both the model implementation and a Jupyter notebook to showcase training and generating text using the implemented model.

### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Repository Structure](#repository-structure)
4. [License](#license)

### Installation

To use this implementation, ensure you have the following dependencies:

- PyTorch
- tqdm
- tokenizers

Install the required packages with the following command:

```bash
pip install torch tqdm tokenizers
```

### Usage

1. Prepare your training data and save it as `input.txt`.
2. Run the `gpt.ipynb` notebook, which contains the entire training pipeline.
3. The notebook tokenizes the input text, initializes the GPT model, and starts the training process.
4. After training, you can generate text using the trained model.

### Repository Structure

- `gpt.ipynb`: Jupyter notebook that contains the main code for loading data, training the GPT model, and generating text.
- `transformer.py`: Contains the implementation of the transformer architecture including MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, DecoderLayer, and the main Transformer model.

### Code Overview

- `gpt.ipynb`:

  - Loads and tokenizes data using the Byte-Pair Encoding (BPE) tokenizer.
  - Initializes the GPT model with specified hyperparameters.
  - Trains the model using a training and validation split.
  - Generates text using the trained model.
- `transformer.py`:

  - Contains the implementation of the various components of the Transformer architecture.
  - `MultiHeadAttention`: Implements the multi-head self-attention mechanism.
  - `PositionwiseFeedForward`: Implements the position-wise feed-forward networks.
  - `EncoderLayer` & `DecoderLayer`: Basic blocks of the encoder and decoder stacks.
  - `PositionalEncoding`: Adds the positional encodings to the input embeddings.
  - `Transformer`: Main model which combines the above components to create the Transformer architecture.

### License

This project is open-source and available to everyone. Before using or contributing to the project, make sure to familiarize yourself with the [LICENSE](./LICENSE) file.

---

Feel free to contribute to this project by raising issues or submitting pull requests. Feedback and contributions are always welcome.
