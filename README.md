# GPT Implementation from Scratch

This repository presents a clean and straightforward implementation of GPT (Generative Pre-trained Transformer) from scratch. We make use of the transformer architecture and train it on the "Tiny Shakespeare" dataset.

### Requirements:

* PyTorch
* tokenizers
* tqdm

### Structure:

* `gpt.ipynb`: This is the main notebook where the GPT model is trained and evaluated.
* `transformer.py`: Contains the transformer model's implementation including multi-head attention, feed-forward layers, positional encodings, and the encoder-decoder layers.

### How the code works:

1. **Preprocessing** :

* We load the data from `input.txt` and tokenize it using Byte Pair Encoding (BPE) tokenizer.
* Tokens are then converted to numerical indices, creating our dataset.

1. **Model Definition** :

* A transformer architecture is defined in `transformer.py`.
* This architecture consists of the standard components like MultiHeadAttention, PositionwiseFeedForward, and PositionalEncoding among others.

1. **Training Loop** :

* The data is divided into batches and fed into the transformer.
* We utilize the Adam optimizer and CrossEntropyLoss criterion for training.
* Learning rate scheduling with warmup is also implemented.

1. **Text Generation** :

* After training, the model is used to generate new text based on a given starting point.

### How to use:

1. Make sure you have all the required libraries installed.
2. Run the `gpt.ipynb` notebook to train and evaluate the GPT model.
3. The model will train on the provided data and generate text in the end.

### Notes:

* Make sure to adjust the hyperparameters as per your computational resources and dataset size.
* For best results, consider using a GPU for training.

---

Thank you for checking out this repository. Contributions are welcome!

---

Made with ❤️ by Bratet
