# Project Overview

## Project Summary

Based on the names of the files provided, the overall purpose of this project appears to be centered around the development and experimentation of various language models, particularly focusing on deep learning techniques to process and understand natural language data.

1. **gru_model, rnn_model, lstm_model, transformer_model** - These files suggest implementations of different neural network architectures for language modeling. GRU (Gated Recurrent Unit), LSTM (Long Short-Term Memory), and RNN (Recurrent Neural Network) are all types of recurrent neural networks that are effective in handling sequences, like text. The Transformer model file indicates the use of a more recent and powerful model architecture that relies on self-attention mechanisms.

2. **train_utils** - This file likely contains utility functions or classes that aid in training the various models. This might include functions to set up training loops, calculate loss, update model weights, or handle checkpoints and evaluations.

3. **tokenizer** - Tokenization is a fundamental step in preparing text data for model training, transforming raw text into a format that can be processed by neural networks. This file likely contains code to perform this tokenization, possibly implementing or adapting existing tokenizer models.

4. **dataset_loader** - This file is expected to contain code for loading and possibly preprocessing datasets that are used to train and test the models. Handling large datasets efficiently is crucial for training language models, so this component is key.

5. **llm_project, llm_experiment** - These files suggest a focus specifically on large language models (LLM), which are a subset of models capable of understanding and generating human-like text based on the training data they’ve been fed. The project file might define the scope and objectives of working with LLMs, while the experiment file could involve testing hypotheses, configurations, or novel approaches specific to these models.

Overall, the project seems to be aimed at exploring various neural network technologies to advance the field of natural language processing, with a particular focus on implementing, training, and experimenting with different architectures and techniques to optimize performance and possibly innovate in the area of large language models.

## File Contributions

- gru_model.py: The `model.py` file defines a class `GRULanguageModel` that uses a Gated Recurrent Unit (GRU) neural network for natural language processing tasks, particularly aimed at text generation. The model includes methods for training, predicting the next token, and generating text sequences.

**Key Components of `GRULanguageModel`**:
1. **Embedding Layer**: Maps vocabulary indices into embedding vectors and handles padding.
2. **GRU Layer**: Processes the sequence data, capable of using multiple layers and handling dropout for regularization.
3. **Dropout Layer**: Applied after the GRU to prevent overfitting.
4. **Fully Connected Layer**: Maps the output of the GRU to the vocabulary size for prediction.

**Methods**:
- **`forward`**: Takes input tokens and an optional hidden state to produce logits for each token and an updated hidden state.
- **`predict_next_token`**: Generates the next token in a sequence by providing the model with the current state and the last token input, using a temperature parameter to control randomness in prediction.
- **`generate`**: Generates a sequence of text starting from a given prompt. It supports different modes such as generating a fixed maximum length sequence and returning either the full generated sequence or just the continuation beyond the prompt.

Overall, this model can be used for tasks where generating natural language text or predicting sequential data is required. The architecture is structured in a way that it can be trained on a large corpus of text and then used to generate text sequences similar to the training data.
- train_utils.py: This file defines Python functions for training and evaluating a machine learning model, specifically a neural network using the PyTorch framework. 

1. **train_model function:** 
   - It takes parameters such as the model, data loaders for training and validation, tokenizer, device, and others to set up and run the training process. 
   - The function initializes the optimizer (`AdamW`) and learning rate scheduler, sets a criterion for loss calculation, and then loops over the specified number of epochs to train the model on the training data.
   - During each epoch, the model's training and validation losses are calculated and printed. The model state is saved whenever a new best validation loss is achieved.
   - Losses across all epochs are plotted and saved using the `plot_losses` function.

2. **plot_losses function:**
   - Used to visually depict the training and validation loss per epoch using matplotlib. It saves the plot as an image file using a timestamp in its filename.

3. **evaluate_model function:**
   - It loads the best-performing model state, evaluates it on a test dataset, and computes the loss.
   - Additionally, it calculates the BLEU score (a metric for evaluating text which compares n-grams of the candidate with n-grams of the reference data) to assess the linguistic quality of model outputs as compared to actual target sequences.

The script also imports necessary libraries for its operations, including PyTorch for modeling and computation, tqdm for progress bars during training loops, and NLTK for calculating BLEU scores.
- transformer_model.py: The provided Python file defines a language model based on the Transformer architecture, specifically for the task of next-word prediction. Here's a summary of its components and functionalities:

1. **Initialization (`__init__`)**: The `TransformerLanguageModel` class initializes with parameters for the vocabulary size, embedding dimensions, number of attention heads, number of Transformer layers, and padding token ID. It also sets up positional embeddings for sequences up to a defined maximum length and applies dropout for regularization.

2. **Forward Pass (`forward`)**: This method processes input tokens through embeddings and the Transformer encoder. It applies a sequence-dependent mask (for causal attention) and a padding mask to handle varying input lengths and pad tokens respectively, while also combining token and position embeddings with dropout. The output is then passed through a linear layer to transform encoder output into vocabulary-sized logits for each position in the input sequence.

3. **Helper Method (`_generate_square_subsequent_mask`)**: Generates a triangular (upper triangular filled with `True` values starting from the diagonal) mask to prevent future positions from influencing the prediction at the current position, used for imposing the autoregressive property in the sequence generation.

4. **Prediction Method (`predict_next_token`)**: This methods performs a forward pass through the model to obtain logits for the last token position, which are then scaled by a temperature parameter, and processed using softmax to get probabilities. The next token is sampled from these probabilities, ensuring that the beginning-of-sequence token is not selected.

5. **Text Generation (`generate`)**: This function takes a text prompt, tokenizes it, and then uses the model to autoregressively generate text by repeatedly predicting the next token until a maximum length or the end-of-sequence token is reached. The generated text can either include the prompt or just the continuation, according to the `return_continuation_only` parameter.

Overall, the file implements a Transformer-based model suitable for generating text or predicting the next token in a sequence, equipped with functionalities to handle input sequences and offer user-friendly interfaces for generating text given a prompt.
- llm_project.py: This script performs natural language processing tasks using various machine learning models built with PyTorch. Here’s a summary of its main functionalities:

1. **Data Download and Tokenizer Training**: The script first downloads training and testing datasets from specified URLs, and merges additional text data to form a corpus. It then trains a tokenizer using the combined text corpus to process the text data.

2. **Dataset Preparation**: Using the trained tokenizer, it transforms train and test data into a format suitable for model training and evaluation by tokenizing and encoding the texts into sequences.

3. **Model Building**: It supports the selection and construction of different types of neural network models for language processing, including GRU, LSTM, RNN, and Transformer models. You can select which model to use via a configuration.

4. **Training and Evaluation**: It trains the chosen model on the processed training dataset using specific hyperparameters, saves the best-performing version of the model, and then evaluates it on the testing dataset.

5. **Text Generation**: Finally, the script uses the trained model to generate text based on custom prompts, demonstrating the model's ability to produce coherent and contextually appropriate language continuations.

The script is structured to be flexible, allowing for easy switching between different machine learning models and adjustment of parameters, making it suitable for experiments with natural language processing tasks.
- tokenizer.py: This file defines a series of functions and a class related to text tokenization using the SentencePiece library, specifically using a Byte Pair Encoding (BPE) model. Key functionalities include:

1. **TokenizerWrapper class:** Wraps the SentencePiece tokenizer for easier use in other parts of an application. It allows for the loading of a model, encoding texts to sequences of token IDs with optional beginning-of-sentence (BOS) and end-of-sentence (EOS) tokens, decoding token IDs back to strings while optionally skipping special tokens like BOS, EOS, and padding (PAD), and retrieving the IDs for special tokens.

2. **download_and_merge_text_files function:** Downloads text files from a provided API URL, merges them into a single file (corpus for training), handling encoding and concatenation.

3. **train_tokenizer function:** Trains a BPE tokenizer using a corpus from a specified path. The tokenizer is configured with a specified vocabulary size and includes definitions for special tokens such as padding, unknown, BOS, and EOS.

4. **download_file_from_url function:** A utility function to download a file from a given URL and save it locally, verifying the download status.

The workflow facilitated by these functions typically involves downloading and preparing textual data, training a tokenizer on this data, and then using the tokenizer for transforming texts to and from sequences of token IDs in applications such as machine learning models for natural language processing.
- dataset_loader.py: The file primarily implements utilities and classes needed for preprocessing and loading textual data for use in machine learning models, specifically those dealing with sequential data like natural language processing tasks.

1. **Function - `add_special_tokens`**: This function is designed to modify lists of strings by adding special tokens at the beginning and end of each string. It prepends "<bos>" (beginning of sentence) to prompts that start with a capital letter and appends "<eos>" (end of sentence) to completions that end with a period, exclamation mark, or question mark.

2. **Class - `TextDataset`**: This class inherits from `torch.utils.data.Dataset` and is customized to handle loading and encoding text data from a JSON-formatted file. Its constructor reads lines from a file, extracts the prompt and completion texts, combines them, and tokenizes them into IDs using a specified tokenizer, while also adding start and end tokens. The dataset is designed to support selective sequence length (capped by `max_seq_len`) and includes only samples that form a valid sequence.

3. **Function - `__len__`** (inside TextDataset): This method returns the number of samples in the dataset.

4. **Function - `__getitem__`** (inside TextDataset): This method returns a tuple of tensors representing inputs and targets for a given index. The inputs consist of all token IDs except the last, and the targets consist of all token IDs except the first to facilitate prediction tasks (like language modeling).

5. **Function - `collate_fn`**: This function is used in data loading to efficiently batch multiple samples. It pads sequences in a batch to a uniform length using a specified padding value (`pad_val`). This function ensures that the data fed into a model is appropriately batched and padded for training, making it compatible with models expecting input of uniform size.

Overall, the file provides tools to facilitate the preprocessing, tokenization, loading, and batching of text data for machine learning applications, ensuring data is model-ready for tasks that involve predicting sequences or understanding language context.
- rnn_model.py: The provided Python file defines a simple RNN-based language model built using PyTorch for generating text sequences. Here is a summary of its functionalities:

1. **Class Definition (`RNNLanguageModel`)**: This class inherits from `torch.nn.Module` and encapsulates the RNN language model. It includes methods for initializing the model, performing a forward pass, predicting the next token, and generating text given a prompt.

2. **Initialization (`__init__`)**: Initializes the language model with an embedding layer, an RNN layer, a dropout layer, and a fully connected layer to output the probabilities over the vocabulary. The constructor parameters specify the size of the vocabulary, embedding dimension, hidden dimension of the RNN, number of RNN layers, the pad token identifier, and the dropout probability.

3. **Forward Pass (`forward`)**: Takes token IDs as input and computes the forward pass of the model using the embedding, RNN, and dropout layers. It returns the logits output by the fully connected layer and the hidden state.

4. **Predict Next Token (`predict_next_token`)**: Given input token IDs, this method uses the model to predict the next token in the sequence. It uses a temperature parameter to scale the logits and adjust the "sharpness" of the probability distribution before sampling from it.

5. **Text Generation (`generate`)**: Generates text starting from a user-provided prompt. It repeatedly uses the model to predict the next token and append it to the sequence until an end-of-sequence token is generated or a maximum length is reached. It returns either the full generated sequence including the prompt or just the continuation depending on the `return_continuation_only` parameter.

The model and methods exemplify a basic implementation of a recurrent neural network for tasks like text prediction and generation, showcasing how to work with embeddings, RNNs, and softmax layers effectively in a practical setting using PyTorch.
- lstm_model.py: This file defines a language model based on Long Short-Term Memory (LSTM) networks in PyTorch. Here's a summary of each component:

1. **LSTMLanguageModel class**: Inherits from PyTorch's `nn.Module`. It's designed to model language using embedded representations and recurrent neural networks.

   - **__init__ method**: Initializes the model with the following:
     - **embedding**: Maps each token in a vocabulary to a high-dimensional vector.
     - **lstm**: A stack of LSTM layers for sequential data processing.
     - **dropout**: Regularization method to prevent overfitting by randomly setting a fraction of the input units to zero during training.
     - **fc (fully connected layer)**: Maps the LSTM output to the vocabulary size to predict the likelihood of each token.

2. **forward method**: Defines the computation performed at every call, taking `input_ids` (token indices) and an optional hidden state. It returns the logits (non-normalized predictions) for the next token in the sequence and the new hidden state.

3. **predict_next_token method**: Predicts the next token in a sequence given the input tokens. Uses temperature scaling to control the randomness of predictions and includes functionality to avoid re-generating the beginning-of-sentence token.

4. **generate method**: Generates text starting from a given prompt. Continues to generate tokens using the `predict_next_token` method until the end-of-sentence token is generated or the maximum length is reached. It offers an option to return only the text generated after the prompt.

This model can be trained to predict the probability of the next token in the sequence, which is typical for language modeling. It can also be used for generating text sequences given a prompt.
- llm_experiment.py: This Python script, part of a Jupyter notebook (`llm_experiment.ipynb`), is set up for conducting language model training experiments using various neural network architectures such as GRU, LSTM, RNN, and Transformer. The script includes the following key steps and functionality:

1. **Imports and Setup**:
    - It imports necessary modules including PyTorch (`torch`), and several helper modules for tokenization (`tokenizer`), dataset loading (`dataset_loader`), and the models themselves (`gru_model`, `lstm_model`, `rnn_model`, `transformer_model`).
    - It defines a device for running the training (CPU or GPU, depending on availability).

2. **Hyperparameter Configuration**:
    - A dictionary named `hyperparams_grid` contains sets of hyperparameters for different models. This allows easy configuration and iteration over various setups to test their performance impacts.

3. **Data Handling**:
    - Functions from the `tokenizer` module handle the downloading, merging, and preparation of text files and training the tokenizer.
    - Text datasets for training and testing are loaded and prepared using a custom `TextDataset` class and `DataLoader` from PyTorch, which handles batching and sequence padding.

4. **Model Training and Evaluation**:
    - The script defines a function `run_experiments()` that iterates through specified model configurations, initializes models, and trains them using the training dataset. It evaluates models on a testing dataset, stores, and logs results including performance metrics like perplexity and BLEU score.
    - The experiment can be conducted over different neural network architectures by initializing respective class objects (`GRULanguageModel`, `LSTMLanguageModel`, etc.).

5. **Execution Flow**:
    - It involves initializing the tokenizer, setting up datasets, data loaders, and then calling `run_experiments()` with the desired model type and hyperparameters to start the training and evaluation process.

The file as a whole is structured to facilitate experimenting with different language model architectures and configurations, aiming to compare their performance on a textual dataset automatically and systematically.
