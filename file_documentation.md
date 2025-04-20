# Detailed File-Level Documentation

# gru_model.py

## File Summary
The `model.py` file defines a GRU-based neural network language model in PyTorch specifically tailored for generating text. Here's a summary of the key components and functionality of the script:

1. **Imports**: The script imports necessary modules from PyTorch including basic neural network layers and functional APIs.

2. **GRULanguageModel Class**: This class inherits from `nn.Module` and implements a language model using a GRU (Gated Recurrent Unit) layer. Its main components are:
   - `embedding`: Maps token indices to embeddings.
   - `gru`: A GRU layer for processing sequences of embeddings with support for dropout and multiple layers.
   - `dropout`: A dropout layer for regularization.
   - `fc`: A linear layer that projects the GRU output back to the vocabulary space.

3. **Constructor `__init__`**: Initializes the model parts and accepts parameters for vocabulary size, embedding dimension, hidden dimension of the GRU, number of GRU layers, padding token ID, and dropout probability.

4. **`forward` Method**: Defines the forward pass of the model which includes embedding the input tokens, passing them through the GRU, applying dropout, and then through a linear layer to get the logits for next token prediction.

5. **`predict_next_token` Method**: This method predicts the next token in the sequence given the current input ids and optional hidden state. It handles setting the model to evaluate mode, prevents logit explosion using temperature scaling, and blocks prediction of a beginning-of-sequence token by setting its logit to negative infinity.

6. **`generate` Method**: Facilitates text generation from a prompt. It encodes the given prompt using a tokenizer, initializes generation with optional device placement, and iteratively predicts next tokens until the end-of-sequence token is predicted or a maximum length is reached. It supports returning either the full text including the prompt or just the continuation after the prompt.

This class provides a comprehensive structure for text generation using a GRU model, with methods for predicting the next token and generating text given a starting prompt, making it suitable for applications in natural language processing tasks like text completion and automated writings.

## Classes
- GRULanguageModel: The `GRULanguageModel` class is a PyTorch `nn.Module` designed for natural language processing tasks such as token prediction and text generation. It uses a Gated Recurrent Unit (GRU) network to model language sequences. Here are the key components and functionalities of the class:

1. **Initialization (`__init__` method)**:
   - **Parameters**:
     - `vocab_size`: Size of the vocabulary.
     - `embed_dim`: Dimensionality of token embeddings.
     - `hidden_dim`: Dimensionality of the hidden state of the GRU.
     - `num_layers`: Number of GRU layers.
     - `pad_token_id`: Token ID used for padding.
     - `dropout_prob`: Probability of dropping out edges during training.
   - **Components**:
     - An embedding layer to convert token IDs to vectors.
     - A GRU network for handling sequences.
     - A dropout layer for regularization.
     - A fully connected layer to map from hidden state space back to vocabulary space for output.

2. **Forward Pass (`forward` method)**:
   - Processes input tokens through the embedding layer, GRU network, and dropout, finally outputting a vector of logits and a hidden state for each token in the sequence.

3. **Token Prediction (`predict_next_token` method)**:
   - Performs inference to predict the next token in a sequence given the input token IDs and possible previous hidden states.
   - Utilizes softmax to convert logits to probabilities after adjusting for temperature, which controls the randomness in prediction.
   - Masks the probability of the beginning-of-sentence token to prevent it from being predicted during generation.

4. **Text Generation (`generate` method)**:
   - Generates text starting from a given prompt and extending up to a specified maximum length.
   - It can operate in two modes based on `return_continuation_only`:
     - When `True`, the method returns only the continuation of the prompt.
     - When `False`, it returns the entire text including the prompt.
   - Uses the `predict_next_token` to iteratively generate each token.

The class effectively encapsulates a language model that leverages a GRU to process and generate text, applying techniques like dropout for better generalization and providing functionalities to interact with higher-level operations like text generation based on provided prompts.

# train_utils.py

## File Summary
The Python file provides code for training, evaluating, and visualizing the performance of a machine learning model using the PyTorch library. Here’s a summary of the main components:

1. **Imports**: The script imports necessary libraries and modules including `torch`, `matplotlib` for plotting, and `nltk` for computing BLEU scores, which are used to assess translation models.

2. **train_model function**: 
   - Trains a given model on a specified training dataset and validates it on a test dataset.
   - Utilizes `AdamW` optimizer with learning rate schedulers and cross-entropy loss.
   - At the end of each epoch, it checks if the validation loss is the lowest encountered and if so, saves the model's state.
   - Losses for each epoch are recorded and subsequently plotted using `plot_losses`.

3. **plot_losses function**:
   - Plots training and validation loss curves and saves the plot to a file named according to the model type and timestamp.

4. **evaluate_model function**:
   - Loads a model from a specified path and evaluates it on a test data loader.
   - Uses cross-entropy loss and BLEU score for evaluation.
   - Outputs include total loss and BLEU scores for each batch.

The script is structured to support typical machine learning workflows of training, evaluating, and visualizing model performance, specifically designed for deep learning models in natural language processing tasks with the example usage of CrossEntropy loss and BLEU for evaluation metrics.

## Top-level Functions
- train_model: The `train_model` function is designed to train a machine learning model using the provided training and testing datasets, with customization for learning parameters and model saving.

Here's an outline of its functionality:

1. **Initialization**: It initializes the AdamW optimizer with a specified learning rate (`lr`) and a scheduler to reduce the learning rate based on the validation loss performance. A cross-entropy loss function integrates an 'ignore_index' for the padding token from the tokenizer.

2. **Training Loop**: Over a series of epochs (default 30):
    - **Train Phase**: For each batch from the `train_loader`, it computes the loss between the model outputs and targets, backpropagates to update model weights, and accumulates the train loss.
    - **Validation Phase**: The model is set to evaluation mode to prevent updates during validation. It calculates the validation loss for batches from the `test_loader`.
    - **Logging and Learning Rate Adjustment**: After each epoch, the average losses for training and validation are calculated and printed. The learning rate scheduler may adjust the learning rate based on the validation loss.
    - **Model Saving**: If the validation loss improves (i.e., decreases below the previously best recorded validation loss), the model's state dictionary is saved to the specified path.

3. **Loss Plotting**: After training, it plots the training and validation loss evolution over the epochs using a helper function, helping to visualize the model's learning progress.

The function effectively handles the entire training and validation process, including dynamic updates of learning parameters and tracking of performance metrics, while saving the best-performing model state.

- plot_losses: The provided function, `plot_losses`, takes three parameters: `train_losses` and `val_losses`, which are lists representing the loss during training and validation respectively, and an optional parameter `model_name` set to "model" by default. The function generates a plot displaying both the training and validation loss curves and titles the plot using the name of the model converted to uppercase followed by "Loss Curve". The x-axis represents epochs, and the y-axis represents loss. It includes a legend to differentiate between the training and validation curves.

The plot is saved as a PNG file named using the model name, the phrase "loss_curve", and the current timestamp to ensure the filename is unique. The function then closes the plot to free up memory and prints out a message indicating where the loss curve was saved, showing the filename. The filename includes a dynamically generated timestamp to ensure it is unique each time the function is run.

- evaluate_model: The function `evaluate_model` is designed to load a pre-trained model and evaluate its performance on test data, specifically assessing its perplexity (PPL) and BLEU score. The process involves the following steps:

1. **Loading the Model**: The function starts by loading the model parameters from a saved state `model_path` using `torch.load`.

2. **Evaluation Mode**: Sets the model to evaluation mode (`model.eval()`), which turns off certain features like dropout.

3. **Loss Function**: Initializes a `CrossEntropyLoss` for the model, which ignores padded indices in sequences as defined by the tokenizer.

4. **Preparation for Metrics Calculation**: The variables `total_loss` and `bleu_scores` are initialized to accumulate the loss values and compute BLEU scores across all test data batches.

5. **Evaluation Loop**: Iterates over `test_loader` which contains batches of input and target sequences.
   - The function moves the current batch to the specified `device`.
   - It computes the logits of the model and evaluates the batch loss.
   - Accumulates the computed loss to `total_loss`.
   - Converts logits to predicted sequences and calculates the BLEU score for each predicted sequence compared to the ground truth, using a smoothing function to adjust the score computation.

6. **Calculation of Final Metrics**:
   - Computes the perplexity from the total loss accumulated across all batches.
   - Calculates the average BLEU score from scores obtained from individual predictions.

7. **Output**: Prints and returns the perplexity and the average BLEU score.

This function is valuable for evaluating translation models or other sequence generation tasks where alignment between the predicted sequence and the ground truth is important.

# transformer_model.py

## File Summary
This Python file defines a Transformer-based language model class named `TransformerLanguageModel` for next-word prediction, built using PyTorch. Here are the key components and functionalities of the model:

1. **Class Definition**: `TransformerLanguageModel` is a subclass of `nn.Module`. It is designed to predict the next word in a sequence by considering the context provided through previous words.

2. **Initialization**:
    - The model is initialized with parameters like vocabulary size, embedding dimension, number of attention heads, number of layers, pad token ID, sequence length, and dropout rate.
    - It uses embeddings for tokens and positions and constructs a Transformer encoder from PyTorch's `nn.TransformerEncoderLayer` and `nn.TransformerEncoder`.
    - A fully connected output layer and a dropout layer are also included.

3. **Forward Pass**:
    - The `forward` method involves computing embeddings, applying dropout, generating subsequent masks (to enforce causality and manage padding), and passing the result through the transformer encoder.
    - The output from the transformer is then fed into the fully connected layer to get the logits.

4. **Auxiliary Methods**:
    - `_generate_square_subsequent_mask`: Generates a mask for the Transformer to prevent attention to future positions.
    - `predict_next_token`: Predicts the next token ID using temperature sampling to soften the probability distribution before sampling.
    - `generate`: Autoregressively generates text based on a provided prompt and optional parameters like maximum length and temperature. It can return either the full text including the prompt or just the continuation.

5. **Usage**:
    - This model can be employed in tasks like text generation or interactive dialogue systems where predicting the next word or generating text based on given inputs is required.

The overall script is structured to provide essential functionalities for text generation utilizing the Transformer architecture, with provisions for sequence-to-sequence transformation, embeddings, and management of sequence lengths and padding.

## Classes
- TransformerLanguageModel: The `TransformerLanguageModel` class is a PyTorch module designed for next-word prediction using a Transformer architecture. Here's a summary of its key components and functionalities:

1. **Initialization Parameters**:
   - `vocab_size`: The size of the vocabulary.
   - `embed_dim`: The dimension of the token embeddings.
   - `num_heads`: The number of attention heads in each Transformer encoder layer.
   - `num_layers`: The number of layers in the Transformer encoder.
   - `pad_token_id`: The ID used for padding tokens.
   - `max_seq_len`: The maximum sequence length for position embeddings, defaulting to 512.
   - `dropout`: The dropout rate used in the Transformer and embedding layers.

2. **Components**:
   - `token_embedding`: Embedding layer for token IDs.
   - `position_embedding`: Embedding layer for position indices up to `max_seq_len`.
   - `transformer`: A Transformer encoder consisting of multiple layers specified by `encoder_layer`.
   - `fc_out`: A linear layer that projects from the embedding dimension back to the vocabulary size.
   - `dropout`: Dropout layer applied to embeddings.

3. **Forward Pass**:
   - Processes input token IDs, adds token and position embeddings, applies dropout, and uses a Transformer encoder with mask handling to generate predictions, returning logits and a placeholder for the hidden state.

4. **Mask Generation**:
   - `_generate_square_subsequent_mask`: Creates a mask to prevent leakage from future tokens during training in the Transformer encoder.

5. **Prediction and Text Generation**:
   - `predict_next_token`: Does inference to predict the next token ID from the last output logits, using temperature to control the randomness of predictions.
   - `generate`: Autoregressively generates text from a prompt using the `predict_next_token` method, handling continuation-only or full text returns.

Overall, this class encapsulates a Transformer-based model for generating text or predicting the next token, effectively modeling language given a series of input tokens. It utilizes embeddings, masked multi-head self-attention, and linear layers within a Transformer configuration to achieve its tasks.


# llm_project.py

## File Summary
The provided Python script automates several tasks related to training language models on a specific dataset with various neural architectures. Here’s a concise breakdown of what the Python file contains and how it processes:

1. **Imports and Setup**
   - The script utilizes `torch` for deep learning operations.
   - Multiple helper modules are imported to handle tokenization (`tokenizer`), data loading (`dataset_loader`), model definitions (`gru_model`, `lstm_model`, `rnn_model`, `transformer_model`), and training utilities (`train_utils`).

2. **Configuration**
   - Parameters required for training such as data URLs, file paths, tokenizer settings, network parameters, and training settings (batch size, epochs, device) are configured.

3. **Data Handling**
   - It downloads training and test datasets from specified URLs.
   - A tokenizer is trained using a corpus obtained and merged from a given data source URL. This tokenizer is then wrapped for further use with datasets.

4. **Dataset Preparation**
   - Text datasets for both training and testing are created using the tokenizer and a maximum sequence length parameter.
   - Data loaders for both datasets are prepared with specific batch sizes and shuffling settings.

5. **Model Building**
   - A function `build_model` is defined to instantiate one of the four model types (`GRU`, `LSTM`, `RNN`, `Transformer`) based on the configuration, with each model appropriately moved to the available computing device (GPU or CPU).

6. **Training and Evaluation**
   - The selected model is trained using the train data loader and evaluated against the test data loader. Both the model training and evaluation phases involve the use of utility functions that handle these processes.
   - The trained model is saved to a specified path.

7. **Sample Text Generation**
   - At the end of the script, the trained model is used to generate text based on custom prompts provided in the script. This demonstrates the model's ability to generate coherent text based on input sequences.

Overall, the script represents a comprehensive to setup, train, evaluate, and utilize deep learning models for natural language processing, showcasing flexibility across different neural network architectures (GRU, LSTM, RNN, and Transformer). The outcomes of the script depend on the data and the model configuration chosen at the outset.

## Top-level Functions
- build_model: The function `build_model` takes a single argument `model_type` and returns an instance of a language model based on the specified model type. It supports four types of models: "gru", "lstm", "rnn", and "transformer". Each model is configured with predefined parameters such as vocabulary size (`VOCAB_SIZE`), embedding dimension (set to 256), and other architecture-specific parameters. These models are moved to a specified device (`DEVICE`) after being instantiated. If the `model_type` is not one of the predefined types, the function raises a `ValueError` indicating that the model type is unsupported.

# tokenizer.py

## File Summary
The Python file defines utilities for tokenization and management of text data. Here’s a summary of the components:

1. **TokenizerWrapper Class:**
   - This class acts as a wrapper around the SentencePiece tokenizer, facilitating tokenization tasks.
   - It initializes by loading a SentencePiece model and sets up special tokens (`<bos>`, `<eos>`, `<pad>`) and their respective token IDs.
   - Provides `encode()` method to convert text into token IDs, with optional addition of beginning-of-sentence (`<bos>`) and end-of-sentence (`<eos>`) tokens.
   - Includes a `decode()` method to convert token IDs back to text, with an option to skip special tokens.
   - Methods to retrieve IDs for pad, bos, and eos tokens.

2. **download_and_merge_text_files Function:**
   - Downloads text files from a specified API URL and merges them into a single output file. It specifically looks for files ending with `.txt`.

3. **train_tokenizer Function:**
   - Trains a Byte Pair Encoding (BPE) tokenizer using SentencePiece on a corpus located at `corpus_path`, saving the model with a specified prefix and a user-defined vocabulary size. The tokenizer is configured with specific token IDs for padding, unknown, beginning of sentence, and end of sentence.

4. **download_file_from_url Function:**
   - Downloads a file from a provided URL and saves it to a specified filename. It checks for successful download and informs the user upon completion.

Overall, the file provides a suite of tools for text data preprocessing, tokenization, and model training suitable for tasks involving natural language processing.

## Classes
- TokenizerWrapper: The `TokenizerWrapper` class is designed as a utility for handling text tokenization using a pre-trained SentencePiece model, which is specified by the `model_path` provided during object initialization. The key functionalities provided by this class include:

1. **Initialization (`__init__` Method)**: 
   - Loads the SentencePiece model.
   - Sets up special tokens for beginning of sentence (`bos`), end of sentence (`eos`), and padding (`pad`).
   - Retrieves and stores the token IDs for these special tokens.

2. **Text Encoding (`encode` Method)**:
   - Converts a given text to a sequence of token IDs using the SentencePiece model.
   - Optionally adds `bos` and `eos` tokens at the beginning and end of the token sequence, respectively.

3. **Text Decoding (`decode` Method)**:
   - Converts a sequence of token IDs back into string form using the SentencePiece model.
   - Optionally skips special tokens (such as `bos`, `eos`, and `pad`) during the conversion process.

4. **Utility Methods**: 
   - `get_pad_id`, `get_eos_id`, `get_bos_id` methods for retrieving the IDs of the pad, eos, and bos tokens, respectively.

This class provides a streamlined interface for tokenizing and detokenizing texts with additional control over the inclusion of special tokens, suitable for tasks in natural language processing that require precise handling of input and output sequences.

## Top-level Functions
- download_and_merge_text_files: The function `download_and_merge_text_files` takes two parameters: `api_url` and `output_file`. It starts by making a GET request to the `api_url` to retrieve a list of files. It then iterates through each file listed in the JSON response. For each file that has a `.txt` extension (identified by the file name ending with ".txt"), the function makes another GET request to the file’s download URL to fetch the text content. This content is then written to a specified output file (`output_file`), appending a newline character after each file's content. This continues for all text files in the list, effectively merging them into the output file with UTF-8 encoding.

- train_tokenizer: The `train_tokenizer` function is designed to train a Byte-Pair Encoding (BPE) tokenizer using a specified corpus. The function is part of a larger code that utilizes the SentencePiece library for tokenizer training.

Here’s a breakdown of its parameters and process:
1. **corpus_path**: The file path to the corpus data that will be used for training the tokenizer.
2. **prefix**: A prefix string for naming the trained model files.
3. **vocab_size** (optional, default=10000): The number of tokens to be included in the tokenizer's vocabulary.

The function configures the training with several parameters specific to the BPE model:
- **model_type**: Set to `'bpe'` for Byte-Pair Encoding.
- **pad_id=3**: Assigns the id `3` to the padding token.
- **unk_id=0**: Sets the id `0` for the unknown token.
- **bos_id=1**: Designates the id `1` for the beginning of sentence token.
- **eos_id=2**: Assigns the end of sentence token id `2`.
- **user_defined_symbols**: Includes special tokens like `<bos>`, `<eos>`, and `<pad>` in the vocabulary.

This setup indicates a basic configuration for a BPE tokenizer, suitable for various text processing tasks that require tokenization with a fixed vocabulary size, special token handling, and using SentencePiece's efficient training mechanism.

- download_file_from_url: The function `download_file_from_url` downloads a file from a specified URL (`file_url`) and saves it to a local file with the name `output_filename`. It performs the following steps:

1. It makes an HTTP GET request to the specified URL.
2. It checks if the request was successful (`response.raise_for_status()`) and raises an exception if there was an error such as a 404 or 500 status code.
3. If the request is successful, the content of the response (assuming it is text) is saved to a file specified by `output_filename`. The file is encoded in UTF-8.
4. It prints a message indicating that the file was successfully downloaded and saved.
5. Additionally, it prints another message confirming the name of the downloaded file. 

This function essentially handles the downloading of text files from the web and ensures they are saved locally with proper error handling and notifications.

# dataset_loader.py

## File Summary
This Python file is focused on text processing for machine learning applications using PyTorch. It involves adding special tokens to text data, managing a custom text dataset, and preparing data batches for training. Here's a summary of the key components:

1. **Imports:**
   - The script imports necessary modules such as `json` for parsing JSON files, `torch`, and components from `torch.utils.data` and `torch.nn`.

2. **Function `add_special_tokens`:**
   - This function adds special tokens "<bos>" (beginning of sentence) and "<eos>" (end of sentence) to the dataset. "<bos>" is added at the beginning if the first letter of the prompt is uppercase, and "<eos>" is appended if the completion ends with '.', '?', or '!'. It returns two lists containing the updated prompts and completions.

3. **Class `TextDataset` (inherits from `Dataset`):**
   - This class is used to handle a text dataset. It takes a file path, a tokenizer, and a maximum sequence length as inputs.
   - It parses a JSON file to extract text data, tokenizes the text, and truncates or pads it to a specified maximum length. The tokenized text objects are stored internally.
   - The `__len__` method returns the number of samples in the dataset.
   - The `__getitem__` method retrieves a text sample by its index, returning a version offset by one token as input and target, effectively setting up data for next-token prediction tasks.

4. **Function `collate_fn`:**
   - This function is used for batching the data. It receives a list of samples and a padding value.
   - Inputs and targets are extracted, then padded to the maximum sequence length in the batch. It ensures that all data samples in a batch are of the same length, returning the batched inputs and targets.

This script is structured to facilitate the preprocessing and loading of text data, utilizing PyTorch's capabilities for efficient handling of sequences and batch preparation, which are essential steps in the pipeline of training models for natural language processing tasks.

## Classes
- TextDataset: The `TextDataset` class, derived from `Dataset`, is designed to process text data for natural language processing tasks. It initializes by loading text samples from a specified file and tokenizing them into numerical IDs using the provided tokenizer. The key steps and features of this class are as follows:

1. **Initialization (`__init__`):**
   - The constructor takes three parameters: `filepath` for the dataset file, `tokenizer` for converting text to tokens, and `max_seq_len` to limit the sequence length of tokens.
   - The text file is read line by line, and each line is expected to be a JSON object with keys 'prompt' and 'completion'. These values are concatenated, stripped of leading/trailing whitespace, and fed into the tokenizer.
   - The tokenizer encodes the concatenated text to produce a sequence of token IDs, capped at `max_seq_len`. The sequence includes beginning-of-sentence (`bos`) and end-of-sentence (`eos`) tokens.
   - Only sequences with at least 2 tokens are added to the dataset (`self.samples`).

2. **Length (`__len__`):**
   - This method returns the number of samples in the dataset.

3. **Item Access (`__getitem__`):**
   - Retrieves a specific sample by its index (`idx`). This method is used to access individual token sequences in the dataset.
   - For each sample accessed, it returns a pair of tensors: the input sequence (`tokens[:-1]`) and the target sequence (`tokens[1:]`). This setup is typical in language modeling, where the model predicts the next token in a sequence based on the previous tokens.

Overall, the `TextDataset` class neatly packages text data into manageable sequences useful for training language models, providing functionality to both preprocess and access the data efficiently.

## Top-level Functions
- add_special_tokens: The function `add_special_tokens` takes a tuple of two lists of strings as input, representing pairs of prompts and completions. It processes each pair to add special tokens: `<bos>` (beginning of sentence) is prepended to prompts that start with an uppercase letter, and `<eos>` (end of sentence) is appended to completions that end with punctuation marks (period, question mark, or exclamation point). The function returns a tuple containing two lists: one for the modified prompts and the other for the modified completions. This helps in marking the beginning and end points of sentences within the dataset, which could be useful for training language models or other NLP applications.

- collate_fn: The `collate_fn` function is designed to process a batch of data, ensuring that both inputs and targets are appropriately padded to the same length for batch processing in neural network models. Here's a step-by-step summary of what this function does:

1. **Unpacking the Batch**: The function takes two parameters: `batch`, which is a list of tuples where each tuple contains an `input` sequence and a corresponding `target` sequence; and `pad_val`, which specifies the padding value to use for sequences shorter than the maximum length in the batch.

2. **Separating Inputs and Targets**: Using `zip(*batch)`, the function separates the inputs and targets from the batch into separate tuples.

3. **Padding the Sequences**: It uses the `pad_sequence` method from `nn.utils.rnn` for both `inputs` and `targets`. This method adjusts the length of all sequences in each tuple to match the longest sequence by adding the `pad_val` value at the end of shorter sequences. The `batch_first=True` argument ensures that the batch dimension is at the first axis (i.e., the shape of the output tensor will be `[batch_size, max_seq_length]`).

4. **Returning Processed Batch**: The function returns a tuple containing the padded `inputs` and `targets`.

This function is particularly useful when building models that handle sequences of varying lengths, such as those commonly found in natural language processing or time series analysis, allowing for efficient batch processing in deep learning architectures.

# rnn_model.py

## File Summary
This Python file defines a simple recurrent neural network (RNN) based language model using PyTorch. The model, named `RNNLanguageModel`, is implemented as a class that inherits from PyTorch's `nn.Module`. Here are the key functionalities and components of the model:

1. **Initialization (`__init__`)**:
   - The model is initialized with several parameters: `vocab_size` (size of the vocabulary), `embed_dim` (dimension of the embedding layer), `hidden_dim` (dimension of the hidden layer in RNN), `num_layers` (number of RNN layers), `pad_token_id` (padding token ID for embeddings), and an optional `dropout_prob` (probability for dropout layers).
   - The model architecture includes an embedding layer for token embeddings, a recurrent neural network (RNN) layer, a dropout layer for regularization, and a fully connected (linear) layer that outputs the logits representing the probability distribution over the vocabulary.

2. **Forward Pass (`forward`)**:
   - The `forward` method defines how the data flows through the model. It takes `input_ids` (token indices) and an optional `hidden` state as inputs. The method processes the input ids through the embedding layer, the RNN layer (where it optionally takes a hidden state), applies dropout, and then returns the output logits from the fully connected layer along with the new hidden state.

3. **Predict Next Token (`predict_next_token`)**:
   - This method is designed to predict the next token in a sequence given the current input tokens. It prevents the sampling of a beginning-of-sentence (`<bos>`) token, scales the logits by a temperature parameter (for controlling randomness), and applies softmax to convert logits to probabilities. The method then samples from these probabilities to get the next token and returns it along with the updated hidden state.

4. **Generate Text (`generate`)**:
   - This method generates text given a prompt. It uses the `predict_next_token` method in a loop to generate tokens up to a maximum length or until an end-of-sentence (`<eos>`) token is encountered. It supports returning either the full text (including the prompt) or just the continuation of the prompt. Text generation is controlled by parameters such as `device` (to place tensors on the correct hardware), `max_length`, `temperature`, and `return_continuation_only`.

Overall, this file provides a basic framework for training and using an RNN-based language model for generating text sequences, applicable in tasks such as autocompletion, chatbot responses, or other language generation tasks.

## Classes
- RNNLanguageModel: The `RNNLanguageModel` class defined above is a recurrent neural network-based language model built using PyTorch's `nn.Module`. 

Key Components of the Class:

1. **Initialization (`__init__` method)**:
   - **Parameters**:
     - `vocab_size`: Total number of tokens in the vocabulary.
     - `embed_dim`: Dimensionality of the token embeddings.
     - `hidden_dim`: Dimensionality of the RNN's hidden state.
     - `num_layers`: Number of RNN layers stacked together.
     - `pad_token_id`: ID used to represent padding tokens.
     - `dropout_prob`: Probability of an element to be zeroed (default is 0.3).
   - **Components**:
     - `self.embedding`: Embedding layer to convert token IDs to embeddings.
     - `self.rnn`: RNN layer that processes the sequences.
     - `self.dropout`: Dropout layer applied to the outputs of the RNN.
     - `self.fc`: Linear layer that projects the RNN output back to the vocabulary size for next token prediction.

2. **Forward Pass (`forward` method)**:
   - Process input through the model in a forward direction to predict outputs based on input ids.
   - It returns the logits (output from `self.fc`) and the updated hidden states.

3. **Predicting Next Token (`predict_next_token` method)**:
   - Uses the model in evaluation mode to predict the next token from the input sequence.
   - Applies a temperature to adjust the "sharpness" of the distribution before sampling a token.
   - Ensures the beginning-of-sentence (BOS) token is not predicted again by setting its logit to negative infinity.

4. **Generating Text (`generate` method)**:
   - Generates text autoregressively using the provided prompt for a specified maximum length or until an end-of-sentence (EOS) token is produced.
   - Accepts parameters like device (e.g., `cpu` or `cuda`), maximum output length, and temperature.
   - Offers an option to return only the continuation of the prompt or the entire generated sequence including the prompt.
   - Leverages the `predict_next_token` method for generating each subsequent token.

This model effectively encapsulates the typical functionalities of an RNN-based language model for tasks such as next-token prediction and text generation, using learned embeddings and recurrent dynamics.

# lstm_model.py

## File Summary
The Python file presents an implementation of an LSTM-based language model using PyTorch. The `LSTMLanguageModel` class, inheriting from `torch.nn.Module`, is defined within this file. Here's a summary of the key components and functionalities provided by this class:

1. **Initialization**: The `__init__` method sets up the LSTM language model with configurable parameters including vocabulary size (`vocab_size`), embedding dimension (`embed_dim`), hidden dimensions (`hidden_dim`), number of layers (`num_layers`), padding token ID (`pad_token_id`), and dropout probability (`dropout_prob`). It initializes an embedding layer, LSTM layer, dropout layer, and a fully connected layer within the model architecture.

2. **Forward Pass**: Implemented in the `forward` method, it processes input tokens (`input_ids`) by first embedding them, then passing them through an LSTM network, applying dropout, and finally projecting the LSTM outputs to the vocabulary space using a linear layer. It returns the logits and the hidden states.

3. **Token Prediction**: The `predict_next_token` method predicts the next token given a sequence of input IDs, a beginning-of-sentence (`bos_id`) token to prevent its generation, hidden states, and temperature for smoothing. It outputs the predicted next token ID and updated hidden states, using softmax for probability distribution and sampling.

4. **Text Generation**: The `generate` method performs text generation starting from a provided prompt. It takes additional parameters such as the tokenizer, the device on which tensors are processed, maximal length for the generated text, temperature for controlling randomness, and a flag whether to return only the text continuation from the prompt. This method manages the sequence generation, token-by-token, stopping when an end-of-sentence token (`eos_id`) is generated or the maximum length is reached.

Overall, this model facilitates building a generative language model leveraging LSTM networks for tasks such as text prediction and generation, accommodating control of text diversity and coherency through temperature adjustments and structural configurations.

## Classes
- LSTMLanguageModel: The class `LSTMLanguageModel` is a PyTorch neural network model that extends `nn.Module`. It's designed to generate language based on an LSTM architecture. The model initialization receives several parameters to configure the LSTM and associated layers:

- `vocab_size`: the size of the vocabulary.
- `embed_dim`: the dimensionality of the embedding layer.
- `hidden_dim`: the number of features in the hidden state of the LSTM.
- `num_layers`: the number of layers in the LSTM.
- `pad_token_id`: the token ID used for padding.
- `dropout_prob`: the dropout rate, defaulting to 0.3.

The model comprises four main components:
1. **Embedding Layer**: Maps each token ID to a high-dimensional space (defined by `embed_dim`).
2. **LSTM Layer**: Processes sequences using `hidden_dim`, `num_layers`, and manages dropout.
3. **Dropout Layer**: Applied to the outputs of the LSTM for regularization.
4. **Fully Connected Layer**: Transforms the hidden state output to the size of the vocabulary for token prediction.

The `forward` method defines the forward pass of the model, which takes token IDs (`input_ids`) and optional hidden states (`hidden`). It processes the input through embedding, LSTM, and dropout layers sequentially, and finally through the fully connected layer to produce output logits and hidden states.

Additional methods:
- `predict_next_token`: Predicts the next token in a sequence given the current input IDs and the beginning of sequence (BOS) token ID, controlling the randomness of predictions with a `temperature` parameter.
- `generate`: Generates text from a given prompt and tokenizer setup, able to continue up to `max_length` tokens and selectively return just the generated continuation or the entire text including the prompt.

This setup makes the model particularly suited for tasks such as text generation, where sequences of tokens need to be predicted.

# llm_experiment.py

## File Summary
This Python script, intended for execution on Jupyter Notebook via Google Colab, establishes a framework for experimenting with different types of language models using the PyTorch library. Here's a breakdown of its functionality:

1. **Imports**:
   - Core functionalities and models from PyTorch.
   - Utility functions related to tokenization, dataset creation, and model evaluations.
   - Various language model implementations such as GRU, LSTM, RNN, and Transformer.

2. **Setting Up the Environment**:
   - Hyperparameters for GRU, LSTM, RNN, and Transformer models are defined in a dictionary called `hyperparams_grid`.
   - Constants like paths for train and test datasets, tokenizer properties, vocabulary size, sequence length, batch size, and epochs are defined.
   - Configuration for the type of device (CPU/GPU) to run the model on.

3. **Data Preparation**:
   - Downloading and preparing corpus files.
   - Tokenization using a specially trained BPE tokenizer.
   - Loading datasets and creating batch loaders.

4. **Model Training and Evaluation**:
   - The script contains a function, `run_experiments`, to systematically evaluate different model configurations from the defined hyperparameter grid.
   - Each configuration is used to instantiate, train, and evaluate a model while tracking the performance metrics such as perplexity and BLEU score.
   - The models are trained using saved data loaders that handle dataset batching and padding appropriately.

5. **Experiment Execution**:
   - Although customizable, by default the script seems set to run experiments for the GRU model type, as indicated by the final incomplete line. This implies manual selection and commenting/uncommenting lines to switch between model types like GRU, LSTM, RNN, or Transformer for running respective experiments.

**Potential Enhancements and Usage**:
- To use this script effectively and to execute full experiments, a user would typically need to select the model type manually, possibly change hyperparameters, and might also ensure all dependent files and utilities are correctly implemented and imported.
- The code snippet is incomplete at the end but implies typical usage where you would specify the model type and possibly iterate over different types if automating comprehensive tests over multiple runs.
- The script seems well structured for modular experimentation with different neural network architectures in NLP applications, particularly for language modeling tasks.

## Top-level Functions
- run_experiments: This function, `run_experiments`, systematically evaluates different configurations of models in a specified type (`model_type`) based on a configuration grid (`grid`). These models are either variants of a `transformer` or another unspecified type, instantiated through a provided `ModelClass`. The configurations differ primarily in parameters such as embedding dimensions, number of heads and layers (for transformers), or hidden dimensions (for other model types). For each configuration:

1. A unique identifier (`model_id`) is generated for the model based on its type, index in the grid, and the current timestamp.
2. The function prints out the model being trained along with its configuration.
3. Depending on the `model_type`, either a transformer model or another model is initialized with specified parameters from the configuration grid and moved to the specified `device` for GPU computation (if available).
4. Each model is then trained and saved to disk (using `train_model` function), specifying learning rate from the config and constraining training to 50 epochs.
5. Post training, each model is evaluated using the `evaluate_model` function, which returns performance metrics such as perplexity (`ppl`) and BLEU score (`bleu`).
6. The results from each model evaluation are accumulated into a list with details on the `model_type`, `config`, `perplexity`, `bleu_score`, and local file path of the trained model (`model_path`). 

This function allows for the analysis of different neural network configurations' performances on training and testing datasets provided (`train_loader` and `test_loader`), leveraging the computational capabilities of the specified `device`.

