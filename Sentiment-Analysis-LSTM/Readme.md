# Sentiment Analysis Using LSTM

This project implements two different LSTM models (Basic LSTM and Stacked LSTM) for sentiment analysis on the "Twitter US Airline Sentiment" dataset. The goal is to analyze and classify the sentiment of tweets towards various airlines.

## Dataset Overview

The dataset used in this project is the "Twitter US Airline Sentiment" dataset, which contains tweets related to different airlines. The dataset comprises:

- **Objective**: Captures the sentiment (positive, negative, or neutral) of Twitter users toward various airlines.
- **Type of Data**: Structured dataset with both numerical and categorical variables.
- **Entries and Variables**: 14,640 tweets with 15 variables, including tweet ID, sentiment labels, confidence scores, reasons for negative sentiment, airline details, user information, tweet content, and additional metadata.

## LSTM Model Architectures

### 1. Basic LSTM Model

#### Model Class: `LSTMModel`

- **Parameters**:
  - `input_dim`: Dimensionality of the input data.
  - `embedding_dim`: Dimensionality of the word embeddings.
  - `hidden_dim1`, `hidden_dim2`, `hidden_dim3`: Dimensions of the hidden states for the three LSTM layers.
  - `output_dim`: Dimensionality of the output.

- **Layers**:
  - `embedding`: Converts input indices into dense vectors of size `embedding_dim`.
  - `lstm1`, `lstm2`, `lstm3`: Three LSTM layers with batch normalization.
  - `dropout`: Dropout layer with a probability of 0.3.
  - `fc`: Fully connected layer to produce the final output.

- **Forward Method**:
  - Embeds input using the embedding layer.
  - Passes through three LSTM layers successively.
  - Applies dropout after the third LSTM layer.
  - Passes the last time-step output through the fully connected layer to obtain the final output.

### 2. Improved Stacked LSTM Model

#### Model Class: `ImprovedLSTM`

- **Parameters**:
  - `input_dim`: Dimensionality of the input data.
  - `embedding_dim`: Dimensionality of the word embeddings.
  - `hidden_dim`: Hidden state dimensions for each LSTM layer.
  - `output_dim`: Dimensionality of the output.
  - `num_layers` (default=3): Number of LSTM layers.
  - `bidirectional` (default=True): Enables bidirectional LSTMs.

- **Layers**:
  - `embedding`: Converts input indices into dense vectors of size `embedding_dim`.
  - `lstm_layers`: A list containing multiple LSTM layers (determined by `num_layers`).
  - `fc`: Fully connected layer to produce the final output.

- **LSTM Layers**:
  - Dynamically created LSTM layers with `batch_first=True`.
  
- **Forward Method**:
  - Embeds input using the embedding layer.
  - Passes input through each LSTM layer in the `lstm_layers` list sequentially.
  - Takes the last time-step output and passes it through the fully connected layer to produce the final output.

## Model Performance and Evaluation

The models were evaluated based on test accuracy:

- **Basic LSTM Model Test Accuracy**: 0.7503
- **Stacked (Improved) LSTM Model Test Accuracy**: 0.7609

### Performance Comparison

The Improved Stacked LSTM model showed a slight improvement (about 1%) over the Basic LSTM model. Potential reasons for this marginal improvement include:

1. **Data Complexity**: The task might not be complex enough to benefit from added model capacity.
2. **Gradient Issues**: The stacked LSTM model may experience gradient vanishing or exploding problems.
3. **Hyperparameter Tuning**: Suboptimal tuning of hyperparameters like the number of layers, hidden state dimensions, and bidirectionality.

### Graphs and Visualizations

- Performance graphs for both models, showing training and validation accuracy over epochs, are included in the `graphs/` directory.

## Strengths and Limitations of Using RNNs for Sentiment Analysis

### Strengths

1. **Capturing Context**: RNNs consider the order and relationships between words, crucial for sentiment analysis.
2. **Learning Long-Term Dependencies**: LSTMs, a type of RNN, can learn dependencies between distant words in a sentence.
3. **Adaptability to Different Text Lengths**: RNNs can handle sentences of varying lengths.

### Limitations

1. **Vanishing/Exploding Gradients**: RNNs can suffer from gradient issues, although LSTMs help mitigate this.
2. **Data Sensitivity**: RNNs are highly data-driven; their performance depends heavily on data quality and size.
3. **Interpretability**: RNNs lack transparency in their decision-making process, making debugging challenging.

## Usage

To run this project, follow these steps:

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Code

You can find the code for the entire project in the `sentiment_analysis_lstm.ipynb` file.

## Detailed Report

A comprehensive report detailing the model architecture, training process, evaluation metrics, and performance graphs is available in `Sentiment analysis using LSTM report.pdf`.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!
