# Deep Learning Projects Repository

Welcome to the Deep Learning Projects Repository! This repository contains a collection of deep learning projects that showcase various techniques and architectures for solving real-world problems. Each project focuses on a different aspect of deep learning, including time-series forecasting, sentiment analysis, anomaly detection, and text classification using state-of-the-art models.

## Projects

### 1. Time-Series Forecasting with RNN

**Description:** This project involves forecasting time-series data using Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks. The dataset used is the Individual Household Electric Power Consumption dataset, which provides minute-by-minute measurements of electricity consumption.

**Key Features:**
- Utilizes three LSTM layers followed by fully connected layers.
- Implements ReLU activation and dropout for regularization.
- Evaluates model performance using Mean Absolute Error (MAE).

**Results:**
- Training MAE: 0.0640
- Validation MAE: 0.1334
- Test MAE: 0.0960

**Graphs:**
- MAE over epochs
- Training and validation loss over epochs

---

### 2. Sentiment Analysis using LSTM

**Description:** This project focuses on sentiment analysis of tweets using Long Short-Term Memory (LSTM) networks. The dataset used is the "Twitter US Airline Sentiment" dataset, which contains tweets about airlines, categorized by sentiment.

**Key Features:**
- Implements both a basic LSTM model and an improved stacked LSTM model.
- Compares performance in terms of accuracy and MAE.

**Results:**
- Basic LSTM Test Accuracy: 0.7503
- Stacked LSTM Test Accuracy: 0.7609

**Graphs:**
- Training and validation loss over epochs
- Summary statistics of reconstruction errors

---

### 3. Anomaly Detection with Autoencoders

**Description:** This project uses autoencoders for anomaly detection on a hard drive dataset. The dataset includes daily operational status of hard drives, aimed at predicting failures based on SMART attributes.

**Key Features:**
- Implements three different autoencoder architectures with various layer configurations.
- Evaluates models based on reconstruction error and anomaly detection accuracy.

**Results:**
- Autoencoder 1 Anomaly Detection Accuracy: 0.8919
- Autoencoder 2 Anomaly Detection Accuracy: 0.8913
- Autoencoder 3 Anomaly Detection Accuracy: 0.9148

**Graphs:**
- Reconstruction error statistics
- Training and validation loss over epochs

---

### 4. Transformer for Text Classification

**Description:** This project involves building a Transformer model for text classification using the AG News dataset. The dataset consists of news articles categorized into four classes, and the model aims to classify these articles effectively.

**Key Features:**
- Utilizes Transformer architecture with embedding layers, positional encoding, and multi-head self-attention.
- Implements L2 regularization, dropout, and early stopping for improved performance.

**Results:**
- Train Accuracy: 0.9315
- Test Accuracy: 0.8829
- Precision: 0.8859
- Recall: 0.8834
- F1 Score: 0.8825

**Graphs:**
- Training and validation accuracy over epochs
- Training and validation loss over epochs
- Confusion matrix
- ROC curve

---

## Repository Overview

This repository contains implementations of various deep learning projects using PyTorch. It aims to demonstrate the application of different deep learning techniques and architectures to diverse datasets and problems, including:

- **Time-Series Forecasting:** Predicting future values based on historical data.
- **Sentiment Analysis:** Classifying the sentiment of text data using LSTM networks.
- **Anomaly Detection:** Identifying outliers in operational data using autoencoders.
- **Text Classification:** Categorizing news articles with Transformer models.

Feel free to explore each project, run the code, and adapt it for your own use cases. Contributions and improvements are always welcome!

---

## Getting Started

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!

