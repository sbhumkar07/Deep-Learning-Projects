# Autoencoders for Anomaly Detection

This project focuses on using various autoencoder architectures to detect anomalies in hard drive health data. The dataset comprises daily operational status information for hard drives, including Self-Monitoring, Analysis, and Reporting Technology (SMART) attributes. The goal is to identify potential hardware failures through the analysis of reconstruction errors.

## Dataset Overview

The dataset provides information about the daily status of operational hard drives, including details such as serial numbers, models, capacities, failure status, and SMART attributes. The dataset contains structured tabular data, with each row representing a daily snapshot of a hard drive's health.

- **Entries:** 3,179,295 rows (daily snapshots)
- **Variables:** 95 columns (features), including:
  - `date`, `serial_number`, `model`, `capacity_bytes`, `failure` status, and various SMART attributes (e.g., `smart_1_normalized`, `smart_1_raw`, etc.)

### Why Choose This Dataset?

Autoencoders are a powerful tool for unsupervised learning tasks like anomaly detection. The rich and structured hard drive health data allows us to explore how well autoencoders can learn to reconstruct input data and identify anomalies or patterns indicative of potential hard drive failures.

## Autoencoder Models

We developed and trained three different autoencoder models with various architectures to evaluate their performance on the anomaly detection task:

### Autoencoder 1

- **Encoder:**
  - Linear layer (input_dim → 64) → ReLU → Dropout (0.2)
  - Linear layer (64 → 32) → ReLU → Dropout (0.2)
- **Decoder:**
  - Linear layer (32 → 64) → ReLU → Dropout (0.2)
  - Linear layer (64 → input_dim) → Sigmoid

### Autoencoder 2

- **Encoder:**
  - Linear layer (input_dim → 128) → ReLU → BatchNorm
  - Linear layer (128 → 64) → ReLU → Dropout (0.2)
  - Linear layer (64 → 32) → ReLU
- **Decoder:**
  - Linear layer (32 → 64) → ReLU → Dropout (0.2)
  - Linear layer (64 → 128) → ReLU → BatchNorm
  - Linear layer (128 → input_dim) → Sigmoid

### Autoencoder 3

- **Encoder:**
  - Linear layer (input_dim → 32) → ReLU → Dropout (0.2)
  - Linear layer (32 → 16) → ReLU → Dropout (0.2)
  - Linear layer (16 → 8) → ReLU → Dropout (0.2)
- **Decoder:**
  - Linear layer (8 → 16) → ReLU → Dropout (0.2)
  - Linear layer (16 → 32) → ReLU → Dropout (0.2)
  - Linear layer (32 → input_dim) → Sigmoid

### Training Configuration

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Mean Squared Error (MSE)
- Epochs: 100
- Batch Size: 256

## Results

### Performance Metrics

- **Autoencoder 2 Metrics:**
  - Train Loss: 0.6261
  - Validation Loss: 0.6789
  - Test Loss: 0.6472
  - Anomaly Detection Accuracy: 89.13%

### Evaluation Metrics for All Models

| Model            | Mean Error | Median Error | Max Error | Min Error | Std. Deviation | Anomaly Detection Accuracy |
|------------------|------------|--------------|-----------|-----------|----------------|-----------------------------|
| Autoencoder 1    | 0.6521     | 0.1048       | 3881.413  | 0.0386    | 24.4024        | 89.19%                      |
| Autoencoder 2    | 0.6472     | 0.1047       | 3866.083  | 0.0385    | 24.2727        | 89.13%                      |
| Autoencoder 3    | 0.6879     | 0.1388       | 3877.990  | 0.0397    | 24.4179        | 91.48%                      |

### Summary

- Autoencoder 3 achieved the highest anomaly detection accuracy at **91.48%**, followed by Autoencoder 1 and Autoencoder 2.
- The models effectively detected anomalies based on reconstruction errors, with Autoencoder 3 showing the most promise in identifying outliers.

## Strengths and Limitations

### Strengths

- **Captures Complex Patterns:** Autoencoders can identify anomalies beyond linear relationships.
- **No Need for Labels:** Suitable for unsupervised learning where labeled data is scarce.
- **Feature Learning:** Learns compressed representations, potentially revealing hidden patterns.
- **Flexible Architecture:** Adaptable to different data types and anomaly detection tasks.
- **Noise Resilience:** Robust to background noise and minor data variations.

### Limitations

- **Reconstruction Challenges:** May struggle with reconstructing normal data, leading to false positives or missed anomalies.
- **Rarity of Anomalies:** Might overlook rare anomalies if they are not well-represented in the training data.
- **Data Bias:** Model performance can be biased if the training data is not representative.
- **Interpretability:** Hard to interpret the learned latent space and understand why certain data points are flagged as anomalies.
- **Computational Complexity:** Training on large datasets can be resource-intensive and time-consuming.

## Getting Started

To get started with this project, clone the repository and follow the instructions in the `requirements.txt` file to set up the environment.

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Code

You can find the code for the entire project in the `Autoencoders_for_Anomaly_Detection.ipynb` file.

## Detailed Report

A comprehensive report detailing the model architecture, training process, evaluation metrics, and performance graphs is available in `Autoencoders for Anomaly Detection report.pdf`.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!
