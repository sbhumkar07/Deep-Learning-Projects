# Transformer for Text Classification

This project implements a Transformer-based model for text classification on the AG's News dataset. The goal is to classify news articles into four distinct categories using a Transformer architecture. The model leverages techniques such as regularization, dropout, and early stopping to improve performance.

## Dataset Overview

The AG's News dataset is a popular benchmark dataset for text classification tasks. It consists of 4 classes of news articles:

- **Classes:** 4 (World, Sports, Business, Science/Technology)
- **Training Samples:** 120,000 (30,000 samples per class)
- **Testing Samples:** 7,600 (1,900 samples per class)

### Justification for Choosing This Dataset

1. **Large and Diverse Dataset:** With over 120,000 training samples and 7,600 testing samples, the dataset offers substantial data for training and evaluation across a wide range of news topics.
2. **Standard Benchmark:** The dataset is widely used in NLP research, providing a standard benchmark for evaluating the performance of different models, including Transformers.
3. **Real-World Applicability:** Text classification is a fundamental NLP task with applications in sentiment analysis, topic categorization, and spam detection. This project aims to develop practical NLP skills and insights applicable to real-world problems.

## Transformer Model Architecture

The model employs a Transformer-based architecture, which is defined as follows:

### Embedding Layer

- **Embedding Layer:** Converts input tokens into dense vectors of a fixed size (`embed_dim`). The weights are initialized using pre-trained GloVe word embeddings.

### Positional Encoding

- **Positional Encoding:** Adds positional information to the input vectors, allowing the model to capture the order of words in a sequence.

### Transformer Encoder

- **Encoder Layers:**
  - A stack of Transformer encoder layers (`nn.TransformerEncoderLayer`), each consisting of:
    - Multi-head self-attention mechanism
    - Position-wise feedforward networks
  - Configurable parameters include the number of encoder layers (`num_layers`), attention heads (`n_heads`), and the feedforward network dimension (`ff_dim`).

### Pooling and Output Layer

- **Mean Pooling:** Computes the mean of the encoded representations to obtain a fixed-size vector for the input sequence.
- **Output Layer:** Fully connected layer to produce logits for each class, followed by a log softmax activation to obtain class probabilities.

## Techniques for Improving Model Performance

The following techniques were applied to improve the model's performance:

1. **L2 Regularization:** Prevents overfitting by penalizing large weights, encouraging the model to generalize better.
2. **Dropout:** Randomly drops a fraction of neurons during training, reducing co-dependency among neurons and preventing overfitting.
3. **Early Stopping:** Stops training when the model's performance on the validation set starts to degrade, avoiding overfitting and ensuring optimal performance.

## Results

### Performance Metrics

- **Best Model (with L2 regularization, dropout, early stopping):**
  - Train Accuracy: **93.15%**
  - Test Accuracy: **88.29%**
  - Train Loss: **0.2005**
  - Validation Loss: **0.3947**
  - Validation Accuracy: **87.50%**
  - Test Loss: **0.3419**

- **Base Model:**
  - Train Accuracy: **89.76%**
  - Test Accuracy: **87.21%**

### Evaluation Metrics

- **Precision:** 0.8859
- **Recall:** 0.8834
- **F1 Score:** 0.8825

### Graphs and Visualizations

- **Training and Validation Accuracy Over Epochs:**
- **Training and Validation Loss Over Epochs:**
- **Confusion Matrix:**
- **ROC Curve:**

## Impact of Regularization, Dropout, and Early Stopping

The optimized model, which incorporates L2 regularization, dropout, and early stopping, outperforms the base model in both training and test accuracy:

- **L2 Regularization:** Prevents overfitting by penalizing large weights, allowing the model to generalize better to unseen data.
- **Dropout:** Helps improve generalization by reducing neuron co-dependency, leading to more robust feature learning.
- **Early Stopping:** Stops training when validation performance degrades, preventing overfitting and allowing the model to converge to a better solution.

## Getting Started

To get started with this project, clone the repository and set up the environment by installing the required packages.

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Code

You can find the code for the entire project in the `Transformer_for_Text_Classification.ipynb` file.

## Detailed Report

A comprehensive report detailing the model architecture, training process, evaluation metrics, and performance graphs is available in `Transformer for Text Classification report.pdf`.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!
