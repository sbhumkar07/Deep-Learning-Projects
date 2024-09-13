# ResNet18 CNN Image Classification

This project implements a ResNet-18-based Convolutional Neural Network (CNN) for multi-class image classification. The model incorporates various techniques such as regularization, dropout, and early stopping to improve performance and generalization on unseen data.

## Model Overview

### ResNet-18 Architecture

The ResNet-18 architecture is designed using a modular building block, called `BasicBlock`, which consists of two convolutional layers with batch normalization and ReLU activation functions, and a shortcut connection to help maintain input-output dimensions.

#### Basic Block

- **Convolutional Layers**: Each `BasicBlock` contains two convolutional layers with batch normalization and ReLU activation.
- **Shortcut Connection**: Maintains the dimensions of input and output, enabling the learning of residual functions.

#### ResNet-18 Model

- The `ResNet18` class constructs the overall model using `BasicBlock` components.
- The architecture starts with an initial convolutional layer followed by batch normalization and ReLU activation.
- It consists of four stages, each containing multiple layers of `BasicBlock`.
  - The number of blocks per stage is defined by `[2, 2, 2, 2]`.
  - The number of output channels doubles, while the spatial dimensions reduce by half with each subsequent stage (except for the first stage).
- An adaptive average pooling layer reduces the spatial dimensions of the output to a fixed size.
- A fully connected layer generates the final output logits.

### ResNet18_custom Function

- This function creates an instance of the ResNet-18 model with the specified configuration, using `BasicBlock` and `[2, 2, 2, 2]` to define the number of blocks for each stage.

## Impact of Techniques on Model Performance

### 1. Regularization

Regularization (L2 regularization or weight decay) is used to prevent overfitting by adding a penalty term to the loss function. 

- **Base Model**: Shows signs of overfitting, with increasing training accuracy and stagnating or slightly decreasing validation/test accuracies.
- **With Regularization**: The gap between training and validation/test accuracies is reduced, indicating better generalization.

### 2. Dropout

Dropout randomly drops neurons during training to prevent co-adaptation of feature detectors and improve model robustness.

- **Effect**: Helps in stabilizing or improving validation and test accuracies over epochs by preventing over-reliance on specific features or neurons.

### 3. Early Stopping

Early stopping halts training when validation performance degrades, preventing overfitting.

- **Base Model**: Training accuracy increases continuously, while validation accuracy stagnates or decreases, indicating overfitting.
- **With Early Stopping**: Training stops after six epochs, preventing excessive overfitting and improving generalization.

### Summary

Regularization, dropout, and early stopping collectively enhance the model's performance by reducing overfitting and improving generalization to unseen data.

## Model Evaluation

The best model achieved the following results:

- **Training Loss**: 0.3521
- **Training Accuracy**: 86.48%
- **Validation Loss**: 0.3312
- **Validation Accuracy**: 87.87%
- **Testing Loss**: 0.3229
- **Testing Accuracy**: 87.93%

### Graphs and Visualizations

- **Training and Validation Accuracy Over Epochs**: A plot displaying changes in accuracy during training and validation.
- **Training and Validation Loss Over Epochs**: A plot showing the loss over time, highlighting model convergence.

### Additional Evaluation Metrics

- **Precision**: 0.8884
- **Recall**: 0.8793
- **F1 Score**: 0.8801

## Usage

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Code

You can find the code for the entire project in the `ResNet18_CNN_Image_Classification.ipynb` file.

## Detailed Report

A comprehensive report detailing the model architecture, training process, evaluation metrics, and performance graphs is available in `report.pdf`.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!
