# VGG13 CNN Image Classification

A deep learning project utilizing a VGG13-based Convolutional Neural Network (CNN) for multi-class image classification. The model is trained and evaluated using various techniques such as regularization, dropout, image augmentation, and early stopping to optimize its performance and generalization capability.

## Model Overview

The CNN model is a variant of the VGG architecture, specifically VGG13, and comprises several layers designed for feature extraction and classification.

### 1. Input Layer
- Accepts 3-channel images (assuming RGB images with channels for red, green, and blue).

### 2. Feature Extraction (Convolutional Layers)
1. Convolutional layer with 64 filters, kernel size of 3x3, padding of 1, followed by ReLU activation.
2. Convolutional layer with 64 filters, kernel size of 3x3, padding of 1, followed by ReLU activation.
   - Max-pooling layer with kernel size 2x2 and stride of 2.
3. Convolutional layer with 128 filters, followed by ReLU activation.
4. Convolutional layer with 128 filters, followed by ReLU activation.
   - Max-pooling layer.
5. Convolutional layer with 256 filters, followed by ReLU activation.
6. Convolutional layer with 256 filters, followed by ReLU activation.
   - Max-pooling layer.
7. Convolutional layer with 512 filters, followed by ReLU activation.
8. Convolutional layer with 512 filters, followed by ReLU activation.
   - Max-pooling layer.
9. Convolutional layer with 512 filters, followed by ReLU activation.
10. Convolutional layer with 512 filters, followed by ReLU activation.
    - Max-pooling layer.

### 3. Fully Connected Layers (Classifier)
- Flatten the output from the convolutional layers to a 1D tensor.
11. Fully connected layer with 4096 neurons, followed by ReLU activation and a dropout layer for regularization.
12. Fully connected layer with 4096 neurons, followed by ReLU activation and a dropout layer.
13. Output layer with 1000 neurons (adjustable for different numbers of classes).

### 4. Output
- The final output is determined by the last fully connected layer. The model can be adapted for different classification tasks by modifying the `num_classes` parameter.

## Impact of Regularization Techniques

1. **Base Model Accuracy: 76.1%**
   - Initial VGG13 model without any additional techniques.

2. **Base Model + L1 Regularization + Dropout Layer: 82.29% Accuracy**
   - **L1 Regularization**: Adds a penalty to the loss function based on the absolute values of weights, helping to prevent overfitting.
   - **Dropout Layer**: Randomly drops neurons during training, introducing noise to prevent co-adaptation of hidden units and improve generalization.

3. **Base Model + L1 Regularization + Dropout Layer + Image Augmentation + Early Stopping: 83.27% Accuracy**
   - **Image Augmentation**: Applies random transformations (rotation, flipping, zooming) to create a more diverse training dataset, enhancing generalization.
   - **Early Stopping**: Stops training once validation performance stops improving, preventing overfitting.

## Model Evaluation

The best-performing model achieved the following metrics:

- **Training Accuracy**: 82.10%
- **Training Loss**: 0.4709
- **Validation Accuracy**: 83.02%
- **Validation Loss**: 0.4416
- **Testing Accuracy**: 83.18% (3rd Epoch)
- **Testing Loss**: 0.4338 (3rd Epoch)

### Graphs and Visualizations

- **Training and Validation Accuracy Over Epochs**: A plot showing the changes in accuracy during training and validation.
- **Training and Validation Loss Over Epochs**: A plot showing the loss over time, highlighting model convergence.

### Additional Evaluation Metrics

- **Precision**: 0.8386
- **Recall**: 0.8278
- **F1 Score**: 0.8292

## Usage

To run this project, follow these steps:

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Code

You can find the code for the entire project in the `VGG_Classification_CNN.ipynb` file.

## Detailed Report

A comprehensive report detailing the model architecture, training process, evaluation metrics, and performance graphs is available in `report.pdf`.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!
