# RNN Time Series Forecasting

This project implements a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers for time series forecasting. The model is designed to predict household electricity consumption using the "Individual Household Electric Power Consumption" dataset from the UCI Machine Learning Repository.

## Dataset Overview

The dataset captures electric power consumption data from a single household over nearly four years, with measurements taken at a one-minute sampling rate. It includes key parameters such as:

- **Voltage Variations**: Changes in voltage over time.
- **Global Active Power**: Overall power consumption in kilowatts.
- **Global Reactive Power**: Reactive power in kilowatts.

The dataset contains granular minute-by-minute entries that provide detailed insights into household electricity usage patterns over time.

## RNN Model Architecture

### Overview

The RNN model used in this project consists of three LSTM layers, followed by fully connected layers. Below is a brief overview of the architecture:

1. **LSTM Layers**:
   - **LSTM Layer 1**: Input size = `input_size`, hidden size = `hidden_size`, `bidirectional=True`.
   - **LSTM Layer 2**: Input size = `hidden_size * 2` (to account for the bidirectional setting), hidden size = `hidden_size`, `bidirectional=True`.
   - **LSTM Layer 3**: Input size = `hidden_size * 2`, output size = 32, `bidirectional=True`.

2. **Fully Connected Layers**:
   - `fc1`: Linear layer with input size `32 * 2` and output size `128`.
   - `fc2`: Linear layer with input size `128` and output size `64`.
   - `fc3`: Linear layer with input size `64` and output size `32`.
   - `fc4`: Linear layer with input size `32` and output size `1`.

3. **Activation and Dropout**:
   - ReLU activation is applied after each fully connected layer.
   - Dropout with a probability of 0.25 is used after each ReLU activation to prevent overfitting.

4. **Forward Pass**:
   - The input tensor `x` passes through the three LSTM layers.
   - The output of the last time step is selected (`out[:, -1, :]`).
   - The output then passes through the fully connected layers with ReLU activation and dropout.
   - A final sigmoid function is applied to produce an output value between 0 and 1.

5. **Model Initialization**:
   - Input size is inferred from the training data shape.
   - Hidden size is set to 128, with one layer of LSTM used.

## Model Performance and Evaluation

The model addresses a regression problem, and we evaluated its performance using Loss and Mean Absolute Error (MAE).

### Results

- **Training Loss**: 0.0640
- **Training MAE**: 0.0640
- **Validation Loss**: 0.1334
- **Validation MAE**: 0.1334
- **Testing Loss**: 0.0960
- **Testing MAE**: 0.0960

### Graphs and Visualizations

- **Mean Absolute Error (MAE) Over Time (Epochs)**: Plotted to visualize the model's performance during training and validation.
- **Loss Over Time (Epochs)**: Plotted to show the convergence of the model during training.

### Confusion Matrix

A confusion matrix is not applicable for this regression task. Instead, the model was evaluated using Mean Absolute Error (MAE).

### Additional Evaluation Metrics

- **Training MAE**: 0.0640
- **Validation MAE**: 0.1334
- **Testing MAE**: 0.0960

## Usage

To run this project, follow these steps:

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Code

You can find the code for the entire project in the `RNN_Time_Series_Forecasting.ipynb` file.

## Detailed Report

A comprehensive report detailing the model architecture, training process, evaluation metrics, and performance graphs is available in `Time-Series Forecasting report.pdf`.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!
