# Student Weight Status Prediction

A deep learning project using various machine learning models to predict weight status categories of students based on regional, county, and area data from the Student Weight Status Category Reporting System (SWSCR).

## Dataset Description

The dataset contains 32,025 entries with 15 columns, consisting of both numerical and categorical data. Key features include:

- **Numerical Features**: `Location Code`, `Number Overweight`, `Percent Overweight`, `Number Obese`, `Percent Obese`, `Number Overweight or Obese`, `Percent Overweight or Obese`, `Number Healthy Weight`, `Percent Healthy Weight`.
- **Categorical Features**: `County`, `Area Name`, `Region`, `Grade Level`, `Sex`.

## Data Preprocessing

Several preprocessing techniques were applied to clean and prepare the data for modeling:

1. **Handling Missing Values**: Rows with missing values were removed, and the 'Region' column was dropped due to a significant number of null values.
2. **Outlier Detection and Handling**: Used z-score thresholding to detect and remove outliers.
3. **Visualization**:
   - Correlation matrix
   - Pairplot for numerical features
   - Bar plot for sample counts by sex
   - Distribution plot of 'Percent Overweight'
   - Boxplot of 'Number Overweight' by 'Grade Level'
4. **Feature Selection**: Dropped unrelated or uncorrelated features based on the correlation matrix.
5. **One-Hot Encoding**: Converted categorical variables (`Grade Level`, `Sex`) to numerical values.
6. **Normalization**: Applied Min-Max scaling to numerical features.

## Visualizations

1. **Correlation Matrix**: Revealed strong positive correlations between variables like 'Number Overweight' and 'Number Obese', and negative correlations with 'Percent Healthy Weight'.
2. **Pairplot of Numeric Features**: Showed positive correlations between overweight and obesity, with no clear pattern between healthy weight and overweight/obesity.
3. **Bar Plot of Count of Samples by Sex**: Indicated a balanced representation of male and female samples.
4. **Distribution Plot of Percent Overweight**: Demonstrated a normal distribution with a peak around 15% overweight.
5. **Boxplot of Number Overweight by Grade Level**: Illustrated the variability in the number of overweight individuals across different grade levels.

## Machine Learning Models

Three machine learning models were implemented to predict the target variable (`Percent Healthy Weight`):

1. **Random Forest Regression**: An ensemble learning method that builds multiple decision trees and averages their predictions.
2. **Gradient Boosting Regression**: Another ensemble method that builds a series of weak learners to minimize a loss function.
3. **Support Vector Regression (SVR)**: Utilizes support vector machines for regression tasks by mapping input data to high-dimensional space and finding the optimal hyperplane.

## Model Evaluation

- **Testing Set Performance:**
  - **Random Forest Regression**: MSE = 0.00119, R² = 0.9509
  - **Gradient Boosting Regression**: MSE = 0.00249, R² = 0.8976
  - **Support Vector Regression (SVR)**: MSE = 0.00741, R² = 0.6952

- **Validation Set Performance:**
  - **Random Forest Regression**: MSE = 0.00102, R² = 0.9568
  - **Gradient Boosting Regression**: MSE = 0.00226, R² = 0.9041
  - **Support Vector Regression (SVR)**: MSE = 0.00687, R² = 0.7091

## Key Findings

- **Random Forest Regression** demonstrated the highest accuracy and lowest error, making it the best-performing model for this dataset.
- **Gradient Boosting Regression** also performed well but required careful tuning of hyperparameters.
- **Support Vector Regression** was less effective due to its sensitivity to kernel choice and parameters.

## Conclusion

The project provides valuable insights into the weight status of students across different demographics and demonstrates the use of machine learning models for predictive analysis. Further improvements could involve exploring additional models, hyperparameter tuning, or incorporating more features.

## Usage

To run this project, follow these steps:

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to view the analysis and model training.

## Code

You can find the code for the entire project in the `Student_Weight_Status_Prediction.ipynb` file.

## Detailed Report

A comprehensive report detailing the model architecture, training process, evaluation metrics, and performance graphs is available in `report.pdf`.

## Contact

For questions or feedback, please <a href="mailto:sarveshbhumkar27@gmail.com" target="_blank">Email</a>. Thank you!
