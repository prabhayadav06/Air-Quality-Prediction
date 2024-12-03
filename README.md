# Air Quality Prediction

This project aims to analyze and predict air quality using machine learning techniques. By leveraging data preprocessing, exploratory data analysis (EDA), and model training, the project provides insights into air quality levels and predicts the Air Quality Index (AQI).

## Features
- **Exploratory Data Analysis (EDA):** Includes data visualization and statistical summaries to understand data trends and relationships.
- **Machine Learning Models:** Employs Random Forest Regressor and Polynomial Regression for AQI prediction.
- **Custom AQI Calculation:** Uses individual pollutant indices (IPIs) to calculate AQI based on pollutants such as PM2.5, PM10, NO2, CO, and others.
- **Data Visualization:** Provides scatter plots, box plots, and time series plots to visualize pollutant distributions and their correlation with AQI.
- **Model Persistence:** Saves trained models using pickle for future predictions.

## Dataset
The dataset used in this project contains daily air quality metrics for multiple cities. Key features include:
- Pollutants: PM2.5, PM10, NO, NO2, CO, Benzene, SO2, and more.
- City names and corresponding AQI values.
- Date of observation.

## Workflow
1. **Data Preprocessing:**
   - Handles missing values by filling them with zero.
   - Converts columns to appropriate data types.
2. **Exploratory Data Analysis (EDA):**
   - Generates statistical summaries.
   - Visualizes distributions and relationships using Seaborn and Matplotlib.
3. **Feature Selection:**
   - Identifies significant features contributing to AQI prediction.
4. **Model Training:**
   - Trains models using Random Forest and Polynomial Regression.
5. **Model Evaluation:**
   - Assesses model performance using metrics like RMSE and R^2 scores.
6. **AQI Prediction:**
   - Predicts AQI values based on pollutant levels.

## Results
- **Correlation Analysis:** Highlights relationships between pollutants and AQI.
- **Visualization Insights:** Demonstrates pollutant impact through various plots.
- **Model Accuracy:** Evaluates prediction accuracy using test data.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Air_Quality_prediction.ipynb
   ```

## Usage
- Load the dataset using the provided code snippets.
- Execute the notebook cells to perform analysis and train models.
- Save or load trained models for future predictions.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure
- **`Air_Quality_prediction.ipynb`**: Main notebook containing the code and explanations.
- **`random_forest_model.pkl`**: Pre-trained Random Forest model (generated during execution).
- **Data Folder:** Contains the air quality dataset (`city_day.csv`).

## Future Improvements
- Integrate the model with a web or mobile application for real-time AQI predictions.
- Enhance model accuracy with additional data and feature engineering.
- Implement advanced machine learning techniques, such as deep learning models.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
Special thanks to public datasets and libraries that made this project possible.



