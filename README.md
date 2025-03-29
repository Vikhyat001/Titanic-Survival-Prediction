# Titanic-Survival-Prediction

# Overview
This project uses machine learning to predict the survival of passengers on the Titanic based on various features such as age, sex, fare, and passenger class. The dataset is taken from the famous Titanic dataset provided by Kaggle.

 # Project Steps

1. Data Preprocessing
   - Handling missing values.
   - Encoding categorical variables.
   - Feature scaling and engineering.

2. Data Splitting
   - Splitting the dataset into training and testing sets.

3. Model Training
   - Training an XGBoost model to predict survival.

4. Making Predictions
   - Using the trained model to predict survival on test data.

5. Model Evaluation
   - Assessing performance using accuracy, precision, recall, and F1-score.

# Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Seaborn & Matplotlib

# File Structure
- Data/: Contains raw and processed datasets.
- Preprocess.py: Cleans and preprocesses the dataset.
- split_data.py: Splits the dataset into training and test sets.
- train_model.py: Trains the XGBoost model.
- predict.py: Makes predictions using the trained model.
- Model/: Stores the trained model file.

# How to Run
1. Clone the repository:
   git clone https://github.com/Vikhyat001/Titanic-Survival-Prediction.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the scripts in order:
   python Preprocess.py
   python split_data.py
   python train_model.py
   python predict.py

  Results
- The model provides insights into which factors were most influential in survival.
- Evaluation metrics such as accuracy, precision, recall, and F1-score indicate the model's performance.

