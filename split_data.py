import pandas as pd
from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
file_path = "C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\titanic_preprocessed.csv"
df = pd.read_csv(file_path)

# Define features and target variable
target = 'Survived'  # Ensure this column exists in your dataset
X = df.drop(columns=[target])
y = df[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets
X_train.to_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\X_train.csv", index=False)
X_test.to_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\X_test.csv", index=False)
y_train.to_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\y_train.csv", index=False)
y_test.to_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\y_test.csv", index=False)

print("Data split complete. Training and testing sets saved.")
