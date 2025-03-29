import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load training and testing datasets
X_train = pd.read_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\X_train.csv")
X_test = pd.read_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\X_test.csv")
y_train = pd.read_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\y_train.csv")
y_test = pd.read_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\y_test.csv")

# Convert target variable to a 1D array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train an XGBoost model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save the trained model
model_path = "C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Model\\titanic_model.pkl"
joblib.dump(model, model_path)

# Print evaluation results
print(f"Model training complete. Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Classification Report:\n", report)
print(f"Model saved at: {model_path}")

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
