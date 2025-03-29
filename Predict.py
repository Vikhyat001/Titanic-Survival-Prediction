import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load trained model
model_path = "C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Model\\titanic_model.pkl"
model = joblib.load(model_path)

# Load new test data
test_data_path = "C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\tested.csv"
df = pd.read_csv(test_data_path)

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')

# Ensure 'Survived' is not present (this should be the prediction target)
if 'Survived' in df.columns:
    df.drop(columns=['Survived'], inplace=True)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature Engineering: Create 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Load feature names from training data
train_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 
                  'Title_Dona', 'Title_Dr', 'Title_Master', 'Title_Miss', 'Title_Mr', 
                  'Title_Mrs', 'Title_Ms', 'Title_Rev', 'FamilySize']

# Ensure the test dataset has the same columns as the training dataset
for col in train_features:
    if col not in df.columns:
        df[col] = 0  # Add missing columns with default value

# Keep only the required feature columns
df = df[train_features]

# Make predictions
predictions = model.predict(df)

# Save predictions
output_path = "C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\predictions.csv"
pd.DataFrame(predictions, columns=['Survived']).to_csv(output_path, index=False)

print(f"Predictions saved at {output_path}")
