import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = "C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\tested.csv"

df = pd.read_csv(file_path)

# Handle missing values
df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())
df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].median())
df.drop(columns=['Cabin'], inplace=True)  # Too many missing values

# Extract titles from names
df['Title'] = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=True)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Drop unnecessary columns
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Save preprocessed dataset
df.to_csv("C:\\Users\\vikhy\\OneDrive\\Desktop\\Titanic-Survival-Prediction\\Data\\titanic_preprocessed.csv", index=False)

print("Preprocessing complete. Processed data saved as titanic_preprocessed.csv.")
