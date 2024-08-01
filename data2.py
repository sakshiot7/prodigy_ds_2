import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_df = pd.read_csv('C:/Users/kanch/Downloads/titanic.csv')
# Display the first few rows of the dataset
print(titanic_df.head())
# Check the dimensions of the dataset
print("Dimensions of Titanic dataset:", titanic_df.shape)

# Get a summary of the numerical attributes
print(titanic_df.describe())

# Check for missing values
print("Missing values:")
print(titanic_df.isnull().sum())


# Assume titanic_df is already loaded with the Titanic dataset

# Get the mode of 'Embarked'
mode_embarked = titanic_df['Embarked'].mode()[0]

# Use loc or direct assignment to set values
titanic_df.loc[titanic_df['Embarked'].isnull(), 'Embarked'] = mode_embarked
# Or alternatively
# titanic_df['Embarked'] = titanic_df['Embarked'].fillna(mode_embarked)

# Drop 'Cabin' column if it's not relevant
titanic_df.drop('Cabin', axis=1, inplace=True)

# Print a summary of missing values after handling
print("Missing values after handling:")
print(titanic_df.isnull().sum())


# Countplot of survival
sns.countplot(x='Survived', data=titanic_df)
plt.title('Count of Passengers by Survival')
plt.show()

# Survival rate by sex
sns.barplot(x='Sex', y='Survived', data=titanic_df, ci=None)
plt.title('Survival Rate by Sex')
plt.show()

# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=titanic_df, ci=None)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival rate by age
sns.histplot(data=titanic_df, x='Age', hue='Survived', kde=True)
plt.title('Survival Distribution by Age')
plt.show()
# Relationship between passenger class and fare
sns.boxplot(x='Pclass', y='Fare', data=titanic_df)
plt.title('Fare Distribution by Passenger Class')
plt.show()

# Pairplot for numerical variables
sns.pairplot(titanic_df[['Survived', 'Age', 'Fare', 'Pclass']])
plt.title('Pairplot of Numerical Variables')
plt.show()

# Correlation heatmap
corr_matrix = titanic_df[['Survived', 'Age', 'Fare', 'Pclass']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

