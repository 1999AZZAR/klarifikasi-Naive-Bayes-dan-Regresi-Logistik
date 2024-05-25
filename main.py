import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV file
data = pd.read_csv('data.csv')

# Convert categorical 'Jenis Kelamin' column to numerical
data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'Laki-Laki': 0, 'Perempuan': 1})

# Split the data into features (X) and target (y)
X = data[['Umur', 'Jenis Kelamin', 'Pendapatan (Ribu Rupiah)']]
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

# Train a Logistic Regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Make predictions using both models
naive_bayes_predictions = naive_bayes_classifier.predict(X_test)
logistic_regression_predictions = logistic_regression_model.predict(X_test)

# Calculate accuracy for both models
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)

# Print the predictions and accuracies
print("Naive Bayes Predictions:")
print(naive_bayes_predictions)

print("\nLogistic Regression Predictions:")
print(logistic_regression_predictions)

print("\nNaive Bayes Accuracy:", naive_bayes_accuracy)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Naive Bayes predictions plot
sns.scatterplot(x=X_test['Umur'], y=X_test['Pendapatan (Ribu Rupiah)'], hue=naive_bayes_predictions, palette='coolwarm', ax=ax[0])
ax[0].set_title(f'Naive Bayes Predictions\nAccuracy: {naive_bayes_accuracy:.2f}')
ax[0].set_xlabel('Umur')
ax[0].set_ylabel('Pendapatan (Ribu Rupiah)')

# Logistic Regression predictions plot
sns.scatterplot(x=X_test['Umur'], y=X_test['Pendapatan (Ribu Rupiah)'], hue=logistic_regression_predictions, palette='coolwarm', ax=ax[1])
ax[1].set_title(f'Logistic Regression Predictions\nAccuracy: {logistic_regression_accuracy:.2f}')
ax[1].set_xlabel('Umur')
ax[1].set_ylabel('Pendapatan (Ribu Rupiah)')

plt.tight_layout()
plt.show()
