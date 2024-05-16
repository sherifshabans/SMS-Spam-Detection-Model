import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from id3 import Id3Estimator
from id3 import export_graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns

# Load and inspect the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

print(df.head())
print("Columns:", df.columns)
print("Class distribution:\n", df['v1'].value_counts())  # Values to see the spam and ham sum

# Preprocess the data
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Using CountVectorizer to convert text data to a format suitable for machine learning
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])  # Fit transform to messages v2 (X)

# Undersample the majority class (ham)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, df['v1'])
print("Class distribution after undersample:\n", y_resampled.value_counts())

# Split into Training (70%) and Temporary Data (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Split Temporary Data into Validation (50%) and Test (50%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Call the ID3 Algorithm and Train the Model
model = Id3Estimator()
model.fit(X_train.toarray(), y_train, check_input=True)

# Visualize the Decision Tree
export_graphviz(model.tree_, 'tree.dot', feature_names=vectorizer.get_feature_names_out())

# Metrics for Validation Set
X_val_arr = X_val.toarray()
y_val_pred = model.predict(X_val_arr)

print("Accuracy on Validation Set:", accuracy_score(y_val, y_val_pred))

# Evaluate the Model on Test Set
X_test_arr = X_test.toarray()
y_test_pred = model.predict(X_test_arr)

print("Accuracy on Test Set:", accuracy_score(y_test, y_test_pred))

# Custom scoring function
accuracy_scorer = make_scorer(accuracy_score)

# Cross-Validation
cv_scores = cross_val_score(model, X_resampled.toarray(), y_resampled, cv=5, scoring=accuracy_scorer)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
