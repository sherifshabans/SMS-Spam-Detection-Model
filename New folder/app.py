import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('spam.csv', encoding='latin-1')

print(df.head())
print("Columns:", df.columns)
print("Class distribution:\n", df['v1'].value_counts())

# Preprocess the Data
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Using CountVectorizer to convert text data to a format suitable for machine learning
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])

# Step 5: Undersample the Majority Class (hum)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, df['v1'])

# Split into Training (70%) and Temporary Data (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Split Temporary Data into Validation (50%) and Test (50%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the Model on Validation Set
y_val_pred = model.predict(X_val)

# Corrected part: Use vectorizer.get_feature_names_out() for feature names
plt.figure(figsize=(18, 12))
plot_tree(model, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=['non-spam', 'spam'], rounded=True)
#plt.show()

# Export the decision tree to a Graphviz file
dot_data = export_graphviz(model, out_file=None,
                           feature_names=vectorizer.get_feature_names_out(),
                           class_names=['non-spam', 'spam'],
                           filled=True, rounded=True, special_characters=True)

# Visualize the Graphviz file using the graphviz library
graph = graphviz.Source(dot_data)
graph.render("spam_decision_tree", format="png")
graph.view("spam_decision_tree")

# Metrics for Validation Set
print("Accuracy on Validation Set:", accuracy_score(y_val, y_val_pred))

# Evaluate the Model on Test Set
y_test_pred = model.predict(X_test)

# Metrics for Test Set
print("Accuracy on Test Set:", accuracy_score(y_test, y_test_pred))
