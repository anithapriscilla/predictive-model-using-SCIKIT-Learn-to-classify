import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
file_path = "C:\\Users\\ASUS\\Desktop\\spam.csv"  # Update this path if needed
try:
    df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\spam.csv", encoding="latin-1")
    print("✅ File found and loaded successfully!")
except FileNotFoundError:
    print("❌ Error: File not found. Check the path and filename.")
    exit()

# *Check dataset structure*
print("📌 Dataset Info:")
print(df.info())
print("📌 First few rows:\n", df.head())

# *Fix column names if needed*
if df.shape[1] > 2:
    df = data.iloc[:, :2]  # Only keep first 2 columns (label, message)
    df.columns = ["label", "message"]

# *Check if any missing data*
print("\n📌 Missing Values:\n", df.isnull().sum())

# *Convert labels to numeric values (ham = 0, spam = 1)*
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print("\n📌 Unique Labels (After Mapping):", df["label"].unique())

# *Ensure no invalid labels exist*
if df["label"].isnull().any():
    print("❌ Error: Some labels were not mapped correctly!")
    exit()

# *Split into train and test sets*
X = df['message']  # Assuming 'message' is the input feature (text data)
y = df['label']    # Assuming 'label' is the target (spam/ham classification)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

# *TF-IDF Vectorization*
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# *Check if TF-IDF worked correctly*
print("📌 X_train_tfidf shape:", X_train_tfidf.shape)
print("📌 X_test_tfidf shape:", X_test_tfidf.shape)

# *Check for empty training data*
if X_train_tfidf.shape[0] == 0:
    print("❌ Error: No training data found after vectorization!")
    exit()

# *Train the Naive Bayes Model*
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

# *Make Predictions*
y_pred = model.predict(X_test_tfidf)
print("Predicted labels:", np.unique(y_pred, return_counts=True))

# *Evaluate Model*
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", accuracy)
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred,zero_division=0))

# *Plot Confusion Matrix*
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
