# %%
# Import necessary libraries
import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression

# %%
# Download NLTK data
nltk.download('stopwords')
#nltk.download('wordnet')    

# %%
# Step 1: Load the dataset
data = pd.read_csv('Spam_or_Ham\email_spam.csv', encoding='ISO-8859-1')  # Update file path as needed

# %%

# Step 2: Preprocessing Function
# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)


# %%

# Apply preprocessing to the text column
# Assuming the email text is in the column named 'v1' or 'v2' based on the global variable output.
# Change 'v1' to the actual column name if it's different.
data['cleaned_text'] = data['v2'].apply(preprocess_text) # Changed 'text' to 'v2' - Update to correct column name if needed. Print data.columns to confirm column names.


# %%

# Step 3: Prepare Features and Labels
texts = data['cleaned_text']  # Use cleaned text for modeling
labels = data['v1']  # Replace 'label' with the actual label column name - Assuming v1 contains the label


# %%

# Step 4: Vectorize the text data
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(texts)
#y = data['v1']

# %%

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.40, random_state=42)


# %% [markdown]
# 

# %%

# Step 6: Train the Decision Tree Classifier
model = MultinomialNB()
model.fit(X_train, y_train)


# %%

# Step 7: Make predictions
y_pred = model.predict(X_test)

# %%

# Step 8: Evaluate the model
print(f"Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# %%
# Step 9: Test the model with new data
new_messages = ["Congratulations, you won a free ticket!", "Hey, how are you doing today?", "Sunshine Quiz Wkly Q!", "You are fool"]
new_messages_cleaned = [preprocess_text(msg) for msg in new_messages]
new_vectors = vectorizer.transform(new_messages_cleaned)
new_predictions = model.predict(new_vectors)

print("New Predictions:", new_predictions)


# Step 9: Test the model with new data
new_messages = ["win cash prize worth $4000", "You're owed a refund!", "You are fool", "Sunshine Quiz Wkly Q!"]
new_vectors = vectorizer.transform(new_messages)  # Transform new messages to numerical vectors
new_predictions = model.predict(new_vectors)     # Predict spam/ham labels
new_texts = vectorizer.inverse_transform(new_vectors)  # Map vectors back to corresponding words

# Print predictions and the corresponding words
#print("New Predictions:", new_predictions)

for i, text in enumerate(new_texts):
    print(f"Message {i + 1}: Predicted as {new_predictions[i]}, Words: {' '.join(text)}")
    #print(f"Message {i + 1}:, Words: {' '.join(text)}, Predicted as {new_predictions[i]}")


