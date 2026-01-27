# Step 2 alternative: Bigger dataset using dictionary
data_dict = {
    "label": [
        "ham","spam","ham","spam","ham","spam","ham","spam","ham","spam",
        "ham","spam","ham","spam","ham","spam","ham","spam","ham","spam",
        "ham","spam","ham","spam","ham","spam","ham","spam","ham","spam"
    ],
    "message": [
        "Hey, are you free tonight?",
        "Congratulations! You won a lottery. Claim now!",
        "Let's meet for lunch tomorrow",
        "Earn $1000 daily from home, click here!",
        "Did you finish your homework?",
        "Win a free iPhone now, limited time offer!",
        "Are you coming to the party?",
        "You have been selected for a prize, click the link!",
        "Don't forget the meeting at 10 AM",
        "Get cheap loans instantly, no credit check!",
        "How was your day?",
        "Exclusive deal! Buy 1 get 1 free!",
        "Call me when you are free",
        "You have won a free vacation to Bahamas!",
        "Are you joining the group chat?",
        "Claim your $500 reward now, hurry!",
        "See you at the movie tonight",
        "Your account is suspended, click here to reactivate!",
        "Can you help me with this assignment?",
        "Earn money from home with this simple trick!",
        "Dinner at my place?",
        "Limited time offer! Buy now and save 50%",
        "Good morning! Have a nice day",
        "You won a free gift card, claim today!",
        "Let's go for a run",
        "Your subscription expires today, renew now!",
        "Are you free this weekend?",
        "Congratulations! You are selected for a cash prize!",
        "Can we talk later?",
        "Act fast! Exclusive deals just for you!"
    ]
}

import pandas as pd
data = pd.DataFrame(data_dict)

# Step 3: Encode labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Step 4: Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

# Step 5: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 7: Predict
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
