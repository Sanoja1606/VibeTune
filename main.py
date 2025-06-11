import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from evaluate_model import evaluate_model  

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_txt_dataset(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(";")
            if len(parts) == 2:
                texts.append(clean_text(parts[0]))
                labels.append(parts[1])
    return np.array(texts), np.array(labels)

genre_map = {
    "happy": "Pop / EDM",
    "sad": "Acoustic / Blues",
    "joy": "Pop / Funk",
    "anger": "Metal / Rock",
    "love": "R&B / Soul",
    "fear": "Ambient",
    "surprise": "Electronic",
    "sadness": "Acoustic / Blues"
}

def recommend_music(text, model):
    text = clean_text(text)
    emotion = model.predict([text])[0]
    genre = genre_map.get(emotion, "Ambient")
    print(f"\nPredicted Emotion: {emotion}")
    print(f"Recommended Music Genre: {genre}")

if __name__ == "__main__":
    X_train, y_train = load_txt_dataset("emotion_dataset/train.txt")
    X_test, y_test = load_txt_dataset("emotion_dataset/test.txt")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    clf1 = LogisticRegression(max_iter=1000, C=1.5)
    clf2 = LinearSVC(C=1.5)
    clf3 = RandomForestClassifier(n_estimators=100, random_state=42)

    ensemble_model = make_pipeline(
        tfidf,
        VotingClassifier(
            estimators=[
                ("lr", clf1),
                ("svc", clf2),
                ("rf", clf3)
            ],
            voting="hard"
        )
    )

    ensemble_model.fit(X_train, y_train)

    user_input = input("\nEnter your feeling/mood: ")
    recommend_music(user_input, ensemble_model)

    evaluate_model(ensemble_model, X_test, y_test)  
