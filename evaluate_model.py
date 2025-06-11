from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")
