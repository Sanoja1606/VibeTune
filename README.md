# VibeTune

VibeTune is a machine learning-based emotion-to-music genre recommendation system. It analyzes user input to detect the underlying emotion and suggests a suitable music genre based on that emotion.

# Features

- Text preprocessing and emotion classification using an ensemble of ML models
- Genre recommendation based on detected emotions
- Supports multiple emotions: joy, sadness, anger, love, fear, surprise, etc.
- Clean, modular codebase suitable for both research and real-world applications

# How It Works

1. User inputs a sentence describing their mood or feeling.
2. The model predicts the emotional tone of the input using an ensemble of:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest Classifier
3. Based on the predicted emotion, a corresponding music genre is recommended.

# Model Performance

The model achieved a classification accuracy of **87.30%** on the test dataset.

# Sample Classification Report:
![Classification Report and Test Accuracy](https://github.com/user-attachments/assets/8782c699-f258-4f1e-bd55-c294ad2e7369)

# Working Sample
![Screenshot of Project Working](https://github.com/user-attachments/assets/aee9c0bb-b36f-4ac0-928a-5dc791f4c995)


