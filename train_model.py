# STEP-3

import joblib
from sklearn.naive_bayes import MultinomialNB

def train_model():
    print("-" * 50)
    print("STEP 3: TRAINING THE MACHINE LEARNING MODEL")
    print("-" * 50)

    # Load the now perfectly clean data
    X_train_numeric = joblib.load('X_train_numeric.pkl')
    y_train = joblib.load('y_train.pkl')
    
    print("Initializing the Multinomial Naive Bayes Algorithm...")
    model = MultinomialNB()

    print("Training in progress... (This should be INSTANT!)")
    model.fit(X_train_numeric, y_train)
    
    # Save the trained model
    joblib.dump(model, 'career_model.pkl')
    
    print("\nSuccess: Model successfully trained and saved. Ready for Step 5!")

if __name__ == "__main__":
    train_model()

# import joblib
# from sklearn.naive_bayes import MultinomialNB

# def train_model():
#     print("-" * 50)
#     print("STEP 3: TRAINING THE MACHINE LEARNING MODEL")
#     print("-" * 50)

#     # 1. Load the training data we prepared in Step 2
#     # Remember: X represents our features (numbers), y represents our labels (Career names)
#     try:
#         X_train_numeric = joblib.load('X_train_numeric.pkl')
#         y_train = joblib.load('y_train.pkl')
#     except FileNotFoundError:
#         print("Error: Could not find training data. Did you run Step 2?")
#         return

#     # Show dataset statistics (helps identify career imbalance issues)
#     print(f"\nTraining Samples: {len(y_train)}")
#     print(f"Unique Careers : {y_train.nunique()}")

#     print("\nCareer Distribution:")
#     print(y_train.value_counts().head(10))

#     # 2. Initialize the Algorithm
#     # We use Multinomial Naive Bayes because it works very well for text classification
#     # tasks using TF-IDF vectors and handles many classes better than Random Forest.
#     print("\nInitializing Naive Bayes Algorithm...")
#     model = MultinomialNB()

#     # 3. Train the Model (The "Fit" step)
#     # This is where the actual Machine Learning happens! The algorithm looks at the numerical inputs (X) and the correct answers 
#     # (y) and tries to find mathematical rules mapping one to the other.
#     print("Training in progress... The model is finding patterns in the data.")
#     model.fit(X_train_numeric, y_train)

#     # 4. Save the trained model
#     # We save the model so we don't have to retrain it every time a user wants a prediction.
#     joblib.dump(model, 'career_model.pkl')

#     print("\nSuccess: Model successfully trained and saved as 'career_model.pkl'.")

# if __name__ == "__main__":
#     train_model()