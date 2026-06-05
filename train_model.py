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
