# STEP-4

import joblib
from sklearn.metrics import accuracy_score

def evaluate_model():
    print("-" * 50)
    print("STEP 4: EVALUATING MODEL ACCURACY")
    print("-" * 50)

    # 1. Load the trained model and the hidden Test Data
    try:
        model = joblib.load('career_model.pkl')
        X_test_numeric = joblib.load('X_test_numeric.pkl')
        y_test_real_answers = joblib.load('y_test.pkl')
    except FileNotFoundError:
        print("Error: Missing files. Ensure you have run Steps 2 and 3.")
        return

    print("Giving the model a 'final exam' using data it has never seen...")

    # 2. Make predictions on the test data
    predictions = model.predict(X_test_numeric)

    # 3. Calculate Accuracy
    accuracy = accuracy_score(y_test_real_answers, predictions)
    accuracy_percentage = round(accuracy * 100, 2)
    
    print(f"\nModel Accuracy: {accuracy_percentage}%")
    
    # Show a real comparison for the first 3 test items
    print("\n--- Let's look at 3 real test examples ---")
    real_answers_list = y_test_real_answers.tolist()
    
    for i in range(min(3, len(predictions))):
        print(f"Test #{i+1}:")
        print(f"  Model Predicted : {predictions[i]}")
        print(f"  Actual Answer   : {real_answers_list[i]}")
        if predictions[i] == real_answers_list[i]:
            print("  Result          : CORRECT")
        else:
            print("  Result          : INCORRECT")
            
    print("\nSuccess: Evaluation complete. Let's move to Step 5!")

if __name__ == "__main__":
    evaluate_model()
