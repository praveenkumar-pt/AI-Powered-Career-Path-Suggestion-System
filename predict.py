# STEP-5

import joblib
import numpy as np

def predict_career():
    # 1. Load our saved tools: Model, Translator, and Cluster Dictionary
    try:
        model = joblib.load('career_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        cluster_dict = joblib.load('career_to_cluster.pkl')
    except FileNotFoundError:
        print("Error: Missing files. Please run the previous steps first.")
        return

    # 2. Get user input
    user_input = input("Input Skills : ")
    
    if not user_input.strip():
        print("You didn't enter any skills!")
        return

    # 3. Translate user text into math
    user_numeric = vectorizer.transform([user_input])

    # 4. Predict probabilities for all careers
    probabilities = model.predict_proba(user_numeric)[0]
    career_names = model.classes_
    
    # 5. Sort to find the Top 4 Matches (1 Best Role + 3 Suggested Roles)
    # We use [:4] here to make sure we grab exactly 4 items
    top_indices = np.argsort(probabilities)[::-1][:4]
    
    # 6. Extract the Best Match and its Cluster
    best_match_idx = top_indices[0]
    best_career = career_names[best_match_idx]
    
    # Look up the cluster using the dictionary we saved in Step 2
    best_cluster = cluster_dict.get(best_career, "Unknown Category")
    
    # 7. Print the exact requested output
    print("\nCareer Category:")
    print(best_cluster)
    
    print("\nBest Role:")
    print(best_career)
    
    print("\nSuggested Roles:")
    # Loop through the remaining 3 backup predictions (indices 1, 2, and 3)
    for i in range(1, len(top_indices)):
        alt_idx = top_indices[i]
        
        # Ensure we only show suggestions if the AI has at least *some* confidence
        if probabilities[alt_idx] > 0:
            alt_career = career_names[alt_idx]
            print(f"• {alt_career}")

if __name__ == "__main__":
    predict_career()

