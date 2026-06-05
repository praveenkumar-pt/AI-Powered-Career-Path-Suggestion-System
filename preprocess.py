# STEP-2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def preprocess_data():
    print("-" * 50)
    print("STEP 2: PREPROCESSING (FIXING HIDDEN SPACES)")
    print("-" * 50)

    # 1. FORCE Python to load ONLY the new clustered file
    file_name = 'cleaned_careers_clustered.csv'
    
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_name}'. Please ensure it is in the folder.")
        return
    
    # --- THE FIX ---
    # Strip hidden whitespace from the beginning and end of all text
    df['Career'] = df['Career'].astype(str).str.strip()
    df['Cluster'] = df['Cluster'].astype(str).str.strip()
    df['Skill'] = df['Skill'].astype(str).str.strip()

    # Now that the spaces are gone, we can successfully delete "No Data"
    df = df[df['Career'] != 'No Data']
    df = df[df['Skill'] != 'No Data']
    # ---------------
    
    X = df['Skill']
    y = df['Career']
    
    # Create dictionary linking Career to Cluster
    career_to_cluster = dict(zip(df['Career'], df['Cluster']))
    joblib.dump(career_to_cluster, 'career_to_cluster.pkl')
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Translate Text to Math
    vectorizer = TfidfVectorizer()
    X_train_numeric = vectorizer.fit_transform(X_train)
    X_test_numeric = vectorizer.transform(X_test)
    
    # Save all files
    joblib.dump(X_train_numeric, 'X_train_numeric.pkl')
    joblib.dump(X_test_numeric, 'X_test_numeric.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(y_test, 'y_test.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    print("Success: Hidden spaces removed and 'No Data' deleted! Ready for Step 3.")

if __name__ == "__main__":
    preprocess_data()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# import joblib # Used to save our Python objects to files

# def preprocess_data():
#     print("-" * 50)
#     print("STEP 2: PREPROCESSING (TEXT TO NUMBERS)")
#     print("-" * 50)

#     # 1. Load the clean data from Step 1
#     df = pd.read_csv('cleaned_careers_clustered.csv')
    
#     # Separate our inputs (X) and the answers we want to predict (y)
#     X = df['Skill']
#     y = df['Career']
    
#     # 2. Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     print(f"Data split: {len(X_train)} rows for training, {len(X_test)} rows for testing.")

#     # 3. Initialize TF-IDF Vectorizer
#     # This is our "translator" that turns words into numerical arrays.
#     vectorizer = TfidfVectorizer()
    
#     # 4. Fit and Transform
#     # 'fit' learns the vocabulary from the training data.
#     # 'transform' converts the text into numbers based on that vocabulary.
#     print("\nConverting text skills into mathematical numbers (TF-IDF)...")
#     X_train_numeric = vectorizer.fit_transform(X_train)
    
#     # We ONLY transform the test data (we don't fit). It must use the training vocabulary.
#     X_test_numeric = vectorizer.transform(X_test)
    
#     print(f"Transformation complete! Vocabulary contains {len(vectorizer.get_feature_names_out())} unique skills/words.")
    
#     # 5. Save all our processed data and our vectorizer for the next steps
#     # We save the vectorizer because we need the EXACT same translator for user input later.
#     joblib.dump(X_train_numeric, 'X_train_numeric.pkl')
#     joblib.dump(X_test_numeric, 'X_test_numeric.pkl')
#     joblib.dump(y_train, 'y_train.pkl')
#     joblib.dump(y_test, 'y_test.pkl')
#     joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
#     print("Success: Processed data and TF-IDF translator saved to files.")

# if __name__ == "__main__":
#     preprocess_data()