# STEP-1 

import pandas as pd # Pandas is the standard library for data manipulation
import os

def load_and_inspect_data():
    print("-" * 50)
    print("STEP 1: LOADING CLUSTERED DATA")
    print("-" * 50)
    
    # Using your new clustered dataset file
    file_name = r'C:\Users\Praveen\Desktop\Internship\Carrer_path_Suggestion\Data\Career_Dataset_Clustered.xlsx'
    
    # Check if the file exists before trying to open it
    if not os.path.exists(file_name):
        print(f"Error: Could not find '{file_name}'. Please ensure it is in the same folder.")
        return

    # Read the file into a Pandas DataFrame (a 2D table of data)
    df = pd.read_excel(file_name)
    
    # 1. Show the shape of the data (Rows, Columns)
    print(f"\nData Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Show the column names (Should be Cluster, Career, Skill)
    print(f"Column Names: {list(df.columns)}")
    
    # 3. Show the first 3 rows as an example
    print("\nSample Data (First 3 rows):")
    print(df.head(3).to_string())
    
    # 4. Check for missing values (empty cells)
    missing_data = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_data.to_string())
    
    # We drop any rows that have missing data so it doesn't break our model later
    df_clean = df.dropna()
    
    # Save this clean data to a CSV file to pass it to Step 2
    df_clean = df.fillna("No Data     ")
    df_clean.to_csv('cleaned_careers_clustered.csv', index=False)
    print("\nSuccess: Cleaned data saved...")

if __name__ == "__main__":
    load_and_inspect_data()



# import pandas as pd 
# import os

# def load_and_inspect_data():
#     print("STEP 1: LOADING DATA")
#     print("-" * 50)

#     # Loading Dataset file...
#     file_name = r'C:\Users\Praveen\Desktop\Internship\Carrer_path_Suggestion\Data\career_dataset.xlsx'
#     df = pd.read_excel(file_name)

#     if not os.path.exists(file_name):
#         print(f"Error: Could not find '{file_name}'. Please ensure it is in the same folder.")
#         return
#     # Finding Dataset Rows and Columns inluding column names
#     print(f"\nData Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
#     print(f"Column Names: {list(df.columns)}")
#     print("\nSample Data (First 3 rows):")
#     print(df.head(3).to_string())

#     #Finding Missing data
#     missing_data = df.isnull().sum()
#     print("\nMissing values in each column:")
#     print(missing_data.to_string())

#     # Cleaning and Converting .xlsx to .csv after dataset cleaning...
#     df_clean = df.dropna()
#     df_clean.to_csv('cleaned_careers.csv', index=False)
#     print("\nSuccess: Cleaned data saved...")
    
# if __name__ == "__main__":
#     load_and_inspect_data()


