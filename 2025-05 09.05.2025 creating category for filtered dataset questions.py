from google import genai
import os
import pandas as pd
import time 

google_api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=google_api_key)
dataset_path = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-05 08.05.25 dataset for classification\\MedQA_open_dataset_classified.xlsx"


def classify_question_category(question):
    """
    Script for identifying the topic of psychiatric question

    """
    prompt = f"""

    Act as experienced mental health specialist. Your task is to classify a provided psychiatric question.
    Question: {question}
    Identify disease discussed in the provided question and put it into one of the following categories:
    1.	Anxiety Disorders
    2.	Bipolar Disorders
    3.	Depressive Disorders
    4.	Dissociative Disorders
    5.	Eating Disorders
    6.	Obsessive-Compulsive Disorders
    7.	Personality Disorders
    8.	Schizophrenia Spectrum and Other Psychotic Disorders
    9.	Somatic Disorders
    10.	Trauma and Stressor Related Disorders
    11. Other Mental Disorders

    Some questions won't have a clear diagnosis. You should identify the diagnosis and then categories it into one of the above categories.
    Respond with only one category for each question. Your answer should consist only of provided categories. 
    If you are not sure about the answer, please choose the "Other Mental Disorders" category.
    """

    psychiatric_categories_keywords = [
    "anxiety disorders",
    "bipolar disorders",
    "depressive disorders",
    "dissociative disorders",
    "eating disorders",
    "obsessive-compulsive disorders",
    "personality disorders",
    "schizophrenia spectrum and other psychotic disorders",
    "somatic disorders",
    "trauma and stressor related disorders",
    "other mental disorders"]
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        result = response.text.strip().lower()
        
        # Ensure the response is either 'psychiatry' or 'non-psychiatry'
        if result in psychiatric_categories_keywords:
            return result
        
        else:
            print(f"Got unexpected category: '{result}")
            return "other mental disorders"
        
    except Exception as e:
        print(f"Error classifying question: {e}")
        return "invalid"
    

def process_file(file_path, batch_size=10, max_rows=150, timeout_interval=25, timeout_seconds=30):
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_categories.xlsx")

    # Check if the output file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        df = pd.read_excel(OUTPUT_PATH)
    else:
        print(f"Starting new classification on: {FILE_PATH}")
        df = pd.read_excel(FILE_PATH)
        df['category'] = None

    df = df[df['classification'] == 'psychiatry']
    df = df.reset_index(drop=True)

    # Get rows that don't have a category yet
    rows_to_process = df[df['category'].isna()].index.tolist()

    # Limit to max_rows
    rows_to_process = rows_to_process[:max_rows]
    total_rows = len(rows_to_process)
        
    print(f"Found {len(df[df['category'].isna()])} unclassified questions total")
    print(f"Will process {total_rows} questions in this run (max_rows={max_rows})")
    
    for i, idx in enumerate(rows_to_process):
        # Get question and answer
        question = df.loc[idx, 'Modified Questions']
            
        # Skip if empty
        if pd.isna(question):
            df.loc[idx, 'category'] = "invalid"
            continue
        # Classify the question
        classification_category = classify_question_category(question)
        df.loc[idx, 'category'] = classification_category

         # Handle API errors
        if df.loc[idx, 'category'].isna():
            print(f"Error encountered. Saving progress and exiting.")
            df.to_excel(OUTPUT_PATH, index=False)
            return
        
        # Update the dataframe
        
        print(f"Processed {i+1}/{total_rows}: Row {idx} - {classification_category}")

       

        # Save after each batch
        if (i + 1) % BATCH_SIZE == 0:
            print(f"Saving progress after batch...")
            df.to_excel(OUTPUT_PATH, index=False)
        
        # Add timeout after specified interval
        if (i + 1) % timeout_interval == 0 and i+1 < total_rows:
            print(f"Taking a {timeout_seconds} second break to avoid rate limits...")
            time.sleep(timeout_seconds)

    # Final save
    print("Saving final results...")
    df.to_excel(OUTPUT_PATH, index=False)
    
    # Report completion status
    remaining = len(df[df['category'].isna()])
    if remaining > 0:
        print(f"Run complete! {total_rows} questions processed. {remaining} questions remain unclassified.")
    else:
        print(f"All questions have been classified! Total: {len(df)} questions.")
                
      


process_file(dataset_path, 
             batch_size=1, 
             max_rows=500, 
             timeout_interval=1, 
             timeout_seconds=5)