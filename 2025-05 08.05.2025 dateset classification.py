from google import genai
import os
import pandas as pd
import time 

google_api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=google_api_key)

dataset_path = "C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\data\\2025-05 08.05.25 dataset for classification\\MedQA_open_dataset.xlsx"

'''
We use MedQA-Open dataset from the paper "Few shot chain-of-thought driven reasoning to prompt LLMs for open ended medical question answering"
The dataset available at (ancillary files setion):
https://arxiv.org/abs/2403.04890

The author of the paper used the USMLE-MedQA dataset (Jin et al., 2021), a medical exam dataset that consists of questions 
sourced from professional medical board exams in the USA.


The authors used the MedQA dataset (Zhang et al., 2018) is a publicly available collection of complex medical questions
with multiple choices based on the United States medical license exams. To emulate real-world medical scenarios, 
they convert these multiple-choice questions into open-ended questions by (1) removing the multiple-choice options and 
(2) re-pharsing the question to be open-ended using LLM, creating MedQA-Open. 

The dataset contains around 9000 questions related to different medical fields, including psychiatry. 
The following script uses LLM to classify questions as psychiatry-related or not psychiatry-related.
It is a screening step that uses fast cost-efficient LLM to filter out non-psychiatry questions 
before using more expensive LLMs for detailed analysis.
'''


def classify_question(question, answer):
    """
    Classify a question as psychiatry-related or not psychiatry related using LLM.
    """
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Is this question related to psychiatry? Respond with only 'psychiatry' or 'non-psychiatry'.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        result = response.text.strip().lower()
        
        # Ensure the response is either 'psychiatry' or 'non-psychiatry'
        if "psychiatry" in result and not ("non-psychiatry" in result):
            return "psychiatry"
        return "non-psychiatry"
    except Exception as e:
        print(f"Error classifying question: {e}")
        return None
    

def process_file(file_path, batch_size=10, max_rows=150, timeout_interval=25, timeout_seconds=30):
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_classified.xlsx")

    # Check if the output file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        df = pd.read_excel(OUTPUT_PATH)
    else:
        print(f"Starting new classification on: {FILE_PATH}")
        df = pd.read_excel(FILE_PATH)
        df['classification'] = None

    # Get rows that don't have a classification yet
    rows_to_process = df[df['classification'].isna()].index.tolist()

    # Limit to max_rows
    rows_to_process = rows_to_process[:max_rows]
    total_rows = len(rows_to_process)
        
    print(f"Found {len(df[df['classification'].isna()])} unclassified questions total")
    print(f"Will process {total_rows} questions in this run (max_rows={max_rows})")
    
    for i, idx in enumerate(rows_to_process):
        # Get question and answer
        question = df.loc[idx, 'Modified Questions']
        answer = df.loc[idx, 'Reasonings']
            
        # Skip if empty
        if pd.isna(question) or str(question).strip() == '':
            df.loc[idx, 'classification'] = "invalid"
            continue
        # Classify the question
        classification = classify_question(question, answer)
        
        # Handle API errors
        if classification is None:
            print(f"Error encountered. Saving progress and exiting.")
            df.to_excel(OUTPUT_PATH, index=False)
            return

        # Update the dataframe
        df.loc[idx, 'classification'] = classification
        print(f"Processed {i+1}/{total_rows}: Row {idx} - {classification}")

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
    remaining = len(df[df['classification'].isna()])
    if remaining > 0:
        print(f"Run complete! {total_rows} questions processed. {remaining} questions remain unclassified.")
    else:
        print(f"All questions have been classified! Total: {len(df)} questions.")
                
      
'''
Iteratively run the function until all questions are classified.
Daily limit is 1500 API calls.

'''
process_file(
    dataset_path, 
    batch_size=10,      # Save every 30 questions
    max_rows=1200,       # Process rows per run
    timeout_interval=1, # Take a break after every 1 question (so we don't ecxeed RPM limit)
    timeout_seconds=8   # Break for 5 seconds is enough to avoid rate limits
)
