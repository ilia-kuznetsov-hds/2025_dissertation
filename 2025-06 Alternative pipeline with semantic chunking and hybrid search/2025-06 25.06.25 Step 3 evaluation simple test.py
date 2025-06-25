
import pandas as pd
import os
from llama_index.llms.google_genai import GoogleGenAI
import time


google_api_key = os.getenv("GOOGLE_API_KEY")



FILE_LLAMA_8B = r"C:\\Users\\kuzne\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\data\\2025-06 hybrid search\\psychiatry_test_dataset_together_meta-llama_Llama-3.2-3B-Instruct-Turbo_answered.csv"

def calculate_answer_correctness_(file_path, 
                                         model_name="gemini-2.0-flash", 
                                         max_rows=20, 
                                         batch_size=1, 
                                         timeout_seconds=10):
    
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".csv", "simply_evaluated.csv")

    # Check if the output file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        print(f"Starting new evaluation on: {FILE_PATH}")
        df = pd.read_csv(FILE_PATH)
        df['GPT-based scoring for vanilla'] = None
        df['GPT-based scoring for RAG'] = None
        df['Evaluation Model Answer Correctness for vanilla'] = model_name


    # Get rows that need evaluation
    evaluation_rows = df[(df['Generated Vanilla Answer'].notna()) & 
                    (df['Generated RAG Answer'].notna()) ]
    
    # Limit to max_rows
    rows_to_process = evaluation_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} vanilla answers to evaluate.")

    llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )

     # Track statistics
    skipped_rows = 0
    evaluated_rows = 0
    error_rows = 0

    for i, idx in enumerate(rows_to_process.index):
        try: 
            question = df.loc[idx, 'Modified Questions']
            answer_vanilla = df.loc[idx, 'Generated Vanilla Answer']
            answer_rag = df.loc[idx, 'Generated RAG Answer']
            ground_truth = df.loc[idx, 'Reasonings']

            prompt = f'''

            You are an expert in medical question answering evaluation. Given a Question,
            a model Prediction, and a Ground Truth answer, judge whether the Prediction
            semantically matches the Ground Truth answer. Follow the instructions:
            1. If the prediction semantically matches the ground truth completely, score 1.
            2. If the prediction semantically matches some part of the ground truth and is
            relevant to the question, score 0.5.
            3. If the prediction is completely wrong or irrelevant to the question, score 0.

            Question: {question}
            Prediction: {answer_vanilla}
            Ground Truth: {ground_truth}

            Output only the score as a number (0, 0.5, or 1) without any additional text.

            '''

            response = llm.complete(prompt)
            score_vanilla = float(response.text.strip())
            df.loc[idx, 'GPT-based scoring for vanilla'] = score_vanilla

            prompt = f'''

            You are an expert in medical question answering evaluation. Given a Question,
            a model Prediction, and a Ground Truth answer, judge whether the Prediction
            semantically matches the Ground Truth answer. Follow the instructions:
            1. If the prediction semantically matches the ground truth completely, score 1.
            2. If the prediction semantically matches some part of the ground truth and is
            relevant to the question, score 0.5.
            3. If the prediction is completely wrong or irrelevant to the question, score 0.

            Question: {question}
            Prediction: {answer_rag}
            Ground Truth: {ground_truth}

            Output only the score as a number (0, 0.5, or 1) without any additional text.

            '''
            response = llm.complete(prompt)
            score_rag = float(response.text.strip())
            df.loc[idx, 'GPT-based scoring for RAG'] = score_rag

            # Add timeout after specified interval
            
            print(f"Taking a {timeout_seconds} second break to avoid rate limits...")
            time.sleep(timeout_seconds)
            
            evaluated_rows += 1
            if i % BATCH_SIZE == 0:
                df.to_csv(OUTPUT_PATH, index=False)
                print(f"Processed {i + 1}/{total_rows} rows. Saved progress to {OUTPUT_PATH}")          


        except Exception as e:
            print(f"Error processing row {i + 1}/{total_rows}: {e}")
            error_rows += 1
            skipped_rows += 1
            df.loc[idx, 'GPT-based scoring for vanilla'] = "Error"
            df.loc[idx, 'GPT-based scoring for RAG'] = "Error"

    # Save final results
    df.to_csv(OUTPUT_PATH, index=False)


    print(f"Evaluation complete. Total rows evaluated: {evaluated_rows}, skipped: {skipped_rows}, errors: {error_rows}")
    print(f"Results saved to {OUTPUT_PATH}")   


calculate_answer_correctness_(FILE_LLAMA_8B, 
                                         model_name="gemini-2.0-flash",
                                            max_rows=450,
                                            batch_size=20,
                                            timeout_seconds=0)









