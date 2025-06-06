import os
import pandas as pd
from llama_index.llms.google_genai import GoogleGenAI
from ragas.llms import LlamaIndexLLMWrapper
from ragas import evaluate
from datasets import Dataset
import time
from ragas.metrics import answer_relevancy, faithfulness
import ast


google_api_key = os.getenv("GOOGLE_API_KEY")

QUESTIONS_FILE = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-06 02.06.2025 dataset for evaluation\\psychiatry_train_dataset_groq_gemma2-9b-it_answered.csv"

# Constants for context window sizes
MODEL_CONTEXT_LIMITS = {
    "gemini-2.0-flash": 1048576,  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash
    "gemini-2.0-flash-lite": 1048576,  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash-lite
}


def count_tokens(text):
    """
    Count the approximate number of tokens in a text string.
    Uses a simple heuristic: ~4 characters per token for common English text.
    
    Args:
        text (str): The text to count tokens for
    
    Returns:
        int: Approximate token count
    """
    if not text or not isinstance(text, str):
        return 0
    # Count characters and divide by 4 (common approximation for English text)
    # Add 1 to round up for partial tokens
    return len(text) // 4 + (1 if len(text) % 4 else 0)


def setup_ragas_evaluator(model_name="gemini-2.0-flash"):
    """
    Initialize the RAGAS evaluator with Google Gemini using LlamaIndex.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    model_name = model_name
    # Create LlamaIndex GoogleGenAI instance
    gemini_llm = GoogleGenAI(
        model=model_name,
        api_key=google_api_key
    )
    # Wrap with RAGAS LlamaIndexLLM adapter
    return LlamaIndexLLMWrapper(gemini_llm)


def calculate_rag_metric(file_path, model_name="gemini-2.0-flash", metric:str= "answer_relevancy", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    Calculate RAGAS metrics for a given dataset file.
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        model_name (str): Name of the model to use for evaluation.
        metric (str): The metric to evaluate. Options: "answer_relevancy", "faithfulness".
        max_rows (int): Maximum number of rows to process from the dataset.
        batch_size (int): Number of rows to process before saving progress.
        timeout_seconds (int): Timeout between evaluations in seconds.

    Returns:
        None: Saves the evaluated results to a new CSV file.
    '''
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".csv", f"_rag_{metric}_evaluated.csv")

    # Map metric strings to actual RAGAS metric objects
    metric_mapping = {
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness
    }
    
    if metric not in metric_mapping:
        raise ValueError(f"Unsupported metric: {metric}. Supported metrics: {list(metric_mapping.keys())}")
    
    selected_metric = metric_mapping[metric]

    # Check if the output file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        print(f"Starting new evaluation on: {FILE_PATH}")
        df = pd.read_csv(FILE_PATH)
         # Dynamic column creation based on metric name
        df[f'{metric} for RAG run 1'] = None
        df[f'{metric} for RAG run 2'] = None
        df[f'{metric} for RAG run 3'] = None
        df[f'Mean {metric} for RAG'] = None
        df[f'Evaluation Notes {metric} for RAG'] = None
        df[f'Evaluation Model {metric} for RAG'] = model_name

    # Get model context window limit
    if model_name not in MODEL_CONTEXT_LIMITS:
        print(f"Warning: Unknown model '{model_name}'. Using default limit of 128000 tokens.")
        model_context_limit = 128000
    else:
        model_context_limit = MODEL_CONTEXT_LIMITS[model_name]

    # Get rows that need evaluation
    rag_rows = df[(df['Generated RAG Answer'].notna()) & 
                    (df[f'{metric} for RAG run 1'].isna()) &
                    (df[f'Evaluation Notes {metric} for RAG'].isna())]
    
    # Limit to max_rows
    rows_to_process = rag_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} RAG answers to evaluate for {metric}.")

     # Track statistics
    skipped_rows = 0
    evaluated_rows = 0
    error_rows = 0

    for i, idx in enumerate(rows_to_process.index):
        try:
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, 'Generated RAG Answer']
            context_string = df.loc[idx, 'Retrieved Context']
            try:
                # Convert string representation of list to actual list
                if isinstance(context_string, str) and context_string.strip().startswith('['):
                    context = ast.literal_eval(context_string)
                else:
                    context = [context_string]  # Wrap single string in list
            except (ValueError, SyntaxError):
                # Fallback: treat as single string wrapped in list
                context = [context_string]
            
                
            # Count tokens for each component
            question_tokens = count_tokens(question)
            answer_tokens = count_tokens(answer)
            ground_truth_tokens = count_tokens(context)
            total_tokens = question_tokens + answer_tokens + ground_truth_tokens

            # Check if we're within the context window limit
            # The way to handle error when the context limit is exceeded
            if total_tokens > model_context_limit:
                print(f"Row {idx} exceeds context limit. Skipping evaluation.")
                df.loc[idx, f'{metric} for RAG run 1'] = None
                df.loc[idx, f'Evaluation Notes {metric} for RAG'] = f"Exceeded context limit"
                skipped_rows += 1
                continue
            
            # Create dictionary for dataset
            # This is requeirement of RAGAS
            data_samples = {
                'user_input': [],
                'response': [],
                'retrieved_contexts': []
            }

            # Add data to dictionary
            data_samples['user_input'].append(df.loc[idx, 'Modified Questions'])
            data_samples['response'].append(df.loc[idx, 'Generated RAG Answer'])
            data_samples['retrieved_contexts'].append(context)
            dataset = Dataset.from_dict(data_samples)
        
            # Run evaluation 3 times with dynamic metric
            scores = []
            for run in range(1, 4):
                score = evaluate(dataset,
                               metrics=[selected_metric], 
                               llm=setup_ragas_evaluator(model_name=model_name))
                
                # Extract the score value - key name matches metric name
                metric_score = score[metric][0]  # Get first (only) element
                df.loc[idx, f'{metric} for RAG run {run}'] = metric_score
                scores.append(metric_score)
                print(f"Row {idx} - Run {run} Score: {metric_score}")
                time.sleep(timeout_seconds)
            
            # Calculate the mean of all three scores
            mean_score = sum(scores) / 3
            df.loc[idx, f'Mean {metric} for RAG'] = mean_score
            print(f"Row {idx} - Mean {metric} Score: {mean_score}")
            df.loc[idx, f'Evaluation Notes {metric} for RAG'] = f"Evaluated 3 runs"
            evaluated_rows += 1
        
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            df.loc[idx, f'Evaluation Notes {metric} for RAG'] = f"Error: {str(e)}"
            error_rows += 1

        # Save after each batch or when an error occurs
        if ((i + 1) % BATCH_SIZE == 0) or error_rows > 0:
            print(f"Saving progress...")
            df.to_csv(OUTPUT_PATH, index=False)
            error_rows = 0  # Reset error counter after saving

    # Final save
    df.to_csv(OUTPUT_PATH, index=False)


    # Report completion status
    remaining = len(df[(df['Generated RAG Answer'].notna()) & 
                      (df[f'{metric} for RAG run 1'].isna()) &
                       (df[f'Evaluation Notes {metric} for RAG'].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.
              Skipped {skipped_rows} rows due to context limit.
              Evaluated {evaluated_rows} rows.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")


calculate_rag_metric(QUESTIONS_FILE, model_name="gemini-2.0-flash", metric="answer_relevancy", max_rows=8, batch_size=1, timeout_seconds=10)



























