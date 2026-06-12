import ast
import os
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from llama_index.llms.google_genai import GoogleGenAI
from ragas import evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness


google_api_key = os.getenv("GOOGLE_API_KEY")

# Repository root, used to build paths that work on different machines.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Folder that contains the evaluation datasets and generated answer files.
DATA_DIR = REPO_ROOT / "data"

# CSV produced by the answer-generation step and used as input for RAGAS.
QUESTIONS_FILE = str(
    DATA_DIR / "test_dataset_gemini_gemini-3-flash-preview_answered.csv"
)

# Model context-window limits used to skip rows that are too large to evaluate.
MODEL_CONTEXT_LIMITS = {
    "gemini-3-flash-preview": 1048576,
}


def count_tokens(text):
    """
    Estimate how many tokens a text will use before sending it to the evaluator.

    This is a lightweight approximation, not an exact tokenizer. Many English
    texts average around 4 characters per token, so dividing character length by
    4 gives a quick estimate that is good enough for deciding whether an input
    may exceed the model context window.
    
    Args:
        text (str): The text to count tokens for
    
    Returns:
        int: Approximate token count
    """
    if not text or not isinstance(text, str):
        return 0

    return len(text) // 4 + (1 if len(text) % 4 else 0)


def setup_ragas_evaluator(model_name="gemini-3-flash-preview"):
    """
    Create the LLM object that RAGAS will use to score answers.

    Workflow:
    1. Check that GOOGLE_API_KEY is available in the environment.
    2. Create a LlamaIndex GoogleGenAI client for the selected Gemini model.
    3. Wrap that client with RAGAS' LlamaIndex adapter, because RAGAS expects
       its evaluator LLMs to follow the RAGAS wrapper interface.
    4. Return the wrapped evaluator so it can be passed into evaluate(...).
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    model_name = model_name
    gemini_llm = GoogleGenAI(
        model=model_name,
        api_key=google_api_key
    )
    
    return LlamaIndexLLMWrapper(gemini_llm)


def calculate_rag_metric(file_path, model_name="gemini-3-flash-preview", metric:str= "answer_relevancy", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    Calculate RAGAS metrics for a given dataset file.

    This script evaluates the quality of generated RAG answers using RAGAS metrics and a Gemini evaluator model.

    At a high level, the script:

    1. Loads the answered test dataset produced by the previous RAG generation step.
    2. Checks each row that contains a generated RAG answer and has not already been evaluated.
    3. Uses the question, generated answer, and retrieved context as inputs for RAGAS evaluation.
    4. Evaluates each answer multiple times for each selected metric.
    5. Saves metric scores, mean scores, evaluation notes, and evaluator model information back to CSV files.
    6. Supports resumable evaluation by continuing from existing output files instead of starting over.

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
    evaluator_llm = setup_ragas_evaluator(model_name=model_name)

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
                               llm=evaluator_llm)
                
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


def calculate_rag_metrics(file_path, metrics, model_name="gemini-3-flash-preview", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    Calculate multiple RAGAS metrics while preserving the existing per-metric
    output files and resume behavior.
    '''
    for metric in metrics:
        calculate_rag_metric(
            file_path,
            model_name=model_name,
            metric=metric,
            max_rows=max_rows,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds
        )



EVALUATION_FILE = QUESTIONS_FILE

calculate_rag_metrics(
    EVALUATION_FILE,
    metrics=["answer_relevancy", "faithfulness"],
    model_name="gemini-3-flash-preview",
    max_rows=450,
    batch_size=10,
    timeout_seconds=0
)
