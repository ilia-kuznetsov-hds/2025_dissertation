import os
import pandas as pd
from llama_index.llms.google_genai import GoogleGenAI
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas import evaluate
from datasets import Dataset
import time
from ragas.metrics import answer_relevancy, faithfulness, context_precision
import json
import ast


google_api_key = os.getenv("GOOGLE_API_KEY")



# Constants for context window sizes
MODEL_CONTEXT_LIMITS = {
    "gemini-2.0-flash": 1048576,  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash
    "gemini-2.0-flash-lite": 1048576,
    "gemini-2.5-flash":1048576  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash-lite
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
        api_key=google_api_key,
        temperature=0.0, # Set temperature to 0 for deterministic outputs
        max_output_tokens=8192  # Maximum output tokens for Gemini models
          
    )

    gemini_embeddings = GoogleGenAIEmbedding(
        model_name="models/text-embedding-004",  # Latest Gemini embedding model
        api_key=google_api_key
    )
    # Wrap both with RAGAS adapters
    llm_wrapper = LlamaIndexLLMWrapper(gemini_llm)
    embeddings_wrapper = LlamaIndexEmbeddingsWrapper(gemini_embeddings)
    # Wrap with RAGAS LlamaIndexLLM adapter
    return llm_wrapper, embeddings_wrapper


def calculate_rag_metric(file_path, model_name="gemini-2.0-flash", metric:str= "answer_relevancy", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    Calculate RAGAS metrics for a given dataset file.
    The answer relevancy could be used for vanilla answer evaluation, 
    while faithfulness and context precision are used for RAG dataset evaluation exclusively.
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        model_name (str): Name of the model to use for evaluation.
        metric (str): The metric to evaluate. Options: "answer_relevancy", "faithfulness".
        max_rows (int): Maximum number of rows to process from the dataset.
        batch_size (int): Number of rows to process before saving progress.
        timeout_seconds (int): Timeout between evaluations in seconds.

    Returns:
        None: Saves the evaluated results to a file.
    '''
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", f"_rag_{metric}_evaluated.json")

    # Map metric strings to actual RAGAS metric objects
    metric_mapping = {
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "context_precision": context_precision
    }
    
    if metric not in metric_mapping:
        raise ValueError(f"Unsupported metric: {metric}. Supported metrics: {list(metric_mapping.keys())}")
    
    selected_metric = metric_mapping[metric]


    # Load JSON data
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        print(f"Starting new evaluation on: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
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

    def parse_context_string(context_string):
        """
        Parse context string into a list of context paragraphs.
        Splits by the separator and removes empty strings and separators.
        
        Args:
            context_string (str): The raw context string with separators
        
        Returns:
            list: List of context paragraphs (strings)
        """
        if not context_string:
            return []
        
        # Split by the separator
        separator = "\n\n\n--- Context from different searches ---\n\n"
        contexts = context_string.split(separator)
        
        # Clean up each context: strip whitespace and filter out empty strings
        cleaned_contexts = []
        for context in contexts:
            cleaned = context.strip()
            if cleaned:  # Only add non-empty contexts
                cleaned_contexts.append(cleaned)
        
        return cleaned_contexts
    
    # Initialize evaluator once outside the loop
    llm_wrapper, embeddings_wrapper = setup_ragas_evaluator(model_name=model_name)

    for i, idx in enumerate(rows_to_process.index):
        try:
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, 'Generated RAG Answer']
            context_string = df.loc[idx, 'Retrieved Context']
            context = parse_context_string(context_string)
            ground_truth = df.loc[idx, 'Reasonings']
               
            # Count tokens for each component
            question_tokens = count_tokens(question)
            answer_tokens = count_tokens(answer)
            context_tokens = count_tokens(context_string)
            ground_truth_tokens = count_tokens(ground_truth)
            total_tokens = question_tokens + answer_tokens + ground_truth_tokens + context_tokens

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
                'question': [question],
                'answer': [answer],
                'retrieved_contexts': [context],
                'ground_truth': [ground_truth],
            }

            dataset = Dataset.from_dict(data_samples)
        
            # Run evaluation 3 times with dynamic metric
            scores = []
            for run in range(1, 4):

                score = evaluate(dataset,
                               metrics=[selected_metric], 
                               llm=llm_wrapper,
                               embeddings=embeddings_wrapper)
                
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
            data = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            error_rows = 0  # Reset error counter after saving

    # Final save
    data = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    error_rows = 0  # Reset error counter after saving

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


def calculate_vanilla_metric(file_path, model_name="gemini-2.0-flash", metric:str= "answer_relevancy", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    Calculate RAGAS answer relevancy metrics for a given dataset file.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        model_name (str): Name of the model to use for evaluation.
        metric (str): The metric to evaluate. Options: "answer_relevancy", "faithfulness".
        max_rows (int): Maximum number of rows to process from the dataset.
        batch_size (int): Number of rows to process before saving progress.
        timeout_seconds (int): Timeout between evaluations in seconds.

    Returns:
        None: Saves the evaluated results to a file.
    '''
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", f"_vanilla_{metric}_evaluated.json")

    # Map metric strings to actual RAGAS metric objects
    metric_mapping = {
        "answer_relevancy": answer_relevancy
    }
    
    if metric not in metric_mapping:
        raise ValueError(f"Unsupported metric: {metric}. Supported metrics: {list(metric_mapping.keys())}")
    
    selected_metric = metric_mapping[metric]


    # Load JSON data
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        print(f"Starting new evaluation on: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
         # Dynamic column creation based on metric name
        df[f'{metric} for Vanilla run 1'] = None
        df[f'{metric} for Vanilla run 2'] = None
        df[f'{metric} for Vanilla run 3'] = None
        df[f'Mean {metric} for Vanilla'] = None
        df[f'Evaluation Notes {metric} for Vanilla'] = None
        df[f'Evaluation Model {metric} for Vanilla'] = model_name

    # Get model context window limit
    if model_name not in MODEL_CONTEXT_LIMITS:
        print(f"Warning: Unknown model '{model_name}'. Using default limit of 128000 tokens.")
        model_context_limit = 128000
    else:
        model_context_limit = MODEL_CONTEXT_LIMITS[model_name]

    # Get rows that need evaluation
    rag_rows = df[(df['Generated Vanilla Answer'].notna()) & 
                    (df[f'{metric} for Vanilla run 1'].isna()) &
                    (df[f'Evaluation Notes {metric} for Vanilla'].isna())]
    
    # Limit to max_rows
    rows_to_process = rag_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} Vanilla answers to evaluate for {metric}.")

     # Track statistics
    skipped_rows = 0
    evaluated_rows = 0
    error_rows = 0


    for i, idx in enumerate(rows_to_process.index):
        try:
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, 'Generated Vanilla Answer']
            context_string = df.loc[idx, 'Retrieved Context']
            ground_truth = df.loc[idx, 'Reasonings']
               
            # Count tokens for each component
            question_tokens = count_tokens(question)
            answer_tokens = count_tokens(answer)
            ground_truth_tokens = count_tokens(ground_truth)
            total_tokens = question_tokens + answer_tokens + ground_truth_tokens

            # Check if we're within the context window limit
            # The way to handle error when the context limit is exceeded
            if total_tokens > model_context_limit:
                print(f"Row {idx} exceeds context limit. Skipping evaluation.")
                df.loc[idx, f'{metric} for Vanilla run 1'] = None
                df.loc[idx, f'Evaluation Notes {metric} for Vanilla'] = f"Exceeded context limit"
                skipped_rows += 1
                continue
            
            # Create dictionary for dataset
            # This is requeirement of RAGAS
            data_samples = {
                'question': [question],
                'answer': [answer],
                'ground_truth': [ground_truth],
            }

            dataset = Dataset.from_dict(data_samples)
        
            # Run evaluation 3 times with dynamic metric
            scores = []
            for run in range(1, 4):
                llm_wrapper, embeddings_wrapper = setup_ragas_evaluator(model_name=model_name)
                score = evaluate(dataset,
                               metrics=[selected_metric], 
                               llm=llm_wrapper,
                               embeddings=embeddings_wrapper)
                
                # Extract the score value - key name matches metric name
                metric_score = score[metric][0]  # Get first (only) element
                df.loc[idx, f'{metric} for Vanilla run {run}'] = metric_score
                scores.append(metric_score)
                print(f"Row {idx} - Run {run} Score: {metric_score}")
                time.sleep(timeout_seconds)
            
            # Calculate the mean of all three scores
            mean_score = sum(scores) / 3
            df.loc[idx, f'Mean {metric} for Vanilla'] = mean_score
            print(f"Row {idx} - Mean {metric} Score: {mean_score}")
            df.loc[idx, f'Evaluation Notes {metric} for Vanilla'] = f"Evaluated 3 runs"
            evaluated_rows += 1
        
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            df.loc[idx, f'Evaluation Notes {metric} for Vanilla'] = f"Error: {str(e)}"
            error_rows += 1

        # Save after each batch or when an error occurs
        if ((i + 1) % BATCH_SIZE == 0) or error_rows > 0:
            print(f"Saving progress...")
            data = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            error_rows = 0  # Reset error counter after saving

    # Final save
    data = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

    # Report completion status
    remaining = len(df[(df['Generated Vanilla Answer'].notna()) & 
                      (df[f'{metric} for Vanilla run 1'].isna()) &
                       (df[f'Evaluation Notes {metric} for Vanilla'].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.
              Skipped {skipped_rows} rows due to context limit.
              Evaluated {evaluated_rows} rows.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")







# RESULTS FILES
LLAMA_SCOUT = 'experiments/test_dataset_together_meta-llama_Llama-4-Scout-17B-16E-Instruct_top5_answered.json'

LLAMA_MAVERICK = 'experiments/test_dataset_together_meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8_top5_answered.json'

'''
calculate_vanilla_metric(LLAMA_MAVERICK,
                     model_name="gemini-2.0-flash", 
                    metric="answer_relevancy",
                    max_rows=2, batch_size=10, timeout_seconds=0)


calculate_rag_metric(LLAMA_MAVERICK,
                     model_name="gemini-2.0-flash", 
                    metric="answer_relevancy",
                    max_rows=2, batch_size=2, timeout_seconds=0)


calculate_rag_metric(LLAMA_MAVERICK,
                     model_name="gemini-2.0-flash", 
                    metric="context_precision",
                    max_rows=2, batch_size=2, timeout_seconds=0)
'''

file_path = 'experiments/qwen2.5_72b/test_dataset_together_Qwen_Qwen2.5-72B-Instruct-Turbo_top5_answered.json'

calculate_rag_metric(file_path,
                        model_name="gemini-2.0-flash", 
                        metric="context_precision",
                        max_rows=370, batch_size=2, timeout_seconds=0)

calculate_vanilla_metric(file_path,
                        model_name="gemini-2.0-flash",
                        metric="answer_relevancy",
                        max_rows=370, batch_size=2, timeout_seconds=0)