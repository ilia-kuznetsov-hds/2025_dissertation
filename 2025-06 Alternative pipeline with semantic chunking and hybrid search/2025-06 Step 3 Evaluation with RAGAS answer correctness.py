import os
import pandas as pd
from llama_index.llms.google_genai import GoogleGenAI
from ragas.llms import LlamaIndexLLMWrapper
from ragas import evaluate
from ragas.metrics import answer_correctness
from datasets import Dataset
import time
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
import json


google_api_key = os.getenv("GOOGLE_API_KEY")

QUESTIONS_FILE = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-06 02.06.2025 dataset for evaluation\\psychiatry_train_dataset_groq_llama3-8b-8192_answered.csv"

# Constants for context window sizes
MODEL_CONTEXT_LIMITS = {
    "gemini-2.0-flash": 1048576,  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash
    "gemini-2.0-flash-lite": 1048576,
    "gemini-2.5-flash":1048576  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash-lite
}



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

    gemini_embeddings = GoogleGenAIEmbedding(
        model_name="models/text-embedding-004",  # Latest Gemini embedding model
        api_key=google_api_key
    )
    # Wrap both with RAGAS adapters
    llm_wrapper = LlamaIndexLLMWrapper(gemini_llm)
    embeddings_wrapper = LlamaIndexEmbeddingsWrapper(gemini_embeddings)
    # Wrap with RAGAS LlamaIndexLLM adapter
    return llm_wrapper, embeddings_wrapper


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


def calculate_answer_correctness_vanilla(file_path, model_name="gemini-2.0-flash", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    Parameters:
    - file_path: str
        Path to the CSV file containing the dataset with already answered questions 
        (produced by script for step 3 answering medical question using LLM).
    - model_name: str
        Name of the model to use for evaluation. Default is "gemini-2.0-flash".
    - max_rows: int
        Maximum number of rows to process from the dataset. Default is 20.
    - batch_size: int
        Number of rows to process in each batch before saving the updated CSV file.
    - timeout_seconds: int
        Number of seconds to wait between evaluations to avoid rate limiting. Default is 10 seconds.
    Output: 
        DataFrame
            The original DataFrame with an 3 Answer Correctness score columns and calculated mean score column saved to the CSV file.
            The output file will be saved with "_vanilla_answer_correctness_evaluated" suffix.
    
    Functionality:
        https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_correctness.html
        https://github.com/dkhundley/llm-rag-guide/blob/main/notebooks/ragas.ipynb

        The assessment of Answer Correctness involves measuring the accuracy of the generated answer 
        when compared to the ground truth. This evaluation relies on the ground truth and the answer, 
        with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated 
        answer and the ground truth, signifying better correctness.
        Answer correctness  is computed as the sum of factual correctness and the semantic similarity 
        between the given answer and the ground truth.

        https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/#factual-correctness
        Factual correctness is a metric that compares and evaluates the factual accuracy of the generated 
        response with the reference. This metric is used to determine the extent to which the generated 
        response aligns with the reference. The factual correctness score ranges from 0 to 1, 
        with higher values indicating better performance. To measure the alignment between the 
        response and the reference, the metric uses the LLM to first break down the response and 
        reference into claims and then uses natural language inference to determine the factual 
        overlap between the response and the reference. Factual overlap is quantified using precision, 
        recall, and F1 score, which can be controlled using the mode parameter. By default, the mode is set to F1, you can change 
        the mode to precision or recall by setting the mode parameter.

        Answer similarity is calculated by following steps:
        https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/semantic_similarity/
        Step 1: Vectorize the ground truth answer using the embedding model.
        Step 2: Vectorize the generated answer using the same embedding model.
        Step 3: Compute the cosine similarity between the two vectors.
        
        https://docs.ragas.io/en/stable/references/embeddings/#ragas.embeddings.embedding_factory
        By default "text-embedding-ada-002" model is used.

        Final score is created by taking a weighted average of the factual correctness (F1 score) and the semantic similarity. 
        (By default, there is a 0.75 : 0.25 weighting.)    
    '''
    
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", "_vanilla_answer_correctness_evaluated.json")

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
        df['Answer Correctness for vanilla run 1'] = None
        df['Answer Correctness for vanilla run 2'] = None
        df['Answer Correctness for vanilla run 3'] = None
        df['Mean Answer Correctness for vanilla'] = None
        df['Evaluation Notes Answer Correctness for vanilla'] = None
        df['Evaluation Model Answer Correctness for vanilla'] = model_name

    # Get model context window limit
    if model_name not in MODEL_CONTEXT_LIMITS:
        print(f"Warning: Unknown model '{model_name}'. Using default limit of 128000 tokens.")
        model_context_limit = 128000
    else:
        model_context_limit = MODEL_CONTEXT_LIMITS[model_name]

    # Get rows that need evaluation
    vanilla_rows = df[(df['Generated Vanilla Answer'].notna()) & 
                    (df['Answer Correctness for vanilla run 1'].isna()) &
                    (df['Evaluation Notes Answer Correctness for vanilla'].isna())]
    
    # Limit to max_rows
    rows_to_process = vanilla_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} vanilla answers to evaluate.")

     # Track statistics
    skipped_rows = 0
    evaluated_rows = 0
    error_rows = 0

    for i, idx in enumerate(rows_to_process.index):
        try: 
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, 'Generated Vanilla Answer']
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
                df.loc[idx, 'Answer Correctness for vanilla'] = None
                df.loc[idx, 'Evaluation Notes Answer Correctness for vanilla'] = f"Exceeded context limit"
                skipped_rows += 1
                continue
            
            # Create dictionary for dataset
            # This is requeirement of RAGAS
            # Create dictionary for dataset
            data_samples = {
                'question': [question],
                'answer': [answer],
                'ground_truth': [ground_truth]
            }

            dataset = Dataset.from_dict(data_samples)

            # Run 3 evaluations and collect scores
            scores = []
            for run_num in range(1, 4):
                llm_wrapper, embeddings_wrapper = setup_ragas_evaluator(model_name=model_name)
                score = evaluate(dataset,
                               metrics=[answer_correctness], 
                               llm=llm_wrapper,
                               embeddings=embeddings_wrapper)
                
                answer_correctness_score = score['answer_correctness'][0]
                scores.append(answer_correctness_score)
                
                # Update dataframe with individual run score
                df.loc[idx, f'Answer Correctness for vanilla run {run_num}'] = answer_correctness_score
                print(f"Row {idx} - Run {run_num} Score: {answer_correctness_score}")
                time.sleep(timeout_seconds)

            # Calculate and store mean score
            mean_score = sum(scores) / len(scores)
            df.loc[idx, 'Mean Answer Correctness for vanilla'] = mean_score
            print(f"Row {idx} - Mean Score: {mean_score}")
            df.loc[idx, 'Evaluation Notes Answer Correctness for vanilla'] = f"Evaluated 3 runs"
            evaluated_rows += 1

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            df.loc[idx, 'Evaluation Notes Answer Correctness for vanilla'] = f"Error"
            error_rows += 1

        # Save after each batch or when an error occurs
        if ((i + 1) % BATCH_SIZE == 0) or error_rows > 0:
            print(f"Saving progress...")
            data = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            error_rows = 0  # Reset error counter after saving


    # Report completion status
    remaining = len(df[(df['Generated Vanilla Answer'].notna()) & 
                      (df['Answer Correctness for vanilla run 1'].isna()) &
                       (df['Evaluation Notes Answer Correctness for vanilla'].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.
              Skipped {skipped_rows} rows due to context limit.
              Evaluated {evaluated_rows} rows.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")

    

def calculate_answer_correctness_rag(file_path, model_name="gemini-2.0-flash", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    The same functionality as calculate_answer_correctness_vanilla, but for RAG answers.
    '''
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", "_rag_answer_correctness_evaluated.json")

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
        df = pd.DataFrame(data)  # Create DataFrame for consistent processing
        df['Answer Correctness for RAG run 1'] = None
        df['Answer Correctness for RAG run 2'] = None
        df['Answer Correctness for RAG run 3'] = None
        df['Mean Answer Correctness for RAG'] = None
        df['Evaluation Notes Answer Correctness for RAG'] = None
        df['Evaluation Model AAnswer Correctness for RAG'] = model_name

    # Get model context window limit
    if model_name not in MODEL_CONTEXT_LIMITS:
        print(f"Warning: Unknown model '{model_name}'. Using default limit of 128000 tokens.")
        model_context_limit = 128000
    else:
        model_context_limit = MODEL_CONTEXT_LIMITS[model_name]

    # Get rows that need evaluation
    rag_rows = df[(df['Generated RAG Answer'].notna()) & 
                    (df['Answer Correctness for RAG run 1'].isna()) &
                    (df['Evaluation Notes Answer Correctness for RAG'].isna())]
    
    # Limit to max_rows
    rows_to_process = rag_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} RAG answers to evaluate.")

     # Track statistics
    skipped_rows = 0
    evaluated_rows = 0
    error_rows = 0

    for i, idx in enumerate(rows_to_process.index):
        try:
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, 'Generated RAG Answer']
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
                df.loc[idx, 'Answer Correctness for RAG run 1'] = None
                df.loc[idx, 'Evaluation Notes Answer Correctness for RAG'] = f"Exceeded context limit"
                skipped_rows += 1
                continue
            
            # Create dictionary for dataset
            # This is requeirement of RAGAS
            data_samples = {
                'question': [],
                'answer': [],
                'ground_truth': []
            }

            # Add data to dictionary
            data_samples['question'].append(df.loc[idx, 'Modified Questions'])
            data_samples['answer'].append(df.loc[idx, 'Generated RAG Answer'])
            data_samples['ground_truth'].append(df.loc[idx, 'Reasonings'])

            dataset = Dataset.from_dict(data_samples)

            scores = []
            for run_num in range(1, 4):
                llm_wrapper, embeddings_wrapper = setup_ragas_evaluator(model_name=model_name)
                score = evaluate(dataset,
                               metrics=[answer_correctness], 
                               llm=llm_wrapper,
                               embeddings=embeddings_wrapper)
                
                answer_correctness_score = score['answer_correctness'][0]
                scores.append(answer_correctness_score)
                
                # Update dataframe with individual run score
                df.loc[idx, f'Answer Correctness for RAG run {run_num}'] = answer_correctness_score
                print(f"Row {idx} - Run {run_num} Score: {answer_correctness_score}")
                
                time.sleep(timeout_seconds)

            # Calculate and store mean score
            mean_score = sum(scores) / len(scores)
            df.loc[idx, 'Mean Answer Correctness for RAG'] = mean_score
            print(f"Row {idx} - Mean Score: {mean_score}")
            df.loc[idx, 'Evaluation Notes Answer Correctness for RAG'] = f"Evaluated 3 runs"
            evaluated_rows += 1
        
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            df.loc[idx, 'Evaluation Notes Answer Correctness for RAG'] = f"Error"
            error_rows += 1
            

        # Save after each batch or when an error occurs
        if ((i + 1) % BATCH_SIZE == 0) or error_rows > 0:
            print(f"Saving progress...")
            data = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            error_rows = 0  # Reset error counter after saving


    # Report completion status
    remaining = len(df[(df['Generated RAG Answer'].notna()) & 
                      (df['Answer Correctness for RAG run 1'].isna()) &
                       (df['Evaluation Notes Answer Correctness for RAG'].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.
              Skipped {skipped_rows} rows due to context limit.
              Evaluated {evaluated_rows} rows.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")




def run_interchangeable_evaluation(file_path, model_name="gemini-2.5-flash", cycles=10, batch_size=5, timeout_seconds=10):
    """
    Run vanilla and RAG answer correctness evaluation interchangeably.
    
    Parameters:
    - file_path: str
        Path to the JSON file containing the dataset
    - model_name: str
        Name of the model to use for evaluation
    - cycles: int
        Number of times to repeat the vanilla->RAG cycle
    - batch_size: int
        Number of rows to process in each batch
    - timeout_seconds: int
        Timeout between evaluations
    """
    
    print(f"Starting interchangeable evaluation for {cycles} cycles with batch size {batch_size}")
    print(f"File: {file_path}")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    for cycle in range(1, cycles + 1):
        print(f"\n CYCLE {cycle}/{cycles}")
        print("-" * 40)
        
        # Run vanilla evaluation
        print(f"Running VANILLA evaluation (Cycle {cycle})")
        try:
            calculate_answer_correctness_vanilla(
                file_path=file_path,
                model_name=model_name,
                max_rows=batch_size,
                batch_size=batch_size,
                timeout_seconds=timeout_seconds
            )
            print(f"Vanilla evaluation completed for cycle {cycle}")
        except Exception as e:
            print(f"Error in vanilla evaluation cycle {cycle}: {str(e)}")
        
        print(f"Running RAG evaluation (Cycle {cycle})")
        try:
            calculate_answer_correctness_rag(
                file_path=file_path,
                model_name=model_name,
                max_rows=batch_size,
                batch_size=batch_size,
                timeout_seconds=timeout_seconds
            )
            print(f"RAG evaluation completed for cycle {cycle}")
        except Exception as e:
            print(f"Error in RAG evaluation cycle {cycle}: {str(e)}")
         
    
    print(f"All {cycles} cycles completed!")
    print("=" * 80)


# Example usage:

# LLAMA 4 SCOUT
LLAMA4_SCOUT = r"experiments\\test_dataset_together_meta-llama_Llama-4-Scout-17B-16E-Instruct_top5_answered.json"

run_interchangeable_evaluation(
     file_path=LLAMA4_SCOUT,
     model_name="gemini-2.5-flash",
     cycles=1,
     batch_size=10,
     timeout_seconds=0)


# Do not use this code with Geminin 2.5 flash model, it is too slow and expensive. 
# Use Gemini 2.0 flash model instead.