import os
import json
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from llama_index.llms.google_genai import GoogleGenAI
from ragas.llms import LlamaIndexLLMWrapper
from ragas import evaluate
from ragas.metrics import answer_correctness
from datasets import Dataset
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

# Step 1: Import required libraries
"""
- asyncio: For asynchronous programming
- concurrent.futures.ThreadPoolExecutor: To run blocking RAGAS calls in threads
- json: To read JSON files
- pandas: For DataFrame operations
- All RAGAS and LlamaIndex imports: For evaluation setup
"""

# Step 2: Constants and configuration
google_api_key = os.getenv("GOOGLE_API_KEY")

# Constants for context window sizes
MODEL_CONTEXT_LIMITS = {
    "gemini-2.0-flash": 1048576,  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash
    "gemini-2.0-flash-lite": 1048576,
    "gemini-2.5-flash":1048576  # Maximum input tokens: 1,048,576; Maximum output tokens: 8,192; https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash-lite
}

# Add timeout constants
EVALUATION_TIMEOUT = 60  # 2 minutes per evaluation
MAX_RETRIES = 2


# Step 3: Helper functions from your original code
def setup_ragas_evaluator(model_name="gemini-2.0-flash"):
    """
    Initialize the RAGAS evaluator with Google Gemini using LlamaIndex.
    This function creates the LLM and embedding wrappers needed for RAGAS.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Create LlamaIndex GoogleGenAI instance
    gemini_llm = GoogleGenAI(
        model=model_name,
        api_key=google_api_key
    )

    gemini_embeddings = GoogleGenAIEmbedding(
        model_name="models/text-embedding-004",
        api_key=google_api_key
    )
    
    # Wrap both with RAGAS adapters
    llm_wrapper = LlamaIndexLLMWrapper(gemini_llm)
    embeddings_wrapper = LlamaIndexEmbeddingsWrapper(gemini_embeddings)
    
    return llm_wrapper, embeddings_wrapper

def count_tokens(text):
    """
    Count approximate tokens using 4 characters per token heuristic.
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text) // 4 + (1 if len(text) % 4 else 0)


def run_ragas_evaluation(question, answer, ground_truth, model_name):
    """
    Synchronous wrapper for RAGAS evaluation.
    This will be run in a thread pool to avoid blocking the async event loop.
    
    Args:
        question (str): The question text
        answer (str): The generated answer to evaluate
        ground_truth (str): The reference/correct answer
        model_name (str): Model name for evaluation
    
    Returns:
        float: Answer correctness score (0-1)
    """
    try:
        # Create RAGAS dataset format
        data_samples = {
            'question': [question],
            'answer': [answer],
            'ground_truth': [ground_truth]
        }
        dataset = Dataset.from_dict(data_samples)
        
        # Setup evaluator
        llm_wrapper, embeddings_wrapper = setup_ragas_evaluator(model_name=model_name)
        
        # Run evaluation
        score = evaluate(
            dataset,
            metrics=[answer_correctness], 
            llm=llm_wrapper,
            embeddings=embeddings_wrapper
        )
        
        return score['answer_correctness'][0]
    
    except Exception as e:
        print(f"RAGAS evaluation failed: {str(e)}")
        raise


# Step 4: Main async function
async def calculate_answer_correctness_vanilla_async(file_path, model_name="gemini-2.5-flash", 
                                                   max_rows=20, max_concurrent=3):
    """
    Async evaluation of vanilla answers from JSON file.
    
    Args:
        file_path (str): Path to JSON file containing the dataset
        model_name (str): Model to use for evaluation
        max_rows (int): Maximum number of rows to process
        max_concurrent (int): Maximum concurrent evaluations
    
    Returns:
        pd.DataFrame: Updated dataframe with evaluation results
    """
    
    # Step 5: Setup file paths and load data
    print(f"Loading data from JSON file: {file_path}")
    OUTPUT_PATH = file_path.replace(".json", "_vanilla_answer_correctness_evaluated.json")
    
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
        
        # Add new columns for evaluation results
        df['Answer Correctness for vanilla run 1'] = None
        df['Answer Correctness for vanilla run 2'] = None
        df['Answer Correctness for vanilla run 3'] = None
        df['Mean Answer Correctness for vanilla'] = None
        df['Evaluation Notes Answer Correctness for vanilla'] = None
        df['Evaluation Model Answer Correctness for vanilla'] = model_name

    # Step 6: Get model context limit
    model_context_limit = MODEL_CONTEXT_LIMITS.get(model_name, 128000)
    print(f"Using context limit: {model_context_limit} tokens for model: {model_name}")

    # Step 7: Filter rows that need evaluation
    vanilla_rows = df[
        (df['Generated Vanilla Answer'].notna()) & 
        (df['Answer Correctness for vanilla run 1'].isna()) &
        (df['Evaluation Notes Answer Correctness for vanilla'].isna())
    ][:max_rows]
    
    total_rows = len(vanilla_rows)
    print(f"Found {total_rows} vanilla answers to evaluate.")
    
    if total_rows == 0:
        print("No rows to evaluate!")
        return df

    # Step 8: Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    print(f"Set up semaphore with max {max_concurrent} concurrent evaluations")
    
    # Step 9: Define async evaluation function with semaphore
    async def evaluate_with_semaphore(idx, question, answer, ground_truth, run_num):
        """
        Async wrapper that runs RAGAS evaluation with semaphore control.
        
        Args:
            idx: DataFrame index
            question, answer, ground_truth: Text data for evaluation
            run_num: Which run (1, 2, or 3) this is
            
        Returns:
            tuple: (idx, run_num, score, error_message)
        """
        async with semaphore:
            print(f"Starting evaluation for row {idx}, run {run_num}")
            
            for attempt in range(MAX_RETRIES + 1):
                try:
                    # Add timeout to the entire evaluation
                    loop = asyncio.get_event_loop()
                    
                    score = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor,
                            run_ragas_evaluation,
                            question, answer, ground_truth, model_name
                        ),
                        timeout=EVALUATION_TIMEOUT
                    )
                    
                    print(f"Completed row {idx}, run {run_num}, score: {score:.4f}")
                    return idx, run_num, score, None
                    
                except asyncio.TimeoutError:
                    print(f"Timeout on row {idx}, run {run_num}, attempt {attempt + 1}")
                    if attempt == MAX_RETRIES:
                        return idx, run_num, None, f"Timeout after {MAX_RETRIES + 1} attempts"
                    await asyncio.sleep(5)  # Wait before retry
                    
                except Exception as e:
                    print(f"Error in row {idx}, run {run_num}, attempt {attempt + 1}: {str(e)}")
                    if attempt == MAX_RETRIES:
                        return idx, run_num, None, str(e)
                    await asyncio.sleep(2)  # Wait before retry

    # Step 10: Create all evaluation tasks
    tasks = []
    skipped_rows = 0
    
    for _, row in vanilla_rows.iterrows():
        idx = row.name
        question = df.loc[idx, 'Modified Questions']
        answer = df.loc[idx, 'Generated Vanilla Answer']
        ground_truth = df.loc[idx, 'Reasonings']
        
        # Check token limits before creating tasks
        total_tokens = (count_tokens(question) + 
                      count_tokens(answer) + 
                      count_tokens(ground_truth))
        
        if total_tokens > model_context_limit:
            print(f"Row {idx} exceeds context limit ({total_tokens} tokens). Skipping.")
            df.loc[idx, 'Evaluation Notes Answer Correctness for vanilla'] = "Exceeded context limit"
            skipped_rows += 1
            continue
        
        # Create 3 evaluation tasks for each valid row
        for run_num in range(1, 4):
            task = evaluate_with_semaphore(idx, question, answer, ground_truth, run_num)
            tasks.append(task)

    print(f"Created {len(tasks)} evaluation tasks. Skipped {skipped_rows} rows due to context limits.")
    
    if not tasks:
        print("No tasks to run!")
        executor.shutdown()
        return df
    print(f"Created {len(tasks)} evaluation tasks. Skipped {skipped_rows} rows due to context limits.")
    
   
        
    print(f"Starting {len(tasks)} evaluations with max {max_concurrent} concurrent...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Step 12: Process results
    row_scores = {}  # Dictionary to collect scores by row index
    error_count = 0
    
    for result in results:
        # Handle exceptions from asyncio.gather
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            error_count += 1
            continue
        
        idx, run_num, score, error = result
        
        if error:
            # Handle evaluation errors
            df.loc[idx, 'Evaluation Notes Answer Correctness for vanilla'] = "Error during evaluation"
            error_count += 1
        else:
            # Store successful evaluation
            df.loc[idx, f'Answer Correctness for vanilla run {run_num}'] = score
            
            # Collect scores for mean calculation
            if idx not in row_scores:
                row_scores[idx] = []
            row_scores[idx].append(score)

    # Step 13: Calculate mean scores for completed evaluations
    completed_evaluations = 0
    for idx, scores in row_scores.items():
        if len(scores) == 3:  # All 3 runs completed successfully
            mean_score = sum(scores) / 3
            df.loc[idx, 'Mean Answer Correctness for vanilla'] = mean_score
            df.loc[idx, 'Evaluation Notes Answer Correctness for vanilla'] = "Evaluated 3 runs"
            completed_evaluations += 1
            print(f"Row {idx} - Mean Score: {mean_score:.4f}")
        else:
            print(f"Row {idx} - Incomplete evaluation ({len(scores)}/3 runs)")

    # Step 14: Save results to JSON
    print(f"Saving results to: {OUTPUT_PATH}")
    result_data = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # Step 15: Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total rows processed: {total_rows}")
    print(f"Completed evaluations: {completed_evaluations}")
    print(f"Skipped (context limit): {skipped_rows}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {OUTPUT_PATH}")
    
    return df


# Step 16: Usage example
async def main():
    """
    Main function to run the async evaluation.
    """
    # Update this path to your JSON file
    DATASET_FILE = r"experiments\\test_dataset_together_meta-llama_Llama-4-Scout-17B-16E-Instruct_top5_answered.json"
    try:
        result = await calculate_answer_correctness_vanilla_async(
            file_path=DATASET_FILE,
            model_name="gemini-2.5-flash",
            max_rows=400,
            max_concurrent=2  # Adjust based on your API limits
        )
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

# Step 17: Run the async function
if __name__ == "__main__":
    asyncio.run(main())