from ragas import evaluate
import os
import asyncio
from ragas.metrics import RubricsScore
from ragas.dataset_schema import SingleTurnSample
from llama_index.llms.google_genai import GoogleGenAI
from ragas.llms import LlamaIndexLLMWrapper
import json
import pandas as pd
import time


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
   
    # Wrap both with RAGAS adapters
    llm_wrapper = LlamaIndexLLMWrapper(gemini_llm)
    
    # Wrap with RAGAS LlamaIndexLLM adapter
    return llm_wrapper



def calculate_rubric_metric(file_path, model_name="gemini-2.0-flash",
                        max_rows=20, batch_size=1, timeout_seconds=10):
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", f"_rubric_score_evaluated.json")

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
        # Ensure necessary columns exist
        df['Vanilla Rubric Score'] = None
        df['RAG Rubric Score'] = None

    # Get model context window limit
    if model_name not in MODEL_CONTEXT_LIMITS:
        print(f"Warning: Unknown model '{model_name}'. Using default limit of 128000 tokens.")
        model_context_limit = 128000
    else:
        model_context_limit = MODEL_CONTEXT_LIMITS[model_name]


    my_psychiatric_rubrics = {
    "score1_description": "The answer is medically incorrect or contradicts established psychiatric knowledge; it does not address the question or ground truth at all.",
    "score2_description": "The answer contains significant inaccuracies or omissions; it only partially addresses the question and differs notably from the ground truth.",
    "score3_description": "The answer is generally correct but lacks important details or contains minor inaccuracies; it is somewhat aligned with the ground truth.",
    "score4_description": "The answer is medically accurate, covers most relevant aspects, and closely matches the ground truth with only minor differences.",
    "score5_description": "The answer is fully medically accurate, comprehensive, and matches the ground truth exactly in content and detail."}

    # Get rows that need evaluation
    rows = df[(df['Generated Vanilla Answer'].notna()) & 
                    (df["Vanilla Rubric Score"].isna())]
    
    # Track statistics
    skipped_rows = 0
    evaluated_rows = 0
    error_rows = 0
    
    # Limit to max_rows
    rows_to_process = rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} answers to evaluate for rubric score.")

    evaluator_llm = setup_ragas_evaluator(model_name="gemini-2.0-flash")
    scorer = RubricsScore(rubrics=my_psychiatric_rubrics, llm=evaluator_llm)

    for i, idx in enumerate(rows_to_process.index):
        try:
            rag_response = df.loc[idx, 'Generated RAG Answer']
            ground_truth = df.loc[idx, 'Reasonings']
               
            rag_answer_tokens = count_tokens(rag_response)
            ground_truth_tokens = count_tokens(ground_truth)
            total_tokens = rag_answer_tokens + ground_truth_tokens 

            if total_tokens > model_context_limit:
                print(f"Row {idx} exceeds context limit. Skipping evaluation.")
                df.loc[idx, 'RAG Rubric Score'] = None
                skipped_rows += 1
                continue

            rag_answer = SingleTurnSample(
                response=rag_response,
                reference=ground_truth)


            rag_rubric_score = asyncio.run(scorer.single_turn_ascore(rag_answer))
            print(f"Row{idx}:RAG Rubric Score:", rag_rubric_score)
            df.loc[idx, 'RAG Rubric Score'] = rag_rubric_score
            time.sleep(timeout_seconds)

            # Save vanilla rubric score if it exists

            vanilla_response = df.loc[idx, 'Generated RAG Answer']
            vanilla_answer_tokens = count_tokens(vanilla_response)
            total_tokens = vanilla_answer_tokens + ground_truth_tokens 

            if total_tokens > model_context_limit:
                print(f"Row {idx} exceeds context limit. Skipping evaluation.")
                df.loc[idx, 'Vanilla Rubric Score'] = None
                skipped_rows += 1
                continue

            vanilla_answer = SingleTurnSample(
                response=vanilla_response,
                reference=ground_truth)


            vanilla_rubric_score = asyncio.run(scorer.single_turn_ascore(vanilla_answer))
            print(f"Row{idx}: Vanilla Rubric Score:", vanilla_rubric_score)
            df.loc[idx, 'Vanilla Rubric Score'] = vanilla_rubric_score
            time.sleep(timeout_seconds)
            evaluated_rows += 1

            # Save after each batch or when an error occurs
            if ((i + 1) % BATCH_SIZE == 0) or error_rows > 0:
                print(f"Saving progress...")
                data = df.to_dict('records')
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                error_rows = 0  # Reset error counter after saving

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            df.loc[idx, 'RAG Rubric Score'] = None
            df.loc[idx," Vanilla Rubric Score"] = None
            error_rows += 1



    # Final save
    data = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    error_rows = 0  # Reset error counter after saving

    # Report completion status
    remaining = len(df[(df['Generated Vanilla Answer'].notna()) & 
                      (df['Vanilla Rubric Score'].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.
              Skipped {skipped_rows} rows due to context limit.
              Evaluated {evaluated_rows} rows.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")


# Main execution

gemma_3b = r'C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/gemma_3b/test_dataset_together_google_gemma-3n-E4B-it_top5_answered.json'
calculate_rubric_metric(file_path=gemma_3b, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

llama4_maverick = r'C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/Llama4_maverick/test_dataset_together_meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8_top5_answered.json'

calculate_rubric_metric(file_path=llama4_maverick, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

mistral_24b = r"C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/mistral24b/test_dataset_together_mistralai_Mistral-Small-24B-Instruct-2501_top5_answered.json"
calculate_rubric_metric(file_path=mistral_24b, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

qwen_25_72b = r"C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/qwen2.5_72b/test_dataset_together_Qwen_Qwen2.5-72B-Instruct-Turbo_top5_answered.json"
calculate_rubric_metric(file_path=qwen_25_72b, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

llama_scout = r"C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/Llama4_Scout/test_dataset_together_meta-llama_Llama-4-Scout-17B-16E-Instruct_top5_answered.json"
calculate_rubric_metric(file_path=llama_scout, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

deepseek_v3 = r"C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/deepseekv3/test_dataset_together_yan_deepseek-ai-deepseek-v3_top5_answered.json"
calculate_rubric_metric(file_path=deepseek_v3, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

llama_31_8b = r"C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/llama3.1_8b/test_dataset_together_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_top5_answered.json"
calculate_rubric_metric(file_path=llama_31_8b, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

exaone_32b = r"C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/exaone_35_32b/test_dataset_together_lgai_exaone-3-5-32b-instruct_top5_answered.json"
calculate_rubric_metric(file_path=exaone_32b, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)

qwen_3_235b = r"C:/Users/kuzne/Documents/Python_repo/2025_01_dissertation/2025_dissertation/experiments/qwen_3_235b/test_dataset_together_Qwen_Qwen3-235B-A22B-fp8-tput_top5_answered.json"
calculate_rubric_metric(file_path=qwen_3_235b, model_name="gemini-2.0-flash",
                        max_rows=370, batch_size=10, timeout_seconds=0)


'''

evaluator_llm, embedding_evaluator = setup_ragas_evaluator(model_name="gemini-2.0-flash")

answer = "The question states the patient is taking a medication that increases norepinephrine nerve stimulation and is experiencing dizziness, dry mouth, and constipation.  The context mentions anticholinergic effects causing dry mouth and constipation. Therefore, the medication prescribed likely has anticholinergic properties, which are often associated with increased norepinephrine stimulation.  However, the provided context does not list specific medications that increase norepinephrine.  Without a list of medications, it's impossible to definitively identify the drug.  The symptoms suggest an anticholinergic medication was likely prescribed."
reference = "Venlafaxine is a serotonin-norepinephrine reuptake inhibitor (SNRI), which increases the levels of both serotonin and norepinephrine neurotransmitters in the brain. It is commonly used to treat depression, anxiety disorders, and fatigue. The patient's symptoms of fatigue, low energy, difficulty concentrating, and excessive sleeping are consistent with a diagnosis of major depressive disorder, which would be treated with an SNRI like venlafaxine. The patient's lack of manic or hypomanic symptoms and absence of suicidal ideation or preoccupation with death also support this diagnosis. Additionally, the patient's laboratory results, including his normal thyroid-stimulating hormone level, rule out other potential causes of fatigue such as hypothyroidism. The side effects of venlafaxine, including dizziness, dry mouth, and constipation, are also consistent with the medication that was likely prescribed to the patient. Thus, the answer is Venlafaxine."

sample = SingleTurnSample(
    response=answer,
    reference=reference
)



my_psychiatric_rubrics = {
    "score1_description": "The answer is medically incorrect or contradicts established psychiatric knowledge; it does not address the question or ground truth at all.",
    "score2_description": "The answer contains significant inaccuracies or omissions; it only partially addresses the question and differs notably from the ground truth.",
    "score3_description": "The answer is generally correct but lacks important details or contains minor inaccuracies; it is somewhat aligned with the ground truth.",
    "score4_description": "The answer is medically accurate, covers most relevant aspects, and closely matches the ground truth with only minor differences.",
    "score5_description": "The answer is fully medically accurate, comprehensive, and matches the ground truth exactly in content and detail.",
}

scorer = RubricsScore(rubrics=my_psychiatric_rubrics, llm=evaluator_llm)

try:
    rubric_score = asyncio.run(scorer.single_turn_ascore(sample))
    print("Rubric Score:", rubric_score)
except Exception as e:
    print("Error during rubric scoring:", e)


    '''