from datasets import Dataset 
from ragas.metrics import answer_similarity
from ragas import evaluate
import pandas as pd
import os
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
import json
import time

google_api_key = os.getenv("GOOGLE_API_KEY")


def setup_ragas_evaluator(model_name="models/text-embedding-004"):
    """
    Initialize the RAGAS evaluator with Google Gemini using LlamaIndex.
    
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    model_name = model_name
    

    gemini_embeddings = GoogleGenAIEmbedding(
        model_name=model_name,  # Latest Gemini embedding model
        api_key=google_api_key
    )
    # Wrap both with RAGAS adapters
    embeddings_wrapper = LlamaIndexEmbeddingsWrapper(gemini_embeddings)
    # Wrap with RAGAS LlamaIndexLLM adapter
    return embeddings_wrapper


def calculate_answer_similarity_vanilla(file_path, model_name="models/text-embedding-004", max_rows=20, batch_size=1, timeout_seconds=10):
    '''
    Calculate answer similarity for vanilla answers in the dataset.
    '''
    
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", "_vanilla_answer_similarity_evaluated.json")

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
        df['Answer Semantic Similarity for vanilla'] = None
        df['Evaluation Model Answer Semantic Similarity for vanilla'] = model_name


    embedding_wrapper = setup_ragas_evaluator(model_name=model_name)

    # Get rows that need evaluation
    vanilla_rows = df[(df['Generated Vanilla Answer'].notna()) & 
                        (df['Answer Semantic Similarity for vanilla'].isna())]
        
    # Limit to max_rows
    rows_to_process = vanilla_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} vanilla answers to evaluate.")


    for i, idx in enumerate(rows_to_process.index):
        try: 
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, 'Generated Vanilla Answer']
            ground_truth = df.loc[idx, 'Reasonings']

            data_samples = {
                    'question': [question],
                    'answer': [answer],
                    'ground_truth': [ground_truth]
            }

            dataset = Dataset.from_dict(data_samples)

            score = evaluate(dataset,
                                metrics=[answer_similarity], 
                                embeddings=embedding_wrapper)
            
            answer_similarity_score = score.to_pandas()

            # Extract the semantic similarity score
            similarity_score = answer_similarity_score['semantic_similarity'].iloc[0]
            # Save to your DataFrame
            df.loc[idx, 'Answer Semantic Similarity for vanilla'] = similarity_score
            df.loc[idx, 'Evaluation Model Answer Semantic Similarity for vanilla'] = 'Google Gemini text-embedding-004'

            print(f"Row {idx} - Answer Similarity Score: {answer_similarity_score}")

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            df.loc[idx, 'Evaluation Notes Answer Similarity for vanilla'] = f"Error: {str(e)}"
            continue

        # Save after each batch or when an error occurs
        if ((i + 1) % BATCH_SIZE == 0):
            print(f"Saving progress...")
            data = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            time.sleep(timeout_seconds)

    # Final save
    print(f"Final save to {OUTPUT_PATH}...")    
    data = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Evaluation complete! Results saved to {OUTPUT_PATH}")

    # Report completion status
    remaining = len(df[(df['Generated Vanilla Answer'].notna()) & 
                        (df['Answer Semantic Similarity for vanilla'].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")





def calculate_answer_similarity_rag(file_path, model_name="models/text-embedding-004", max_rows=20, batch_size=1, timeout_seconds=0):
    '''
    Calculate answer similarity for RAG answers in the dataset.
    '''
    
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", "_rag_answer_similarity_evaluated.json")

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
        df['Answer Semantic Similarity for rag'] = None
        df['Evaluation Model Answer Semantic Similarity for rag'] = model_name


    embedding_wrapper = setup_ragas_evaluator(model_name=model_name)

    # Get rows that need evaluation
    rag_rows = df[(df['Generated RAG Answer'].notna()) & 
                    (df['Answer Semantic Similarity for rag'].isna())]
        
    # Limit to max_rows
    rows_to_process = rag_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} RAG answers to evaluate.")


    for i, idx in enumerate(rows_to_process.index):
        try: 
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, 'Generated RAG Answer']
            ground_truth = df.loc[idx, 'Reasonings']

            data_samples = {
                    'question': [question],
                    'answer': [answer],
                    'ground_truth': [ground_truth]
            }

            # Why can't we just create a Dataset directly from the DataFrame?
            # Because RAGAS evaluate expects a Dataset or EvaluationDataset as an input.
            # If we define a Dataset before the loop, we cannot iterate over it since all rows will be processed in one run.

            dataset = Dataset.from_dict(data_samples)
            score = evaluate(dataset,
                                metrics=[answer_similarity], 
                                embeddings=embedding_wrapper)

            answer_similarity_score = score.to_pandas()

            # Extract the semantic similarity score
            similarity_score = answer_similarity_score['semantic_similarity'].iloc[0]
            # Save to your DataFrame
            df.loc[idx, 'Answer Semantic Similarity for rag'] = similarity_score
            df.loc[idx, 'Evaluation Model Answer Semantic Similarity for rag'] = 'Google Gemini text-embedding-004'

            print(f"Row {idx} - Answer Similarity Score: {similarity_score}")

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            df.loc[idx, 'Evaluation Notes Answer Similarity for rag'] = f"Error: {str(e)}"
            continue

        # Save after each batch or when an error occurs
        if ((i + 1) % BATCH_SIZE == 0):
            print(f"Saving progress...")
            data = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            time.sleep(timeout_seconds)

    # Final save
    print(f"Final save to {OUTPUT_PATH}...")    
    data = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Evaluation complete! Results saved to {OUTPUT_PATH}")

    # Report completion status
    remaining = len(df[(df['Generated RAG Answer'].notna()) & 
                        (df['Answer Semantic Similarity for rag'].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")


# Function call for RAG evaluation

llama_3b = "experiments/naive_rag/Llama 3.2 2B/test_dataset_together_meta-llama_Llama-3.2-3B-Instruct-Turbo_top3_answered.json"


calculate_answer_similarity_vanilla(llama_3b, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)

calculate_answer_similarity_rag(llama_3b, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)


MISTRAL = 'experiments/naive_rag/Mistral_24B/test_dataset_together_mistralai_Mistral-Small-24B-Instruct-2501_top3_answered.json'

calculate_answer_similarity_vanilla(MISTRAL, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)

calculate_answer_similarity_rag(MISTRAL, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)


QWEN_3 = "experiments/naive_rag/Qwen_3_275B/test_dataset_together_Qwen_Qwen3-235B-A22B-fp8-tput_top3_answered.json"

calculate_answer_similarity_vanilla(QWEN_3, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)
calculate_answer_similarity_rag(QWEN_3, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)

QWEN_2 = "experiments/naive_rag/Qwen_3_275B/test_dataset_together_Qwen_Qwen3-235B-A22B-fp8-tput_top3_answered.json"

calculate_answer_similarity_vanilla(QWEN_2, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)
calculate_answer_similarity_rag(QWEN_2, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)

kimi_2 = "experiments/advanced_rag/Kimi K2/test_dataset_together_moonshotai_Kimi-K2-Instruct_top5_answered.json"

calculate_answer_similarity_vanilla(kimi_2, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)
calculate_answer_similarity_rag(kimi_2, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)

gemma_3n = "experiments/naive_rag/Gemma 3n/test_dataset_together_google_gemma-3n-E4B-it_top3_answered.json"
calculate_answer_similarity_vanilla(gemma_3n, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)
calculate_answer_similarity_rag(gemma_3n, model_name="models/text-embedding-004", max_rows=370, batch_size=10, timeout_seconds=0)


