import os
import pandas as pd
from llama_index.llms.google_genai import GoogleGenAI
from ragas.llms import LlamaIndexLLMWrapper
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from ragas import EvaluationDataset
from ragas import SingleTurnSample
from datasets import Dataset



google_api_key = os.getenv("GOOGLE_API_KEY")

QUESTIONS_FILE = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-05 22.05.25 dataset for evaluation\\psychiatry_train_dataset_gemini_gemini-2.0-flash_answered.csv"

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


'''
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            reference=reference)
        
        score = evaluate(sample, metrics=[answer_correctness], llm=llm)
        result.append(score)
        

        
        #metric = answer_correctness(llm=llm)
        #result.append(metric.single_turn_score(sample))

 '''


def calculate_answer_correctness_vanilla(file_path):
    '''
    Parameters:
    - file_path: str
        Path to the CSV file containing the dataset with already answered questions 
        (produced by script for step 3 answering medical question using LLM).
    Output: 
    DataFrame
        The original DataFrame with an correctness score saved to column 'Answer Correctness for RAG'
    Functionality:
        https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_correctness.html

        The assessment of Answer Correctness involves gauging the accuracy of the generated answer 
        when compared to the ground truth. This evaluation relies on the ground truth and the answer, 
        with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated 
        answer and the ground truth, signifying better correctness.
        Answer correctness  is computed as the sum of factual correctness and the semantic similarity 
        between the given answer and the ground truth.



        Answer similarity is calculated by following steps:
        Step 1: Vectorize the ground truth answer using the specified embedding model.
        Step 2: Vectorize the generated answer using the same embedding model.
        Step 3: Compute the cosine similarity between the two vectors.
        
        https://docs.ragas.io/en/stable/references/embeddings/#ragas.embeddings.embedding_factory
        By default "text-embedding-ada-002" model is used.

        Final score is created by taking a weighted average of the factual correctness and the semantic similarity.     
    '''
    # Load data
    df = pd.read_csv(file_path)

    # Get rows that need evaluation
    vanilla_rows = df[(df['Generated Vanilla Answer'].notna()) & 
                      (df['Answer Correctness for vanilla'].isna())]
    print(f"Found {len(vanilla_rows)} vanilla answers to evaluate.")

    for idx in vanilla_rows.index.tolist():
        if pd.isna(df.loc[idx, 'Generated Vanilla Answer']) or pd.isna(df.loc[idx, 'Modified Questions']):
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
        data_samples['answer'].append(df.loc[idx, 'Generated Vanilla Answer'])
        data_samples['ground_truth'].append(df.loc[idx, 'Reasonings'])
        
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(dataset,metrics=[answer_correctness], llm=setup_ragas_evaluator(model_name="gemini-2.0-flash"))
        print(score)
        # Extract the score value from the dictionary (it's a dictionary with 'answer_correctness' key)
        answer_correctness_score = score['answer_correctness'][0]  # Get first (only) element
        
        # Update dataframe directly with the score value
        df.loc[idx, 'Answer Correctness for vanilla'] = answer_correctness_score
        # Problem - each run it produces slightly different values
        print(f"Row {idx} - Score: {answer_correctness_score}")

    return df

calculate_answer_correctness_vanilla(QUESTIONS_FILE)


def calculate_answer_correctness_rag(file_path):
    '''
    Parameters:
    - file_path: str
        Path to the CSV file containing the dataset with already answered questions 
        (produced by script for step 3 answering medical question using LLM).
    Output: 
    DataFrame
        The original DataFrame with an correctness score saved to column 'Answer Correctness for RAG'
    Functionality:
        https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_correctness.html

        The assessment of Answer Correctness involves gauging the accuracy of the generated answer 
        when compared to the ground truth. This evaluation relies on the ground truth and the answer, 
        with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated 
        answer and the ground truth, signifying better correctness.
        Answer correctness  is computed as the sum of factual correctness and the semantic similarity 
        between the given answer and the ground truth.



        Answer similarity is calculated by following steps:
        Step 1: Vectorize the ground truth answer using the specified embedding model.
        Step 2: Vectorize the generated answer using the same embedding model.
        Step 3: Compute the cosine similarity between the two vectors.
        
        https://docs.ragas.io/en/stable/references/embeddings/#ragas.embeddings.embedding_factory
        By default "text-embedding-ada-002" model is used.

        Final score is created by taking a weighted average of the factual correctness and the semantic similarity.
    '''
    # Load data
    df = pd.read_csv(file_path)
    # Get rows that need evaluation
    rag_rows = df[(df['Generated RAG Answer'].notna()) & 
                      (df['Answer Correctness for RAG'].isna())]
    print(f"Found {len(rag_rows)} vanilla answers to evaluate.")

    for idx in rag_rows.index.tolist():
        if pd.isna(df.loc[idx, 'Generated RAG Answer']) or pd.isna(df.loc[idx, 'Modified Questions']):
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
        # Create dataset
        # This is requeirement of RAGAS
        dataset = Dataset.from_dict(data_samples)
        rag_score = evaluate(dataset,metrics=[answer_correctness])
        # Extract the score value from the dictionary (it's a dictionary with 'answer_correctness' key)
        answer_correctness_score_rag = rag_score['answer_correctness'][0]  # Get first (only) element
        
        # Update dataframe directly with the score value
        df.loc[idx, 'Answer Correctness for RAG'] = answer_correctness_score_rag
        
        print(f"Row {idx} - Score: {answer_correctness_score_rag}")

    return df







