from llama_index.llms.google_genai import GoogleGenAI
import os
import pandas as pd
import time 
import json

google_api_key = os.getenv("GOOGLE_API_KEY")

# Load the verified dataset
with open("data\\2025-06 Step 0 Dataset preparation\\MedQA_open_dataset_classified_psychiatry_evaluation.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
verified_df = pd.DataFrame(data)
verified_df = verified_df[verified_df["psychiatry_classification"] == "include"].reset_index(drop=True)
columns_to_exclude = ['evaluation_status', 'psychiatry_classification']
verified_df = verified_df.drop(columns=columns_to_exclude)


def classify_question_category(question: str, model_name="gemini-2.5-pro"):
    """
    Script for identifying the topic of psychiatric question

    """
    prompt = f"""
    Act as an experienced mental health specialist. Your task is to classify a provided psychiatric question.

    Question: "{question}"

    Instructions:
    1. Carefully analyze the question to identify the mental health condition or disorder being discussed
    2. Classify it into ONE of the following categories:
       • Anxiety Disorders
       • Bipolar Disorders  
       • Depressive Disorders
       • Dissociative Disorders
       • Eating Disorders
       • Obsessive-Compulsive Disorders
       • Personality Disorders
       • Schizophrenia Spectrum and Other Psychotic Disorders
       • Somatic Disorders
       • Trauma and Stressor Related Disorders
       • Other Mental Disorders

    3. Provide your response in the following JSON format:
    {{
        "reasoning": "Brief explanation of why this question fits the selected category, including any key symptoms, conditions, or diagnostic criteria mentioned",
        "category": "Selected category name exactly as listed above",
        "confidence": "high/medium/low"
    }}

    Important notes:
    - If the question doesn't clearly fit into categories 1-10, use "Other Mental Disorders"
    - Focus on the primary disorder being discussed
    - Use exact category names as provided
    - Be concise but thorough in your reasoning
    """

    valid_categories = [
        "Anxiety Disorders",
        "Bipolar Disorders", 
        "Depressive Disorders",
        "Dissociative Disorders",
        "Eating Disorders",
        "Obsessive-Compulsive Disorders",
        "Personality Disorders",
        "Schizophrenia Spectrum and Other Psychotic Disorders",
        "Somatic Disorders",
        "Trauma and Stressor Related Disorders",
        "Other Mental Disorders"
    ]

    try:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        response = llm.complete(prompt)
        result_text = response.text.strip()
        print(f"Response: {result_text}")

        # Clean up the response by removing markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]  # Remove ```json
        if result_text.startswith("```"):
            result_text = result_text[3:]   # Remove ```
        if result_text.endswith("```"):
            result_text = result_text[:-3]  # Remove trailing ```

        # Strip any remaining whitespace
        result_text = result_text.strip()
        parsed_result = json.loads(result_text)
        
        # Validate category
        if parsed_result.get("category") in valid_categories:
            return parsed_result
        
        else:
            # Handle invalid category
            invalid_category = parsed_result.get("category", "unknown")
            print(f"Warning: Invalid category returned: '{invalid_category}'")
            print(f"Raw response: {result_text}")
            
            # Return a corrected result with "Other Mental Disorders" as default
            return {
                "reasoning": f"Model returned invalid category '{invalid_category}'. Original reasoning: {parsed_result.get('reasoning', 'No reasoning provided')}",
                "category": "Other Mental Disorders",
                "confidence": "low"
            }

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {result_text}")
        return {
            "reasoning": "Failed to parse model response",
            "category": "error",
            "confidence": "low"
        }
    except Exception as e:
        print(f"Error classifying question: {e}")
        return {
            "reasoning": "Error occurred during classification",
            "category": "error",
            "confidence": "low"
        }


def process_file_json(file_path, batch_size=10, max_rows=150, timeout_interval=25, timeout_seconds=30):
    """
    
    """
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".json", "_with_categories.json")

    # Check if the output JSON file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing JSON file: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Convert back to DataFrame for processing
        df = pd.DataFrame(results_data)
    else:
        print(f"Starting new evaluation on: {FILE_PATH}")
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        df = pd.DataFrame(results_data)
        df = df[df["psychiatry_classification"] == "include"].reset_index(drop=True)
        columns_to_exclude = ['evaluation_status', 'psychiatry_classification']
        df = df.drop(columns=columns_to_exclude)


        # Initialize the evaluation columns
        df['psychiatric_category'] = None
        df['category_reasoning'] = None
        df['category_confidence'] = None



    # Get rows that don't have evaluation yet
    rows_to_process = df[df['psychiatric_category'].isna()].index.tolist()

    # Limit to max_rows
    rows_to_process = rows_to_process[:max_rows]
    total_rows = len(rows_to_process)
        
    print(f"Found {len(df[df['psychiatric_category'].isna()])} unevaluated questions total")
    print(f"Will process {total_rows} questions in this run (max_rows={max_rows})")
    
    for i, idx in enumerate(rows_to_process):
        # Get question
        question = df.loc[idx, 'Modified Questions']
            
        # Skip if empty
        if pd.isna(question):
            df.loc[idx, 'psychiatric_category'] = "invalid"
            df.loc[idx, 'category_reasoning'] = "Empty question"
            df.loc[idx, 'category_confidence'] = "error"
            continue
            
        # Evaluate the question using is_clinical_psychiatry_focused
        evaluation_result = classify_question_category(question, model_name="gemini-2.5-pro")
        
        # Extract results from the returned dictionary
        df.loc[idx, 'psychiatric_category'] = evaluation_result.get('category', 'error')
        df.loc[idx, 'category_reasoning'] = evaluation_result.get('reasoning', 'No reasoning provided')
        df.loc[idx, 'category_confidence'] = evaluation_result.get('confidence', 'unknown')

        # Handle API errors
        if evaluation_result.get('category') == 'error':
            print(f"Error encountered at row {idx}. Continuing...")
        
        # Update progress
        classification = evaluation_result.get('category', 'error')
        print(f"Processed {i+1}/{total_rows}: Row {idx} - {classification}")

        # Save after each batch
        if (i + 1) % BATCH_SIZE == 0:
            print(f"Saving progress after batch...")
            # Convert DataFrame to JSON and save
            results_json = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        # Add timeout after specified interval
        if (i + 1) % timeout_interval == 0 and i+1 < total_rows:
            print(f"Taking a {timeout_seconds} second break to avoid rate limits...")
            time.sleep(timeout_seconds)

    # Final save
    print("Saving final results...")
    results_json = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # Report completion status
    remaining = len(df[df['psychiatric_category'].isna()])
    if remaining > 0:
        print(f"Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.")
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")

    # Get all classified questions (not null)
    classified_df = df[df['psychiatric_category'].notna()]
    
    if len(classified_df) > 0:
        # Count categories
        category_counts = classified_df['psychiatric_category'].value_counts()
        
        print(f"Total classified questions: {len(classified_df)}")
        print(category_counts)

    return df


    

FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-06 Step 0 Dataset preparation\\MedQA_open_dataset_classified_psychiatry_evaluation.json"


process_file_json(FILE_PATH, batch_size=10, max_rows=732, timeout_interval=10, timeout_seconds=0)











