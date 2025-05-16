from google import genai
import os
import pandas as pd
import time 

google_api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=google_api_key)
dataset_path = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-05 08.05.25 dataset for classification\\MedQA_open_dataset_classified.xlsx"

df = pd.read_excel(dataset_path)


psychiatric_keywords = [
    'ptsd',
    'major depressive disorder', 
    'anxiety disorder', 
    'schizophrenia', 
    'bipolar disorder', 
    'psychosis', 
    'mood disorder', 
    'adhd', 
    'autism',
    'eating disorder', 
    'bulimia', 
    'acute stress disorder', 
    'ocd', 
    'panic attack', 
    'personality disorder'
]


def classify_question_category(question):
    """
    Script for identifying the topic of psychiatric question
    """
    prompt = f"""

    Act as experienced mental health specialist. Your task is to classify a provided psychiatry question into one of the following categories:
    1.	Anxiety Disorders
    2.	Bipolar Disorders
    3.	Depressive Disorders
    4.	Dissociative Disorders
    5.	Eating Disorders
    6.	Obsessive-Compulsive Disorders
    7.	Personality Disorders
    8.	Schizophrenia Spectrum and Other Psychotic Disorders
    9.	Somatic Disorders
    10.	Trauma and Stressor Related Disorders
    11. Other Mental Disorders

    Question: {question}
    Respond with only one category for each question. Your answer should consist of only one word. If you are not sure about the answer, 
    please choose the "Other Mental Disorders" category.
    """

    psychiatric_categories_keywords = [
    "anxiety disorders",
    "bipolar disorders",
    "depressive disorders",
    "dissociative disorders",
    "eating disorders",
    "obsessive-compulsive disorders",
    "personality disorders",
    "schizophrenia spectrum and other psychotic disorders",
    "somatic disorders",
    "trauma and stressor related disorders",
    "other mental disorders"]
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        result = response.text.strip().lower()
        
        # Ensure the response is either 'psychiatry' or 'non-psychiatry'
        if result in psychiatric_categories_keywords:
            print(f"Question: {question}\nClassification: {result}\n")
        
            return result
        
    except Exception as e:
        print(f"Error classifying question: {e}")
        return None
    

df = df[df['classification'] == 'psychiatry']
df = df.reset_index(drop=True)
df['category'] = None
print(len(df))


question = df.loc[20, 'Modified Questions']
answer = df.loc[20, 'Reasonings']
            
#classify_question_category(question)
print(answer)
        

        
        

