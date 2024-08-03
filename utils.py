import time
import re
import pandas as pd
from tqdm import tqdm

def remove_release_number(data: pd.DataFrame,
                          column: str) -> pd.DataFrame:

    data[column] = [re.findall('(.*?)(?:\s+\[3GPP Release \d+]|$)', x)[0] for x in data[column]]
    return data


def get_option_5(row: pd.Series) -> str:

    option_5 = row['option 5']
    if pd.isna(option_5):
        option_5 = ''
    else:
        option_5 = f'E) {option_5}'
    return option_5


def encode_answer(answer: str | int,
                  encode_letter: bool = True) -> int | str:

    letter_to_number = {'A': 1,
                        'B': 2,
                        'C': 3,
                        'D': 4,
                        'E': 5}
    if encode_letter:
        encoded = letter_to_number[answer]
    else:
        number_to_letter = {y: x for x, y in letter_to_number.items()}
        encoded = number_to_letter[answer]
    return encoded


def rag(row: pd.Series, query_eng) -> str:

    query = row['question']
    response = query_eng.invoke(query)
    context = 'Context:\n'
    context = context + " ".join([doc.page_content for doc in response]) + '\n'

    context = re.sub('\s+', ' ', context)
    return context


def generate_prompt(row: pd.Series,
                    context: str) -> str:
    prompt = f"""
    Provide a correct answer to a multiple choice question. Use only one option from A, B, C, D or E.
    {row['question']}
    A) {row['option 1']}
    B) {row['option 2']}
    C) {row['option 3']}
    D) {row['option 4']}
    {get_option_5(row)}
    {context}
    Answer:
    """
    return prompt

def llm_inference(data: pd.DataFrame, model, 
                  tokenizer, retriever,
                  store_wrong: bool = False, sleep_time: int = 2) -> tuple[pd.DataFrame, list]:
    
    wrong_format = []
    answers = []

    for _, question_batch in tqdm(data.iterrows(), total=len(data)):
        prompt_context = rag(question_batch, retriever)
        prompt = generate_prompt(question_batch, prompt_context)
        print(f"\n{question_batch['Question_ID']}")
                
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[-1]+1, pad_token_id=tokenizer.eos_token_id)
        answer_letter = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):len(prompt)+1]
        
        try:
            answer = encode_answer(answer_letter)
        except:
            try:
                print(f"Question {question_batch['Question_ID']} output was improper ({answer_letter})! Checking if it wasn't because of spaces...")
                outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[-1]+4, pad_token_id=tokenizer.eos_token_id)
                full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                print(f'Full output:\n{full_output}')
                answer_letter = re.findall('(A|B|C|D|E)', full_output[len(prompt)-5:len(prompt)+5])[0]
                answer = encode_answer(answer_letter)
                print(f'New answer: {answer}')
            except:
                print(f"Question {question_batch['Question_ID']} output was improper ({answer_letter})! Changing answer to 1")
                answer = 1
                outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[-1]+20, pad_token_id=tokenizer.eos_token_id)
                answer_letter = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                print(answer_letter)
                if store_wrong:
                    wrong_format.append([question_batch['Question_ID'], answer_letter])
        
        answers.append([question_batch['Question_ID'], answer])
        
    answers_df = pd.DataFrame(answers, columns=['Question_ID', 'Answer_ID'])
    return answers_df, wrong_format
