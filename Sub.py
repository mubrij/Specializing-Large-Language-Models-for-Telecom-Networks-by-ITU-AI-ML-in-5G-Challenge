import os
import json
import re
from transformers import pipeline
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

os.environ['OPENAI_API_KEY'] = "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"
embeddings = OpenAIEmbeddings()
model = ChatOpenAI()

model_name = "microsoft/phi-2"
pipe = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
    trust_remote_code=True,
)
# Update file paths relative to /telecom
input_path = "input/rag-for-telecom-networks/SampleSubmission (22).csv"
ss = pd.read_csv(input_path)

file_path1 = 'input/rag-for-telecom-networks/TeleQnA_testing1.txt'
file_path = 'input/rag-for-telecom-networks/questions_new.txt'

loader = DirectoryLoader('input/tex-data', glob="**/*.txt")
docs = loader.load()
print("Done loading docs.....................")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(docs)

vec_db = FAISS.from_documents(docs, embeddings)
print("Done creating vector store.....................")

def split_and_clean_text(input_text):
    return [item for item in re.split(r"<<|>>", input_text) if item.strip()]

def flatten_and_unique_documents(documents):
    flattened_docs = [doc for sublist in documents for doc in sublist]
    return list(dict.fromkeys(flattened_docs))

def extract_first_digit(text):
    match = re.search(r'\d', text)
    return match.group() if match else 0

def gen_doc_retrieval(query):
    docs = []
    query_docs = vec_db.similarity_search(query)
    for doc in query_docs:
        docs.append(doc.page_content)
    return docs

def create_prompt(question_data, context):
    prompt = (
        "You are a top-tier expert in telecommunications and standards specifications, preparing for a highly competitive and expensive exam worth 10,000 USD. Carefully read the question and the provided context, then choose the correct answer from the options given.\n\n"
        "### Context:\n"
        f"{context}\n\n"
        "### Question:\n"
        f"{question_data['question']}\n\n"
        "### Options:\n"
    )
    
    option_keys = [key for key in question_data if key.startswith('option ')]
    for key in sorted(option_keys):
        option_label = key.split()[1].upper()  
        prompt += f"{option_label}. {question_data[key]}\n"
        
    prompt += (
        "\n**Instruction**: When answering the question, start by saying 'The correct option is [Option Label]'. "
        "Provide a brief explanation for your choice based on the given context."
    )
    return prompt

def prompt_llm(prompt):
    outputs = pipe(prompt, max_new_tokens=70)
    return(outputs[0]["generated_text"])

def multiquery(query):
    list_of_questions = hyde_chain.invoke(query)
    docs = [gen_doc_retrieval(q) for q in list_of_questions]
    
    docs = flatten_and_unique_documents(documents=docs)
    context = " ".join([doc for doc in docs])
    
    return context

HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five hypothetical answers to the user's query. These answers should offer diverse perspectives or interpretations, aiding in a comprehensive understanding of the query. Present the hypothetical answers as follows:

    <<Answer considering a specific perspective>>
    <<Answer from a different angle>>
    <<Answer exploring an alternative possibility>>
    <<Answer providing a contrasting viewpoint>>
    <<Answer that includes a unique insight>>

    Note: Present only the hypothetical answers, without numbering (or "-", "1.", "*") and so on, to provide a range of potential interpretations or solutions related to the query.
    Original question: {question}""",
)

hyde_chain = (
    HYDE_PROMPT | model | StrOutputParser() | RunnableLambda(split_and_clean_text)
)

with open(file_path1, 'r') as file:
    data = json.load(file)

with open(file_path, 'r') as file:
    data.update(json.load(file))

test_qids = ss["Question_ID"].to_list()

from tqdm import tqdm
Answers = []
for qid in tqdm(test_qids):
    try:
        context = multiquery(data[f"question {qid}"])
        prompt = create_prompt(data[f"question {qid}"], context)
        answer = prompt_llm(prompt)
        Answers.append(answer.replace(prompt, "").strip())
    except:
        print(f"Failed ID no {qid}")
        Answers.append("The correct option is 0")

x = ss.copy()
x["Answer_ID"] = Answers
x.to_csv("Raw_text_answers.csv", index=False)

final_answers = []
for i, answer in enumerate(Answers):
    try:
        a = answer.split("The correct option is ")[-1]
        final_answers.append(int(extract_first_digit(a)))
    except:
        print(f"Failed no {i+1}")
        final_answers.append(0)

ss["Answer_ID"] = final_answers
ss["Task"] = "phi-2"

ss.to_csv("__submission__.csv", index=False)
