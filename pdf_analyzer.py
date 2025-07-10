from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_deepseek import ChatDeepSeek

import json

import result_to_markdown
import argparse


def load_queries(by_section=False):
    queries = list()
    query_separator = "\n" if not by_section else "### "
    with open('queries.txt', 'r') as f:
        query = list()
        for line in f:
            if line.startswith(query_separator):
                # This is a new query
                if query:
                    queries.append("".join(query).strip())
                    query = list()
            elif line.startswith('**Row'):
                # This is the dummy line, skip it
                continue
            else:
                query.append(line)
    
    print(f"Loaded {len(queries)} queries from the file.")
    for i, q in enumerate(queries):
        print(f"\nQuery {i + 1}:-------------------------------------------\n{q}")
    return queries

def format_docs(docs):
    # for i, doc in enumerate(docs):
    #     # Print the document content and metadata
    #     print(f"Document {i + 1} --------- \n meta {doc.metadata} \n---------content:\n{doc.page_content}")

    # Format the documents with page labels and content
    ret = "\n\n".join(f'Page: {doc.metadata["page_label"]}\n{doc.page_content}' for doc in docs)
    # print(ret)
    return ret

def save_final_prompt(prompt, filename):
    """
    Saves the final prompt to a text file.
    
    Args:
        prompt (PromptTemplate): The prompt template to save
        filename (str): The name of the file to save the prompt to
    """
    with open(filename, 'a') as f:
        f.write("\n**********************************************************************\n")
        f.write(prompt.to_string())
    print(f"Prompt saved to {filename}")
    return prompt

def intialize_chain(document_fn, prompt_fn):

    # embeddings = OllamaEmbeddings(model="znbang/bge:small-en-v1.5-f32")
    embeddings = OllamaEmbeddings(model="dengcao/Qwen3-Embedding-0.6B:F16")

    # ! Replace the API key with your actual DeepSeek API key
    from api_key import deepseek_api_key  # Import the API key from a separate file
    llm = ChatDeepSeek(
        api_key=deepseek_api_key,  # 替换为实际 API Key
        base_url="https://api.deepseek.com/v1",  # DeepSeek API 地址
        # model="deepseek-chat",  # 可选模型：deepseek-chat / deepseek-coder
        model="deepseek-reasoner",  # 可选模型：deepseek-chat / deepseek-coder
        temperature=0.2,
        model_kwargs={
            "response_format": {
                "type": "json_object",
            }
        }
    )


    # num_predict = 4096  # Number of tokens to predict
    # llm = OllamaLLM(model="deepseek-r1:8b",
    #                 num_predict=num_predict)

    # 2. Load the PDF file and create a retriever to be used for providing context
    loader = PyPDFLoader(document_fn)
    pages = loader.load_and_split()
    store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    num_pages = 5
    retriever = store.as_retriever(search_kwargs={'k': num_pages})

    # 3. Create the prompt template
    template = """
Context: {context}
Questions: {question}

Given the context provided, answer each question using the guidance with the question.
Also, include a <think> process that explains how you arrived at the answer.
Respond with a json object containing a list of answer and a brief explanation with page number to each question.
The answer should be binary (1 for yes, 0 for no).

Example response format:
{{
    "answers": [
        {{
                "question": "Does the report provide quantitative data on Scope 1 GHG emissions?",
                "answer": 1,
                "explanation": "Yes. The report provides FY2022 and FY2023 data for Scope 1 GHG emissions (Page 61)."
        }},
    ]
}}
"""

    prompt = PromptTemplate.from_template(template)

    # 4. Build the chain of operations
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(lambda x: save_final_prompt(x, prompt_fn))
        | llm
    )

    return chain

def get_output_fn(document_fn):
    assert(document_fn.endswith('.pdf'))
    from utils import replace_suffix
    stem = replace_suffix(document_fn, '.pdf', '')

    from utils import generate_unique_filename
    prompt_fn = generate_unique_filename(
        base_name=f'{stem}.prompt',
        extension='txt',
        start_index=1,
        separator='_',
        padding=3
    )
    result_fn = generate_unique_filename(
        base_name=f'{stem}.result',
        extension='json',
        start_index=1,
        separator='_',
        padding=3
    )
    markdown_fn = generate_unique_filename(
        base_name=f'{stem}.result',
        extension='md',
        start_index=1,
        separator='_',
        padding=3
    )
    return prompt_fn, result_fn, markdown_fn
    

def analyze_document(document_fn, queries):

    prompt_fn, result_fn, markdown_fn = get_output_fn(document_fn)
    chain = intialize_chain(document_fn, prompt_fn)
    
    print("****************************************************")
    print(f'Starting the analyze {document_fn}\n')
    final_results = []
    total_question_idx = 0
    for i, question in enumerate(queries[:]):
        print("-" * 80)
        print(f"Question {i + 1}: {question}\n")
        # context = format_docs(retriever.invoke(question))
        # print(chain.invoke({'context': context, 'question': question}))
        output = chain.invoke(question)
        print(output)

        print(f"Think Process: {output.additional_kwargs.get('reasoning_content', '')}")
        print(f"Content: {output.content}\n")
        # result = extract_think_and_json(output)
        # print(result['json_result'])
        print("\n\n")
        answers = json.loads(output.content)['answers']
        for answer in answers:
            total_question_idx += 1
            answer['total_question_idx'] = total_question_idx
        final_results.append({
            'query_idx': i + 1,
            'query': question,
            'think_process': output.additional_kwargs.get('reasoning_content', ''),
            'answers': answers,
        })

        # Save the current result to the result file
        final_result = {
            'document_fn': document_fn,
            'results': final_results,
        }
        with open(result_fn, 'w') as f:  
            json.dump(final_result, f, indent=4)

        with open(markdown_fn, 'w', encoding='utf-8') as f:
            md_output = result_to_markdown.json_to_markdown_both(final_result)
            f.write(md_output)

    print(f"Analysis complete. Results saved to '{result_fn}'.")


def main(args):
    """
    Main function to analyze the document.
    
    Args:
        document_fn (str): The path to the PDF document to analyze.
    """
    queries = load_queries(by_section=True)
    analyze_document(args.document_fn, queries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a PDF document with queries.")
    parser.add_argument('document_fn', type=str, help='Path to the PDF document to analyze.')
    args = parser.parse_args()

    if not args.document_fn.endswith('.pdf'):
        raise ValueError("The document file must be a PDF file with a '.pdf' extension.")

    main(args)