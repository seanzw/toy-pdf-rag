from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_deepseek import ChatDeepSeek

import pandas as pd
import json

import result_to_markdown
import argparse


def load_queries(by_section=False, queries_fn='queries.txt'):
    queries = list()
    query_separator = "\n" if not by_section else "### "
    with open(queries_fn, 'r') as f:
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
    # for i, q in enumerate(queries):
    #     print(f"\nQuery {i + 1}:-------------------------------------------\n{q}")
    return queries

def format_docs(docs, filename):
    # for i, doc in enumerate(docs):
    #     # Print the document content and metadata
    #     print(f"Document {i + 1} --------- \n meta {doc.metadata} \n---------content:\n{doc.page_content}")

    # Format the documents with page labels and content
    # print(docs[0].metadata)
    # assert(False)
    ret = "\n\n".join(f'Document: {doc.metadata["source"]}\nPage: {doc.metadata["page"] + 1}\n{doc.page_content}' for doc in docs)
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n**********************************************************************\n")
        for idx, doc in enumerate(docs):
            f.write(f'{idx:2} Document: {doc.metadata["source"]}\nPage: {doc.metadata["page"] + 1}\n\n')
        for idx, doc in enumerate(docs):
            f.write(f'{idx:2} Document: {doc.metadata["source"]}\nPage: {doc.metadata["page"] + 1}\n{doc.page_content}\n\n')
    print(f"Context saved to {filename}")
    # assert(False)
    return ret

def save_final_prompt(prompt, filename):
    """
    Saves the final prompt to a text file.
    
    Args:
        prompt (PromptTemplate): The prompt template to save
        filename (str): The name of the file to save the prompt to
    """
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n**********************************************************************\n")
        f.write(prompt.to_string())
    print(f"Prompt saved to {filename}")
    return prompt

def intialize_chain(document_fn, prompt_fn, extra_fn=[]):

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
    all_document_fn = [document_fn] + extra_fn
    all_retrievers = []
    all_keyword_retrievers = []
    for fn in all_document_fn:
        loader = PyPDFLoader(fn)
        pages = loader.load_and_split()
        print(f"Loaded {len(pages)} pages from {fn}")

        # Create an embedding retriever
        store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        num_pages = 5  # Number of pages to retrieve
        retriever = store.as_retriever(search_kwargs={'k': num_pages})
        all_retrievers.append(retriever)

        # Create a BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            pages,
            disable_rand=True,
            sort_by_similarity=True,
            random_state=42)
        bm25_retriever.k = num_pages + 10  # Set the number of pages to retrieve
        all_keyword_retrievers.append(bm25_retriever)
    
    # loader = PyPDFLoader(document_fn)
    # pages = loader.load_and_split()
    # store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    # num_pages = 5
    # retriever = store.as_retriever(search_kwargs={'k': num_pages})

    # 3. Create the prompt template
    template = """
<context>
{context}
</context>
<questions>:
{question}
</questions>

Given the context provided, answer each question using the guidance.
Respond with a json object containing a list of answer,
a brief explanation with page number and quote from the document, and the document name.
The answer should be binary (1 for yes, 0 for no).

Example response:
{{
    "answers": [
        {{
            
                "question": "Does the report provide quantitative data on Scope 1 GHG emissions?",
                "answer": 1,
                "page_number": 61,
                "quote": "Our Scope 1 GHG emissions for FY2022 were 1000 tonnes CO2e.",
                "doucument": "2023-Annual-Report.pdf",
                "explanation": "Yes. The report provides FY2022 and FY2023 data for Scope 1 GHG emissions (Page 61)."
        }},
    ]
}}
"""

    prompt = PromptTemplate.from_template(template)

    def retrieve_and_format_docs(question):
        """
        Retrieve documents based on the question and format them.
        
        Args:
            question (str): The question to retrieve documents for.
        
        Returns:
            str: Formatted documents as a string.
        """
        # Retrieve documents from all retrievers
        retrieved_docs = []
        for retriever in all_retrievers:
            docs = retriever.invoke(question)
            retrieved_docs.extend(docs)

        # collect keyword from question
        keywords = set()
        for line in question.split('\n'):
            if line.startswith('Keywords:'):
                # Extract keywords from the line
                keywords.update(word.strip().lower() for word in line.split(':')[1].split(',') if word.strip())
        print(f"Keywords: {keywords}")
        for retriever in all_keyword_retrievers:
            # Use the keywords to retrieve documents
            keyword_docs = retriever.invoke(','.join(keywords))
            for keyword_doc in keyword_docs:
                # Check if the document is already in retrieved_docs
                already_exists = False
                for doc in retrieved_docs:
                    if doc.metadata['source'] == keyword_doc.metadata['source'] and doc.metadata['page'] == keyword_doc.metadata['page']:
                        # If the document is already in retrieved_docs, skip it
                        already_exists = True
                        break
                if not already_exists:
                    print(f"Adding keyword document: {keyword_doc.metadata['source']} page {keyword_doc.metadata['page'] + 1}")
                    # print(keyword_doc.page_content)
                    retrieved_docs.append(keyword_doc)
        
        # Format the retrieved documents
        # assert(False)
        return format_docs(retrieved_docs, prompt_fn)
    
    # 4. Build the chain of operations
    chain = (
        {
            "context": RunnableLambda(retrieve_and_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        # | RunnableLambda(lambda x: save_final_prompt(x, prompt_fn))
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
    

def analyze_document(document_fn, queries, extra_fn=[], extra_header=[]):

    prompt_fn, result_fn, markdown_fn = get_output_fn(document_fn)
    chain = intialize_chain(document_fn, prompt_fn, extra_fn=extra_fn)
    
    print("****************************************************")
    print(f'Starting the analyze {document_fn}\n')
    final_results = []
    total_question_idx = 0
    for i, question in enumerate(queries[:]):
        # print("-" * 80)
        # print(f"Question {i + 1}: {question}\n")
        # context = format_docs(retriever.invoke(question))
        # print(chain.invoke({'context': context, 'question': question}))
        output = chain.invoke(question)
        # print(output)

        # print(f"Think Process: {output.additional_kwargs.get('reasoning_content', '')}")
        # print(f"Content: {output.content}\n")
        # result = extract_think_and_json(output)
        # print(result['json_result'])
        # print("\n\n")
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
    
    answers = [answer for item in final_results for answer in item['answers']]
    # Save answers to a xlsx file with extra header
    xlsx_fn = result_fn.replace('.json', '.xlsx')
    with pd.ExcelWriter(xlsx_fn, engine='openpyxl') as writer:
        if extra_header:
            df_header = pd.DataFrame(extra_header)
            df_header.to_excel(writer, startrow=0, startcol=0, header=False, index=False)
        # Write table starting at row 2 (leaving row 1 for header)
        df = pd.DataFrame(answers)
        df.to_excel(writer, startrow=2, index=False)
    
    return result_fn



def main(args):
    """
    Main function to analyze the document.
    
    Args:
        document_fn (str): The path to the PDF document to analyze.
    """
    queries = load_queries(by_section=True)
    analyze_document(args.document_fn, queries, extra_fn=args.extra_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a PDF document with queries.")
    parser.add_argument('document_fn', type=str, help='Path to the PDF document to analyze.')
    parser.add_argument('--extra_fn', type=str, nargs='*', default=[], help='Additional PDF files to analyze.')
    parser.add_argument('--queries_fn', type=str, default='queries.txt', help='Path to the queries file. Default is "queries.txt".')
    args = parser.parse_args()

    if not args.document_fn.endswith('.pdf'):
        raise ValueError("The document file must be a PDF file with a '.pdf' extension.")

    main(args)