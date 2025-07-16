import json
import textwrap

def json_to_markdown_list(json_data):
    md_output = f"# {json_data['document_fn']}\n\n"
    
    for query_block in json_data['results']:
        query_idx = query_block["query_idx"]
        md_output += f"## Query {query_idx}\n\n"
        
        for answer in query_block["answers"]:
            # 清理问题文本中的多余空格
            clean_question = " ".join(answer["question"].split())
            clean_explanation = " ".join(answer["explanation"].split())

            # 清理quote
            clean_quote = " ".join(answer.get("quote", "").split()) if "quote" in answer else ""
            
            md_output += (
                f"**{answer['total_question_idx']}. {clean_question}**\n\n"
                f"- **Answer**: {'✅ Yes' if answer['answer'] == 1 else '❌ No'}\n"
                f"- **Page Number**: {answer['page_number']}\n"
                f"- **Quote**: {clean_quote}\n"
                f"- **Explanation**: {clean_explanation}\n\n"
            )
    
    return md_output

def json_to_markdown_with_think(json_data):
    md_output = f"# {json_data['document_fn']}\n\n"

    for query_block in json_data['results']:
        query_idx = query_block["query_idx"]
        md_output += f"## Query {query_idx}\n\n"
        
        # 添加思考过程部分
        think_process = query_block.get("think_process", "")
        if think_process:
            md_output += "### Think Process\n\n"
            md_output += f"{think_process}\n\n"
        
        # 添加问题和答案
        md_output += "### Questions & Answers\n\n"
        for answer in query_block["answers"]:
            # 清理问题文本
            clean_question = " ".join(answer["question"].split())
            
            # 清理解释文本
            clean_explanation = " ".join(answer["explanation"].split())

            # 清理quote
            clean_quote = " ".join(answer.get("quote", "").split()) if "quote" in answer else ""
            
            md_output += (
                f"**{answer['total_question_idx']}. {clean_question}**\n\n"
                f"- **Answer**: {'✅ Yes' if answer['answer'] == 1 else '❌ No'}\n"
                f"- **Page Number**: {answer['page_number']}\n"
                f"- **Quote**: {clean_quote}\n"
                f"- **Explanation**: {clean_explanation}\n\n"
            )
    
    return md_output

def json_to_markdown_both(json_data):
    """
    生成Markdown格式的输出，包含思考过程和问题答案。
    
    Args:
        json_data (list): 包含查询和答案的JSON数据。
    
    Returns:
        str: Markdown格式的字符串。
    """
    md_output = json_to_markdown_list(json_data)
    md_output += "\n\n"
    md_output += json_to_markdown_with_think(json_data)
    return md_output

# 示例使用
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JSON results to Markdown format.")
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing the results.')
    args = parser.parse_args()
    # 从文件加载JSON数据 (替换为实际文件路径)
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # 生成Markdown
    markdown_output = json_to_markdown_both(data)
    
    # 打印结果或保存到文件
    # print(markdown_output)
    from utils import replace_suffix
    output_fn = replace_suffix(args.json_file, '.json', '.md')
    with open(output_fn, 'w', encoding="utf-8") as md_file:
        md_file.write(markdown_output)