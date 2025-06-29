import json
import os
import pandas as pd
from openai import OpenAI
from pathlib import Path


def read_markdown_file(file_path: str) -> str:
    """
    读取 Markdown 文件内容。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"文件 {file_path} 编码错误，请确保是 UTF-8 编码")


def call_deepseek_api(message: str, api_key: str = "", base_url: str = "https://api.siliconflow.cn/v1") -> str:
    """
    调用 DeepSeek API 获取信息分析结果。
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[{"role": "user", "content": message}],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"DeepSeek API 调用失败: {str(e)}")


def extract_json_from_response(response: str) -> dict:
    """
    从 API 响应中提取 JSON 数据。
    """
    try:
        json_str = response.split("```json")[1].split("```")[0].strip()
        return json.loads(json_str)
    except IndexError:
        raise IndexError("无法从响应中提取 JSON 数据")
    except json.JSONDecodeError:
        raise json.JSONDecodeError("JSON 格式无效")


def json_to_xlsx(json_data: dict, xlsx_file_path: str) -> None:
    """
    将 JSON 数据转换为 XLSX 文件。
    """
    try:
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            df = pd.DataFrame([json_data])
        else:
            raise ValueError("JSON 格式不支持：必须是列表或字典")
        df.to_excel(xlsx_file_path, index=False, engine='openpyxl')
        print(f"成功转换为 {xlsx_file_path}")
    except ValueError as ve:
        raise ValueError(str(ve))
    except Exception as e:
        raise Exception(f"保存 XLSX 文件失败: {str(e)}")



def batch_process_markdown_files_combined(input_dir: str, output_file: str) -> None:
    """
    批量处理 Markdown 文件，将所有结果合并到一个 XLSX 文件。

    Args:
        input_dir (str): 输入目录，包含 Markdown 文件
        output_file (str): 输出 XLSX 文件路径
    """
    all_results = []

    # 遍历目录中的 .md 文件
    for file_path in Path(input_dir).glob("*.md"):
        try:
            content = read_markdown_file(str(file_path))
            prompt = f"""{content}\n你现在是信息分析大师，请根据上述文档给出关键信息，要求仅包含以下六项内容：
            '赛项名称', '赛道', '发布时间', '报名时间', '组织单位', '官网'。
            例子：’第八届全国青少年人工智能创新挑战赛人工智能综合创新专项赛‘的赛项名称是’第八届全国青少年人工智能创新挑战赛‘ ，赛道是'人工智能综合创新专项赛'。
            发布时间：查找文档中提到的发布日期，通常以“xxxx年xx月”或类似格式出现，一般在文档的头部可以查得。。
            报名时间：寻找报名相关的日期范围，通常以“xxxx年xx月xx日-xxxx年xx月xx日”格式出现，可能在“报名”或“选拔赛”部分。
            组织单位：查找主办单位或组织机构的名称，可能在“主办单位”或文档开头。
            官网：查找官方网址，通常以“http”开头，可能在“组委会”或“信息发布平台”部分, 只需返回一个即可。
            以上六项内容以json格式返回，如果未找到对应的信息，请如实表明未查询到。"""
            print(f"处理文件: {file_path}")
            response = call_deepseek_api(prompt)
            json_data = extract_json_from_response(response)
            # 添加文件名作为标识
            # json_data["source_file"] = file_path.name
            all_results.append(json_data)
            print(f"DeepSeek JSON: {json_data}")
        except Exception as e:
            print(f"处理文件 {file_path} 失败: {str(e)}")

    # 合并结果并保存
    if all_results:
        try:
            df = pd.DataFrame(all_results)
            df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"成功合并到 {output_file}")
        except Exception as e:
            print(f"保存合并文件失败: {str(e)}")
    else:
        print("没有成功处理任何文件")


def main():
    """
    程序入口。
    """
    input_dir = r''
    combined_output_file = r'./knowledge/result_1.xlsx'

    # 合并输出
    print("\n开始合并输出处理...")
    batch_process_markdown_files_combined(input_dir, combined_output_file)


if __name__ == '__main__':
    main()