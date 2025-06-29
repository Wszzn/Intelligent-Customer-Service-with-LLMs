import json
import os

def remove_embeddings_from_json(input_path: str, output_path: str = None):
    """
    从JSON文件中删除embedding字段
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径，如果不指定则覆盖原文件
    """
    # 如果没有指定输出路径，则覆盖原文件
    if output_path is None:
        output_path = input_path
        
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是列表类型
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'embedding' in item:
                    del item['embedding']
        # 如果是字典类型
        elif isinstance(data, dict):
            if 'embedding' in data:
                del data['embedding']
        
        # 保存处理后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"成功处理文件：{input_path}")
        print(f"已保存到：{output_path}")
        
    except json.JSONDecodeError:
        print(f"错误：{input_path} 不是有效的JSON文件")
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")

if __name__ == "__main__":
    # 处理 documents.json 文件
    json_path = "/root/ty/data/embeddings/documents.json"
    remove_embeddings_from_json(json_path)
