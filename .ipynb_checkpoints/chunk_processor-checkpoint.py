import re
from typing import List, Dict
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

# 初始化模型和处理器（单例模式）
_model = None
_processor = None

def init_vl_model():
    """初始化视觉语言模型"""
    global _model, _processor
    if _model is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/root/autodl-tmp/modelscope/models/qwen_VL_7B", 
            torch_dtype="auto", 
            device_map="auto"
        )
    if _processor is None:
        _processor = AutoProcessor.from_pretrained("/root/autodl-tmp/modelscope/models/qwen_VL_7B")

def extract_image_content(image_path: str, base_dir: str = "") -> str:
    """
    从图片路径提取图片内容(图生文)
    
    Args:
        image_path: 图片路径(可能是相对或绝对路径)
        base_dir: 基准目录，用于解析相对路径
        
    Returns:
        图片内容的文本描述
    """
    # 初始化模型
    init_vl_model()
    
    # 处理路径
    full_path = os.path.join(base_dir, image_path) if base_dir else image_path
    if not os.path.exists(full_path):
        return f"[图片不存在: {full_path}]"
    
    # 构建输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": full_path},
                {"type": "text", "text": "详细描述这张图片的内容。"}
            ],
        }
    ]
    
    try:
        # 准备输入
        text = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = _processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        # 生成描述
        generated_ids = _model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = _processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        print(output_text[0])
        return f"图片描述：[{output_text[0]}]" if output_text else f"[无法解析图片: {full_path}]"
    
    except Exception as e:
        print(f"图片处理错误: {e}")
        return f"[图片处理失败: {full_path}]"

def clean_text(text: str, remove_images: bool = False, base_dir: str = "") -> str:
    """增强版文本清洗"""
    # 1. 基础清理
    text = re.sub(r'[\u3000\xa0\s]+', '', text)  # 保留常规空格
    text = re.sub(r'<[^>]+>', '', text)  # HTML标签
    
    # 2. 处理Markdown元素
    if remove_images:
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    else:
        # 提取图片路径并替换为图片内容
        def replace_image(match):
            alt_text = match.group(1)
            image_path = match.group(2)
            return extract_image_content(image_path, base_dir)
        text = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, text)
    
    # 3. 处理表格和代码块
    # text = re.sub(r'```[\s\S]*?```', '', text)  # 代码块
    # text = re.sub(r'`[^`]+`', '', text)  # 行内代码
    # text = re.sub(r'\|.*?\|\n?', '', text)  # 简单表格
    
    # 4. 处理URL和特殊字符
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[△※★☆▪■▶◀●]+', '', text)  # 更多特殊符号
    
    # 5. 统一标点符号
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r"['']", "'", text)
    text = re.sub(r'—', '-', text)
    
    # 6. 规范化空白字符
    text = re.sub(r'\n{3,}', '\n\n', text)  # 多个空行合并为两个
    text = re.sub(r'[ \t]+', '', text)  # 多个空格合并
    
    return text.strip()

def split_by_title(text: str, base_dir: str = "", max_chunk_size: int = 1024) -> List[Dict]:
    """
    按Markdown标题分割文本，将标题作为元数据保存
    Args:
        text: 输入的Markdown文本
        base_dir: 基础目录路径
        max_chunk_size: 最大chunk大小
    Returns:
        List[Dict]: 包含文本块和元数据的列表
    """
    # 清理多余的空行
    text = re.sub(r'\n{2,}', '\n', text)
    
    # 初始化结果列表
    chunks = []
    current_chunk = []
    current_headers = {}  # 用于存储当前标题层级
    
    for line in text.split('\n'):
        # 检测标题行
        title_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if title_match:
            # 如果当前有正在收集的chunk，先保存
            if current_chunk:
                chunk_text = clean_text('\n'.join(current_chunk), base_dir=base_dir)
                if chunk_text:  # 确保清理后的文本不为空
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {"headers": current_headers.copy()}
                    })
            
            # 开始新的chunk
            current_chunk = []
            level = len(title_match.group(1))
            title = title_match.group(2)
            
            # 更新标题层级
            if level == 1:
                current_headers = {"Header 1": title}
            elif level == 2:
                if "Header 1" not in current_headers:
                    current_headers["Header 1"] = "未命名章节"  # 确保有父级标题
                current_headers["Header 2"] = title
            # 可以继续添加更多层级的标题
        else:
            current_chunk.append(line)
    
    # 处理最后一个chunk
    if current_chunk:
        chunk_text = clean_text('\n'.join(current_chunk), base_dir=base_dir)
        if chunk_text:  # 确保清理后的文本不为空
            chunks.append({
                "text": chunk_text,
                "metadata": {"headers": current_headers.copy()}
            })
    
    # 处理过大的chunk
    final_chunks = []
    for chunk in chunks:
        print(chunk)
        if len(chunk["text"]) > max_chunk_size:
            # 对过大的chunk进行分割，但保留元数据
            sub_chunks = chunk_text(chunk["text"], max_chunk_size)
            for sub_chunk in sub_chunks:
                final_chunks.append({
                    "text": sub_chunk,
                    "metadata": chunk["metadata"]
                })
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 200) -> List[str]:
    """
    将文本分割成指定大小的chunk，每个chunk之间有固定重叠
    
    Args:
        text: 输入文本
        chunk_size: 每个chunk的长度
        overlap: chunk之间的重叠长度
        
    Returns:
        分割后的chunk列表
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # 下一个chunk从当前chunk末尾减去重叠部分开始
    
    return chunks