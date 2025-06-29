import json
import os
import shutil
import requests
from huggingface_hub import snapshot_download

def download_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.2.0':
            data = download_json(url)
    else:
        data = download_json(url)

    for key, value in modifications.items():
        data[key] = value

    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 1. 指定自定义下载目录
    CUSTOM_MODELS_DIR = "/root/autodl-tmp/custom_models"  # ← 修改为您想要的路径
    os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)

    # 2. 定义下载模式（可修改为您的实际需求）
    mineru_patterns = [
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*", 
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
    ]

    # 3. 下载主模型（指定自定义目录）
    model_dir = snapshot_download(
        repo_id='opendatalab/PDF-Extract-Kit-1.0',
        allow_patterns=mineru_patterns,
        cache_dir=CUSTOM_MODELS_DIR,  # 关键参数：指定下载目录
        local_files_only=False,       # 允许下载
        resume_download=True
    )

    # 4. 下载LayoutReader模型
    layoutreader_pattern = ["*.json", "*.safetensors"]
    layoutreader_model_dir = snapshot_download(
        repo_id='hantian/layoutreader',
        allow_patterns=layoutreader_pattern,
        cache_dir=CUSTOM_MODELS_DIR,  # 使用相同自定义目录
        local_files_only=False
    )

    # 5. 处理路径（兼容后续使用）
    model_dir = os.path.join(model_dir, 'models')
    print(f'[INFO] 主模型目录: {model_dir}')
    print(f'[INFO] LayoutReader目录: {layoutreader_model_dir}')

    # 6. 生成配置文件
    json_url = 'https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json'
    config_file = os.path.join(os.path.expanduser('~'), 'magic-pdf.json')
    
    download_and_modify_json(
        json_url,
        config_file,
        modifications={
            'models-dir': model_dir,
            'layoutreader-model-dir': layoutreader_model_dir
        }
    )
    print(f'[SUCCESS] 配置文件已生成: {config_file}')