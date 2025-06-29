import json
import os
import requests
from modelscope import snapshot_download

def download_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        print("true")
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.1.1':
            data = download_json(url)
    else:
        data = download_json(url)

    for key, value in modifications.items():
        data[key] = value

    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 指定自定义下载目录
    custom_model_dir = "/root/autodl-tmp/models"
    os.makedirs(custom_model_dir, exist_ok=True)  # 确保目录存在

    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small_2501/*",
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    
    # 使用cache_dir指定下载目录
    model_dir = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', 
                                allow_patterns=mineru_patterns,
                                cache_dir=custom_model_dir)
    layoutreader_model_dir = snapshot_download('ppaanngggg/layoutreader',
                                            cache_dir=custom_model_dir)
    
    model_dir = os.path.join(model_dir, 'models')
    print(f'model_dir is: {model_dir}')
    print(f'layoutreader_model_dir is: {layoutreader_model_dir}')

    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/magic-pdf.template.json'
    config_file_name = 'magic-pdf.json'
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f'The configuration file has been configured successfully, the path is: {config_file}')
