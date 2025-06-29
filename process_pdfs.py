import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

def process_pdf(input_path: str, output_dir: str):
    """处理单个PDF文件"""
    try:
        cmd = f"magic-pdf -p {input_path} -o {output_dir}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"成功处理: {input_path}")
    except subprocess.CalledProcessError as e:
        print(f"处理失败 {input_path}: {e}")

def batch_process_pdfs(input_dir: str = "./data/pdfs", 
                      output_dir: str = "./output",
                      specific_files: List[str] = None):
    """批量处理PDF文件
    
    Args:
        input_dir: PDF文件输入目录
        output_dir: 输出目录
        specific_files: 指定要处理的PDF文件列表，如果为None则处理所有文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if specific_files is not None:
        # 只处理指定的文件
        pdf_files = [
            os.path.join(input_dir, f) 
            for f in specific_files 
            if f.lower().endswith('.pdf') and os.path.exists(os.path.join(input_dir, f))
        ]
    else:
        # 处理目录中的所有PDF文件
        pdf_files = [
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith('.pdf')
        ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        for pdf in pdf_files:
            executor.submit(process_pdf, pdf, output_dir)

if __name__ == "__main__":
    # 从环境变量获取参数
    input_dir = os.environ.get("INPUT_DIR", "./data/pdfs")
    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    specific_files = os.environ.get("SPECIFIC_FILES")
    
    if specific_files:
        # 如果环境变量中指定了文件，将其转换为列表
        specific_files = specific_files.split(",")
    
    print("开始批量处理PDF文件...")
    batch_process_pdfs(input_dir, output_dir, specific_files)
    print("所有PDF处理完成")
