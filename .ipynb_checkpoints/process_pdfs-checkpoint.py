import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def process_pdf(input_path, output_dir):
    """处理单个PDF文件"""
    try:
        cmd = f"magic-pdf -p {input_path} -o {output_dir}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"成功处理: {input_path}")
    except subprocess.CalledProcessError as e:
        print(f"处理失败 {input_path}: {e}")

def batch_process_pdfs(input_dir="./data", output_dir="./output"):
    """批量处理PDF文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith('.pdf')
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        for pdf in pdf_files:
            executor.submit(process_pdf, pdf, output_dir)

if __name__ == "__main__":
    print("开始批量处理PDF文件...")
    batch_process_pdfs()
    print("所有PDF处理完成")