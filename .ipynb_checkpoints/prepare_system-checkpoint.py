import os
import json
from typing import List, Dict
from chunk_processor import split_by_title
from kg_builder import KGBuilder
from embedding_handler import EmbeddingHandler
import re
from LLM_tool import LLMHandler
from LLMhandler import AsyncLLMHandler
import asyncio
import faiss
import numpy as np
import datetime
import hashlib

class SystemPreparer:
    def __init__(self):
        
        self.embedding_handler = EmbeddingHandler()
        self.kg_builder = KGBuilder(embeddingHandler=self.embedding_handler)
        # self.llm_handler = LLMHandler()
        self.llm_handler = AsyncLLMHandler()
        self.data_dir = "/root/ty/data"
        self.pdf_dir = "/root/ty/data/pdfs"
        self.update_log_path = os.path.join(self.data_dir, "update_log.json")
    def process_all_pdfs(self):
        """处理所有PDF文件"""
        os.system("python /root/ty/process_pdfs.py")
    
    def load_and_chunk_markdowns(self) -> List[Dict]:
        """加载并分块所有Markdown文件"""
        chunks = []
        
        for root, _, files in os.walk("/root/ty/output"):
            # 跳过.ipynb_checkpoints目录
            if ".ipynb_checkpoints" in root:
                continue
            
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)
                    print(f"正在处理 {path}")
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    # 从文件名提取比赛名
                    match = re.search(r'\d+_(.*?)\.md', file)
                    competition_name = match.group(1) if match else file
                    
                    # 分块并添加到新chunks
                    for chunk_data in split_by_title(text, root):
                        chunk_metadata = chunk_data["metadata"]["headers"].copy()
                        chunk_metadata["source"] = competition_name
                        chunks.append({
                            "chunk": chunk_data["text"],
                            "metadata": chunk_metadata
                        })
        
        # 保存分块结果
        # with open("/root/ty/data/chunks.json", "w", encoding="utf-8") as f:
        #     json.dump(chunks, f, ensure_ascii=False)
            
        return chunks
    
    async def build_knowledge_graph(self, chunks: List[Dict]):
        """构建知识图谱并保存"""
        # 提取文本构建KG
        texts = [c["chunk"] for c in chunks]
        metadata = [c["metadata"] for c in chunks]
        kg_data = []

        # with open("/root/ty/data/kg_data.json", "r", encoding="utf-8") as f:
        #     kg_data = json.load(f)
        # print("检测到已有kg_data.json文件，直接加载...")
        
            # for i,text in enumerate(texts):  # 改为逐个处理
            #     extraction_result = self.llm_handler.get_second_extraction_result(text)
            #     extraction_result["chunkid"] = i
            #     kg_data.append(extraction_result)
            

            # 直接定义并await协程函数
        async def process_text(i, text, meta):
            extraction_result = await self.llm_handler.get_second_extraction_result(text)
            extraction_result["chunkid"] = i
            extraction_result["metadata"] = meta
            return extraction_result

        # 创建任务列表并gather，传入metadata
        tasks = [process_text(i, text, meta) for i, (text, meta) in enumerate(zip(texts, metadata))]
        kg_data = await asyncio.gather(*tasks)
        
        # 保存KG数据
        with open("/root/ty/data/kg_data.json", "w", encoding="utf-8") as f:
            json.dump(kg_data, f, ensure_ascii=False)

        # 构建向量数据库
        self.kg_builder.store_to_neo4j(kg_data)
        self.kg_builder.save_vector_indexes("/root/ty/data/vector_indexes")
        # 保存chunk向量
        chunk_texts = []
        for i in range(len(texts)):
            chunk_texts.append(f"这是关于{metadata[i]}的相关信息:{texts[i]}")
        chunk_embeddings = self.embedding_handler.embed_texts(texts)
        embeddings_data = []
        for i, (chunk, embedding, meta) in enumerate(zip(chunks, chunk_embeddings, metadata)):
            embeddings_data.append({
                "id": i,
                "chunk": chunk["chunk"],
                # "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "keywords": kg_data[i]["content_keywords"],
                "metadata": meta
            })
        
        # 保存到JSON文件
        with open("/root/ty/data/chunk_embeddings.json", "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
        with open("/root/ty/data/embeddings/documents.json", "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
        # 新增：保存文本和FAISS向量到/root/ty/data/embeddings
        os.makedirs("/root/ty/data/embeddings", exist_ok=True)
        
        # 创建并保存FAISS索引
        embeddings_array = np.array([e["embedding"] for e in embeddings_data]).astype('float32')
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        index.add(embeddings_array)
        faiss.write_index(index, "/root/ty/data/embeddings/faiss.index")
        
        # # 保存文本内容
        # with open("/root/ty/data/embeddings/documents.json", "w", encoding="utf-8") as f:
        #     json.dump([{"id": e["id"], "text": e["chunk"]} for e in embeddings_data], 
        #              f, ensure_ascii=False, indent=2)
        
        # 保存向量索引
        
    
    def _create_initial_update_log(self):
        """创建初始的更新日志文件"""
        initial_log = {
            "last_update": datetime.datetime.now().isoformat(),
            "file_hashes": {},
            "pdf_hashes": {},
            "processed_chunks": []
        }

        # 记录所有PDF文件的哈希值
        for file in os.listdir(self.pdf_dir):
            if file.endswith('.pdf'):
                file_path = os.path.join(self.pdf_dir, file)
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                initial_log["pdf_hashes"][file] = file_hash

        # 保存更新日志
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.update_log_path, 'w', encoding='utf-8') as f:
            json.dump(initial_log, f, ensure_ascii=False, indent=2)

    async def run_full_preparation(self):
        """执行完整预处理流程"""
        print("1. 处理PDF文件...")
        # self.process_all_pdfs()
        
        print("2. 分块Markdown内容...")
        chunks = self.load_and_chunk_markdowns()
        
        print("3. 构建知识图谱和向量数据库...")
        await self.build_knowledge_graph(chunks)

        print("4. 创建初始更新日志...")
        self._create_initial_update_log()
        
        print("预处理完成！所有中间结果已保存到/root/ty/data目录")

async def main():
    # 创建数据目录
    os.makedirs("/root/ty/data", exist_ok=True)

    preparer = SystemPreparer()
    await preparer.run_full_preparation()

if __name__ == "__main__":
#     # 创建数据目录
#     os.makedirs("/root/ty/data", exist_ok=True)
    
#     preparer = SystemPreparer()
#     preparer.run_full_preparation()
    asyncio.run(main())