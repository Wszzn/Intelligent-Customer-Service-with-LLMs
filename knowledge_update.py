from prepare_system import SystemPreparer
from typing import List, Dict, Tuple
import os
import json
import asyncio
from datetime import datetime
import hashlib
import faiss
import numpy as np
import shutil
import uuid

class KnowledgeBaseUpdater:
    def __init__(self):
        """初始化更新器"""
        self.preparer = SystemPreparer()
        self.data_dir = "/root/ty/data"
        self.pdf_dir = "/root/ty/data/pdfs"  # PDF源文件目录
        self.output_dir = "/root/ty/output"  # markdown输出目录
        self.update_dir = "/root/ty/update"  # 更新文件临时目录
        self.update_log_path = os.path.join(self.data_dir, "update_log.json")
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _process_pdfs(self, pdf_files: List[str]) -> List[str]:
        """处理PDF文件并返回生成的markdown文件路径"""
        # 确保更新目录存在
        os.makedirs(self.update_dir, exist_ok=True)
        
        # 复制需要处理的PDF到临时目录
        temp_pdf_dir = os.path.join(self.update_dir, "pdfs")
        os.makedirs(temp_pdf_dir, exist_ok=True)
        
        for pdf_file in pdf_files:
            shutil.copy2(
                os.path.join(self.pdf_dir, pdf_file),
                os.path.join(temp_pdf_dir, pdf_file)
            )
        
        # 修改process_pdfs.py中的输出目录和指定要处理的文件
        import subprocess
        env = os.environ.copy()
        env["OUTPUT_DIR"] = self.update_dir
        env["INPUT_DIR"] = temp_pdf_dir
        env["SPECIFIC_FILES"] = ",".join(pdf_files)  # 将文件列表转换为逗号分隔的字符串
        subprocess.run(["python", "/root/ty/process_pdfs.py"], env=env)
        
        # 返回生成的markdown文件路径
        markdown_files = []
        for root, _, files in os.walk(self.update_dir):
            for file in files:
                if file.endswith(".md"):
                    markdown_files.append(os.path.join(root, file))
        
        return markdown_files
            
    def _load_update_log(self) -> Dict:
        """加载更新日志"""
        if os.path.exists(self.update_log_path):
            with open(self.update_log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                # 确保包含PDF文件哈希
                if "pdf_hashes" not in log_data:
                    log_data["pdf_hashes"] = {}
                # 确保processed_chunks是列表
                if "processed_chunks" not in log_data:
                    log_data["processed_chunks"] = []
                return log_data
        return {
            "last_update": "",
            "file_hashes": {},
            "pdf_hashes": {},
            "processed_chunks": []
        }

    def _check_pdf_changes(self) -> Tuple[List[str], bool, List[str]]:
        """检查PDF文件变化
        Returns:
            Tuple[List[str], bool, List[str]]: 
            - 变化的PDF文件列表
            - 是否有文件删除
            - 内容更新的PDF文件列表
        """
        update_log = self._load_update_log()
        changed_pdfs = []      # 新增的文件
        updated_pdfs = []      # 内容更新的文件
        has_deleted = False
        
        # 检查新增和修改的文件
        for file in os.listdir(self.pdf_dir):
            if file.endswith('.pdf'):
                file_path = os.path.join(self.pdf_dir, file)
                current_hash = self._calculate_file_hash(file_path)
                
                if file not in update_log["pdf_hashes"]:
                    # 新文件
                    changed_pdfs.append(file)
                    update_log["pdf_hashes"][file] = current_hash
                elif update_log["pdf_hashes"][file] != current_hash:
                    # 文件内容已更新
                    updated_pdfs.append(file)
                    update_log["pdf_hashes"][file] = current_hash
        
        # 检查删除的文件
        for file in list(update_log["pdf_hashes"].keys()):
            if not os.path.exists(os.path.join(self.pdf_dir, file)):
                del update_log["pdf_hashes"][file]
                has_deleted = True
        
        # 保存更新的哈希记录
        self._save_update_log(update_log)
        
        return changed_pdfs, has_deleted, updated_pdfs

    def _generate_stable_chunk_id(self, chunk_text, source):
        """
        为文本块生成一个稳定的ID
        使用文本内容和源文件的组合哈希值，确保相同内容的块总是有相同的ID
        """
        # 使用文本内容和源文件名的组合生成哈希
        content_hash = hashlib.md5((chunk_text + str(source)).encode('utf-8')).hexdigest()
        return content_hash

    async def update_knowledge_base(self):
        """执行知识库更新"""
        print("开始更新知识库...")
        
        # 检查PDF文件变化
        changed_pdfs, has_deleted, updated_pdfs = self._check_pdf_changes()
        
        if not changed_pdfs and not has_deleted and not updated_pdfs:
            print("没有检测到PDF文件变化，无需更新")
            return
        
        print(f"检测到{len(changed_pdfs)}个新增PDF文件")
        print(f"检测到{len(updated_pdfs)}个内容更新的PDF文件")
        
        # 如果有文件被删除，需要处理
        if has_deleted:
            print("检测到文件被删除，将精确移除相关数据")
            deleted_pdfs = self._identify_deleted_pdfs()
            if deleted_pdfs:
                await self._remove_data_for_pdfs(deleted_pdfs)
        
        # 处理新增和更新的PDF文件
        files_to_process = changed_pdfs + updated_pdfs
        
        # 如果有内容更新的文件，需要删除旧的知识图谱数据
        if updated_pdfs:
            await self._remove_data_for_pdfs(updated_pdfs)
        
        # 如果没有文件需要处理，直接返回
        if not files_to_process:
            print("没有新的文件需要处理")
            return
        
        markdown_files = self._process_pdfs(files_to_process)
        
        # 处理markdown文件
        new_chunks = []
        chunk_id_map = {}  # 映射文本到生成的chunk_id
        
        for file_path in markdown_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # 从文件名提取比赛名
            import re
            match = re.search(r'\d+_(.*?)\.md', os.path.basename(file_path))
            competition_name = match.group(1) if match else os.path.basename(file_path)
            
            # 分块并添加到新chunks
            from chunk_processor import split_by_title
            for chunk in split_by_title(text):
                # 为每个chunk生成稳定ID
                chunk_text = chunk["chunk"]
                stable_id = self._generate_stable_chunk_id(chunk_text, competition_name)
                
                # 记录ID映射，稍后用于KG数据
                chunk_id_map[chunk_text] = stable_id
                
                new_chunks.append({
                    "chunk": chunk_text,
                    "id": stable_id,
                    "metadata": {"source": competition_name}
                })

        if not new_chunks:
            print("没有新的内容需要处理")
            return

        print(f"开始处理{len(new_chunks)}个新的文本块...")

        # 处理新的文本块并更新知识库
        texts = [c["chunk"] for c in new_chunks]
        metadata = [c["metadata"] for c in new_chunks]
        chunk_ids = [c["id"] for c in new_chunks]

        # 处理新的文本块
        async def process_text(i, text, chunk_id):
            extraction_result = await self.preparer.llm_handler.get_second_extraction_result(text)
            extraction_result["chunkid"] = chunk_id  # 使用稳定ID
            return extraction_result

        tasks = [process_text(i, text, chunk_ids[i]) for i, text in enumerate(texts)]
        kg_data = await asyncio.gather(*tasks)

        # 更新向量索引
        chunk_texts = []
        for i in range(len(texts)):
            chunk_texts.append(f"这是关于{metadata[i]}的相关信息:{texts[i]}")
        
        chunk_embeddings = self.preparer.embedding_handler.embed_texts(texts)
        
        embeddings_data = []
        for i, (chunk, embedding, meta, chunk_id) in enumerate(zip(new_chunks, chunk_embeddings, metadata, chunk_ids)):
            embeddings_data.append({
                "id": chunk_id,  # 使用稳定ID
                "chunk": chunk["chunk"],
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "keywords": kg_data[i]["content_keywords"],
                "metadata": meta
            })

        # 合并新旧数据
        await self._merge_embeddings(embeddings_data)
        await self._merge_kg_data(kg_data)

        # 更新处理记录
        update_log = self._load_update_log()
        update_log["processed_chunks"] = list(set(update_log["processed_chunks"] + chunk_ids))
        update_log["last_update"] = datetime.now().isoformat()
        self._save_update_log(update_log)

        # 清理临时目录
        if os.path.exists(self.update_dir):
            shutil.rmtree(self.update_dir)

        print("知识库更新完成！")

    def _save_update_log(self, log_data: Dict):
        """保存更新日志"""
        # 将集合转换为列表以便JSON序列化
        log_data["processed_chunks"] = list(log_data["processed_chunks"])
        with open(self.update_log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    async def _merge_embeddings(self, new_embeddings_data: List[Dict]):
        """合并新旧向量索引，使用稳定ID"""
        try:
            # 加载现有的embeddings数据
            existing_embeddings_path = os.path.join(self.data_dir, "embeddings", "documents.json")
            if os.path.exists(existing_embeddings_path):
                with open(existing_embeddings_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    # 只过滤无效数据
                    existing_data = [item for item in existing_data if "embedding" in item]
            else:
                existing_data = []

            # 创建一个现有ID的集合，用于检查重复
            existing_ids = {d["id"] for d in existing_data}
            
            # 合并数据，避免ID重复
            for emb in new_embeddings_data:
                # 如果ID已存在，更新现有数据
                if emb["id"] in existing_ids:
                    # 找到并替换现有数据
                    for i, item in enumerate(existing_data):
                        if item["id"] == emb["id"]:
                            existing_data[i] = emb
                            break
                else:
                    # 否则添加新数据
                    existing_data.append(emb)
                    existing_ids.add(emb["id"])

            # 保存更新后的数据
            os.makedirs(os.path.join(self.data_dir, "embeddings"), exist_ok=True)
            with open(existing_embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            # 更新FAISS索引
            embeddings_array = np.array([e["embedding"] for e in existing_data]).astype('float32')
            index = faiss.IndexFlatIP(embeddings_array.shape[1])
            index.add(embeddings_array)
            faiss.write_index(index, os.path.join(self.data_dir, "embeddings", "faiss.index"))
            
            print(f"成功合并向量索引，共有{len(existing_data)}条数据")
            
        except Exception as e:
            print(f"更新向量索引时发生错误: {str(e)}")
            raise

    async def _merge_kg_data(self, new_kg_data: List[Dict]):
        """合并新旧知识图谱数据，使用稳定ID"""
        # 加载现有的KG数据
        kg_data_path = os.path.join(self.data_dir, "kg_data.json")
        if os.path.exists(kg_data_path):
            with open(kg_data_path, 'r', encoding='utf-8') as f:
                existing_kg_data = json.load(f)
        else:
            existing_kg_data = []

        # 创建一个现有chunkid的映射，用于检查重复和更新
        existing_chunks = {data["chunkid"]: i for i, data in enumerate(existing_kg_data)}
        
        # 合并数据，更新现有数据或添加新数据
        for kg_item in new_kg_data:
            if kg_item["chunkid"] in existing_chunks:
                # 更新现有数据
                existing_kg_data[existing_chunks[kg_item["chunkid"]]] = kg_item
            else:
                # 添加新数据
                existing_kg_data.append(kg_item)

        # 保存更新后的数据
        with open(kg_data_path, 'w', encoding='utf-8') as f:
            json.dump(existing_kg_data, f, ensure_ascii=False, indent=2)

        # 更新Neo4j和向量索引
        self.preparer.kg_builder.store_to_neo4j(existing_kg_data)
        self.preparer.kg_builder.save_vector_indexes(os.path.join(self.data_dir, "vector_indexes"))

    def _identify_deleted_pdfs(self) -> List[str]:
        """识别已删除的PDF文件"""
        update_log = self._load_update_log()
        deleted_pdfs = []
        
        # 检查日志中记录的PDF文件是否还存在
        for file in list(update_log["pdf_hashes"].keys()):
            if not os.path.exists(os.path.join(self.pdf_dir, file)):
                deleted_pdfs.append(file)
        
        return deleted_pdfs

    async def _remove_data_for_pdfs(self, pdf_files: List[str]):
        """删除与指定PDF文件相关的所有数据"""
        # 从日志中识别与这些PDF相关的chunk_ids
        chunk_ids_to_remove = set()
        
        # 将PDF文件名转换为对应的markdown文件名模式
        md_patterns = [pdf.replace('.pdf', '') for pdf in pdf_files]
        
        # 加载向量数据
        embeddings_path = os.path.join(self.data_dir, "embeddings", "documents.json")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # 找出需要删除的chunk_ids
            for item in existing_data:
                for pattern in md_patterns:
                    if pattern in item.get('metadata', {}).get('source', ''):
                        chunk_ids_to_remove.add(item['id'])
        
        if chunk_ids_to_remove:
            # 1. 从向量数据中删除
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # 过滤掉要删除的数据
                existing_data = [data for data in existing_data if data.get('id') not in chunk_ids_to_remove]
                
                # 保存更新后的数据
                with open(embeddings_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
                # 重建FAISS索引
                if existing_data:
                    embeddings_array = np.array([e["embedding"] for e in existing_data]).astype('float32')
                    index = faiss.IndexFlatIP(embeddings_array.shape[1])
                    index.add(embeddings_array)
                    faiss.write_index(index, os.path.join(self.data_dir, "embeddings", "faiss.index"))
            
            # 2. 从知识图谱数据中删除
            kg_data_path = os.path.join(self.data_dir, "kg_data.json")
            if os.path.exists(kg_data_path):
                with open(kg_data_path, 'r', encoding='utf-8') as f:
                    kg_data = json.load(f)
                
                # 过滤掉要删除的数据
                kg_data = [data for data in kg_data if data.get('chunkid') not in chunk_ids_to_remove]
                
                # 保存更新后的数据
                with open(kg_data_path, 'w', encoding='utf-8') as f:
                    json.dump(kg_data, f, ensure_ascii=False, indent=2)
            
            # 3. 从Neo4j中删除对应的数据
            await self.preparer.kg_builder.remove_data_by_chunk_ids(list(chunk_ids_to_remove))
            
            # 4. 更新处理记录
            update_log = self._load_update_log()
            update_log["processed_chunks"] = list(set(update_log["processed_chunks"]) - chunk_ids_to_remove)
            self._save_update_log(update_log)
            
            print(f"已删除与{len(pdf_files)}个PDF文件相关的{len(chunk_ids_to_remove)}个数据块")

async def main():
    updater = KnowledgeBaseUpdater()
    await updater.update_knowledge_base()

if __name__ == "__main__":
    asyncio.run(main())