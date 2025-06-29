import os
import json  # 添加这行导入
import torch
from typing import List, Optional, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models import Model
from modelscope.preprocessors import Preprocessor
import re

class EmbeddingHandler:
    def __init__(self, model_type: str = "qwen", model_name: Optional[str] = None, load_from: str = None):
        """
        初始化嵌入模型和向量数据库
        
        Args:
            model_type: 模型类型 ("qwen"或"bert")
            model_name: 使用的模型名称/路径
            load_from: 从指定路径加载已保存的索引
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.stopwords_path = "/root/ty/data/stopwords_hit.txt"
        self.stopwords = self._load_stopwords()
        
        if model_type == "qwen":
            self.model = SentenceTransformer(
                model_name or "/root/autodl-tmp/modelscope/models/gte_qwen_1.5b",
                trust_remote_code=False,
                device=self.device  # 添加设备参数
            )
            self.model.max_seq_length = 4096
        elif model_type == "bert":
            # 添加模型加载进度显示
                print("正在加载BERT模型...")
                self.pipeline = pipeline(
                    Tasks.sentence_embedding,
                    model=model_name or "/root/autodl-tmp/modelscope/models/gte_embedding_bert",
                    sequence_length=512,
                    device=self.device,
                    # 添加以下优化参数
                    model_revision='v1.0',  # 指定模型版本
                    pipeline_kwargs={
                        'cache_dir': '/root/autodl-tmp/cache',  # 指定缓存目录
                        'load_in_8bit': True  # 8位量化减少内存占用
                    }
                )
                print("BERT模型加载完成")
        
        # 初始化FAISS向量数据库
        self.index = None
        self.documents = []
        self.document_ids = []
        self.metadata = []
    
    def _load_stopwords(self) -> set:
        """加载停用词表"""
        if os.path.exists(self.stopwords_path):
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        return set()
    
    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理
        1. 过滤停用词
        2. 过滤标点符号
        3. 清理多余空格
        """
        # 过滤标点符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        # 分割成词
        words = text.split()
        
        # 过滤停用词
        words = [word for word in words if word not in self.stopwords]
        
        # 重新组合文本
        return ' '.join(words)
    
    def embed_texts(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        对文本进行嵌入
        Args:
            texts: 单个文本或文本列表
        Returns:
            嵌入向量列表
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 预处理文本
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        if self.model_type == "qwen":
            return self.model.encode(processed_texts, batch_size=1).tolist()
        else:  # bert
            inputs = {"source_sentence": processed_texts}
            result = self.pipeline(input=inputs)
            return result["text_embedding"].tolist()
    
    def store_embeddings(self, chunks: List[str], metadata: List[dict] = None):
        """
        将chunk及其嵌入向量存储到向量数据库
        
        Args:
            chunks: 文本chunk列表
            metadata: 可选的元数据列表
        """
        if metadata is None:
            metadata = [{} for _ in chunks]
            
        # 逐个处理chunk
        for i, chunk in enumerate(chunks):
            # 单个chunk向量化
            embedding = np.array(self.embed_texts([chunk])).astype('float32')
            
            # 生成ID
            doc_id = str(len(self.documents))
            
            # 保存文档和ID
            self.documents.append(chunk)
            self.document_ids.append(doc_id)
            
            # 初始化或更新FAISS索引
            if self.index is None:
                d = embedding.shape[1]
                self.index = faiss.IndexFlatL2(d)
                self.index.add(embedding)
            else:
                self.index.add(embedding)
    
    def query_embeddings(self, query: str, k: int = 5) -> List[dict]:
        """
        查询与输入文本最相似的k个chunk
        """
        query_embedding = np.array(self.embed_texts([query])).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        return [
            {
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "chunkid": self.document_ids[idx],
                "distance": float(dist)
            }
            for idx, dist in zip(indices[0], distances[0])
        ]
    
    def save_index(self, dir_path: str):
        """保存FAISS索引到文件"""
        os.makedirs(dir_path, exist_ok=True)
        
        # 保存索引
        faiss.write_index(self.index, os.path.join(dir_path, "faiss.index"))
        
        # 保存文档和ID
        with open(os.path.join(dir_path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "document_ids": self.document_ids
            }, f, ensure_ascii=False)

    def load_index(self, dir_path: str):
        """从文件加载FAISS索引"""
        try:
            # 加载索引
            index_path = os.path.join(dir_path, "faiss.index")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            else:
                print(f"警告: 索引文件 {index_path} 不存在")
                self.index = None
            
            # 加载文档和ID
            docs_path = os.path.join(dir_path, "documents.json")
            if os.path.exists(docs_path):
                with open(docs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = [f"[{data[i]['metadata']['source']}][{data[i]['metadata']['Header 1']}]{[val for val in data[i]['keywords']]}{data[i]['chunk']}" for i in range(len(data))]
                    # self.documents = [chunk["chunk"] for chunk in data]
                    self.document_ids = [chunk["id"] for chunk in data]
                    self.metadata = [chunk["metadata"]["source"] for chunk in data]
            else:
                print(f"警告: 文档文件 {docs_path} 不存在")
                self.documents = []
                self.document_ids = []
                
        except json.JSONDecodeError:
            print(f"错误: 无法解析 {docs_path}，文件可能为空或损坏")
            self.documents = []
            self.document_ids = []
        except Exception as e:
            print(f"加载索引时发生错误: {str(e)}")
            self.index = None
            self.documents = []
            self.document_ids = []

if __name__ == "__main__":
    # 测试Qwen模型
    qwen_handler = EmbeddingHandler(model_type="qwen")
    # 测试BERT模型
    bert_handler = EmbeddingHandler(model_type="bert")
