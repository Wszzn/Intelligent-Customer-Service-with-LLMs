from typing import List, Dict, Optional
from ahocorasick import Automaton
from embedding_handler import EmbeddingHandler
import numpy as np
from kg_builder import KGBuilder

class Retriever:
    def __init__(self, embedding_handler: EmbeddingHandler, kg_builder: KGBuilder):
        """
        初始化检索器
        
        Args:
            embedding_handler: 嵌入处理器实例
            kg_builder: 知识图谱构建器实例
        """
        self.embedding_handler = embedding_handler
        self.kg_builder = kg_builder
        self.ac_automaton = Automaton()
        
        # 初始化BM25
        self.bm25 = None
        self.chunks = []
        
        # 初始化AC自动机
        self.init_ac_automaton()  # 只需要调用一次初始化
    
    def init_ac_automaton(self):
        """初始化AC自动机"""
        # 从知识图谱加载所有实体名称
        entities = self.kg_builder.get_all_entities()
        
        # 添加实体到自动机
        for entity in entities:
            self.ac_automaton.add_word(entity['name'], entity)
            
        # 构建自动机
        self.ac_automaton.make_automaton()
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Unified search method that combines multiple retrieval methods
        """
        results = []
        
        # 1. Vector retrieval
        vector_results = self.vector_retrieve(query, k)
        results.extend({
            "document": r["document"],
            "score": 1 - r["distance"],
            "type": "vector"
        } for r in vector_results)
        
        # 2. BM25 retrieval
        bm25_results = self.bm25_retrieve(query, k)
        results.extend({
            "document": r["document"],
            "score": r["score"],
            "type": "bm25"
        } for r in bm25_results)
        
        # # 3. 实体检索 (改为调用kg_builder的方法)
        # entity_results = self.kg_builder.vector_entity_retrieve(query, k)
        # results.extend({
        #     "name": r["entity"],
        #     "type": "entity",
        #     "score": 1 - r["distance"],
        #     "source": "entity"
        # } for r in entity_results)
        
        # # 4. 关系检索 (改为调用kg_builder的方法)
        # relation_results = self.kg_builder.vector_relation_retrieve(query, k)
        # results.extend({
        #     "name": r["relation"],
        #     "type": "relation",
        #     "score": 1 - r["distance"],
        #     "source": "relation"
        # } for r in relation_results)
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]
    
    # def vector_entity_retrieve(self, query: str, k: int = 5) -> List[Dict]:
    #     """
    #     向量实体检索: 使用嵌入向量查找相似实体
        
    #     Args:
    #         query: 查询文本
    #         k: 返回结果数量
            
    #     Returns:
    #         相似实体列表
    #     """
    #     query_embedding = np.array(self.embedding_handler.embed_texts([query])).astype('float32')
    #     distances, indices = self.kg_builder.entity_index.search(query_embedding, k)
        
    #     return [
    #         {
    #             "entity": self.kg_builder.entity_texts[idx],
    #             "distance": float(dist)
    #         }
    #         for idx, dist in zip(indices[0], distances[0])
    #     ]
    
    # def vector_relation_retrieve(self, query: str, k: int = 5) -> List[Dict]:
    #     """
    #     向量关系检索: 使用嵌入向量查找相似关系
        
    #     Args:
    #         query: 查询文本
    #         k: 返回结果数量
            
    #     Returns:
    #         相似关系列表
    #     """
    #     query_embedding = np.array(self.embedding_handler.embed_texts([query])).astype('float32')
    #     distances, indices = self.kg_builder.relation_index.search(query_embedding, k)
        
    #     return [
    #         {
    #             "relation": self.kg_builder.relation_texts[idx],
    #             "distance": float(dist)
    #         }
    #         for idx, dist in zip(indices[0], distances[0])
    #     ]
    
    def vector_retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        向量检索: 使用embedding_handler的query_embeddings方法查找相似文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        # 直接调用embedding_handler的query_embeddings方法
        results = self.embedding_handler.query_embeddings(query, k)
        
        return [
            {
                "document": f"这是关于{r['metadata']}的相关信息：{r['document']}",
                "distance": r["distance"]
            }
            for r in results
        ]
    
    def bm25_retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        BM25检索: 使用BM25算法查找相似文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if self.bm25 is None:
            return []
            
        # 获取BM25分数
        scores = self.bm25.get_scores(query.split())
        
        # 获取top k结果
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return [
            {
                "document": self.chunks[idx],
                "score": float(scores[idx])
            }
            for idx in top_k_indices
        ]
if __name__ == "__main__":
    # 测试两种模型
    for model_type in ["qwen", "bert"]:
        print(f"\n===== 测试 {model_type} 模型 =====")
        
        # 初始化
        embedding_handler = EmbeddingHandler(model_type=model_type)
        kg_builder = KGBuilder("bolt://localhost:7687", "neo4j", "password")
        retriever = Retriever(embedding_handler, kg_builder)
        
        # 模拟数据
        test_chunks = ["这是第一个测试chunk", "这是第二个测试chunk"]
        test_entities = ["测试实体1", "测试实体2"]
        test_relations = ["测试关系1", "测试关系2"]
        
        # 构建索引
        retriever.build_bm25_index(test_chunks)
        retriever.build_ac_automaton(test_entities)
        
        # 测试所有检索方法
        query = "测试"
        print("\n向量检索结果:", retriever.vector_retrieve(query))
        print("实体检索结果:", retriever.entity_retrieve(query))
        print("关系检索结果:", retriever.relation_retrieve(query))
        print("BM25检索结果:", retriever.bm25_retrieve(query))
        print("向量实体检索结果:", retriever.vector_entity_retrieve(query))
        print("向量关系检索结果:", retriever.vector_relation_retrieve(query))