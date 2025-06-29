from embedding_handler import EmbeddingHandler
from typing import Dict, List
import neo4j
import os
import json
import faiss
import numpy as np
import re
class KGBuilder:
    def __init__(self, 
        neo4j_uri: str = 'bolt://localhost:7687', 
        neo4j_user: str = 'neo4j', 
        neo4j_password: str = 'taidineo4j',
        embeddingHandler: EmbeddingHandler = None
    ):
        self.driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        self.embedding_handler = embeddingHandler or EmbeddingHandler()
        self.entity_index = None
        self.relation_index = None
        self.entity_texts = []
        self.relation_texts = []
        # self.retriever = Retriever(self.embedding_handler, self)  # 添加retriever初始化

    def extract_entities_relations(self, entities: List[Dict], relations: List[Dict]):
        """
        使用大模型从文本中提取实体和关系
        
        Args:
            text: 输入文本
            
        Returns:
            包含实体和关系的字典
        """
        prompt = f"""根据所给文本提取实体和关系:
        返回格式:
        {{
            "entities": [
                {{"name": "实体1", "type": "类型"}},
                {{"name": "实体2", "type": "类型"}}
            ],
            "relations": [
                {{"subject": "实体1", "predicate": "关系", "object": "实体2"}}
            ]
        }}"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            stream=False
        )
        # response = self.deepseek.generate(prompt)
        content = response.choices[0].message.content
        print(content)
        

        
        # 尝试从三重引号中提取JSON
        json_match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
        
        if json_match:
            # 从三重引号中提取内容
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("无法解析提取的JSON内容，尝试直接解析整个响应")
        
        # 如果没有三重引号或解析失败，尝试直接解析整个内容
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("无法直接解析响应为JSON，返回空结果")
            return {"entities": [], "relations": []}


    def process_text(self, text: str, skip_neo4j=False):
        """
        处理文本: 提取实体和关系并存储到知识图谱
        
        Args:
            text: 输入文本
            skip_neo4j: 是否跳过Neo4j存储步骤
        
        Returns:
            提取的实体和关系
        """
        result = self.extract_entities_relations(text)
        print(f"提取结果: {result}")
        
        if not skip_neo4j:
            try:
                self.store_to_neo4j(result["entities"], result["relations"])
                print("成功存储到Neo4j")
            except neo4j.exceptions.ServiceUnavailable as e:
                print(f"Neo4j连接失败: {e}")
                print("请确保Neo4j服务器已启动且可访问")
        
        return result

    def save_vector_indexes(self, dir_path: str):
        """保存向量索引到文件"""
        os.makedirs(dir_path, exist_ok=True)
        
        # 保存实体索引
        faiss.write_index(self.entity_index, os.path.join(dir_path, "entity.index"))
        with open(os.path.join(dir_path, "entity_texts.json"), "w") as f:
            json.dump(self.entity_texts, f)
        
        # 保存关系索引
        faiss.write_index(self.relation_index, os.path.join(dir_path, "relation.index"))
        with open(os.path.join(dir_path, "relation_texts.json"), "w") as f:
            json.dump(self.relation_texts, f)
    
    def load_vector_indexes(self, dir_path: str):
        """从文件加载向量索引"""
        # 加载实体索引
        self.entity_index = faiss.read_index(os.path.join(dir_path, "entity.index"))
        with open(os.path.join(dir_path, "entity_texts.json"), "r") as f:
            self.entity_texts = json.load(f)
        
        # 加载关系索引
        self.relation_index = faiss.read_index(os.path.join(dir_path, "relation.index"))
        with open(os.path.join(dir_path, "relation_texts.json"), "r") as f:
            self.relation_texts = json.load(f)

    def query_knowledge_graph(self, query_lowkeyword, query_highkeyword) -> List[Dict]:
        """查询知识图谱并返回关联信息的自然语言描述"""
        # 查询实体
        entity_results = []
        for keyword in query_lowkeyword:
            entity_results.extend(self.vector_entity_retrieve(keyword))
            # if entity_results:
            #     break
        # entity_results = self.vector_entity_retrieve(query_lowkeyword)
        # 查询关系
        relation_results = []
        for keyword in query_highkeyword:
            relation_results.extend(self.vector_relation_retrieve(keyword))
            # if relation_results:
            #     break
        # relation_results = self.vector_relation_retrieve(query_highkeyword)
        
        entity_results_knowdge = []
        relation_results_knowdge = []
        # 处理实体结果
        for r in entity_results:
            print(r)
            entity_name = r["entity"]["name"].split()  # 提取实体名称                  
            entity_type = r["entity"]["type"]
            entity_description = r["entity"]["description"]
            entity_chunkid = r["entity"]["chunkid"]
            with self.driver.session() as session:
                # 查询该实体作为主语的三元组
                subject_relations = session.run(
                    """MATCH (s:Entity {name: $name, description: $description})-[r]->(o:Entity)
                    RETURN s.name as subject, s.type as subject_type, s.description as subject_description, 
                           s.chunk_id as subject_chunkid, r.predicate as predicate, 
                           o.name as object, o.type as object_type, o.description as object_description,
                           o.chunk_id as object_chunkid, r.type as rel_type, 
                           r.strength as rel_strength, r.description as rel_description,
                           r.chunk_id as rel_chunkid""",
                    name=entity_name,
                    description=entity_description
                )
                for record in subject_relations:
                    entity_results_knowdge.extend([
                        {
                            "name":record["object"],
                            "type":record["object_type"],
                            "description":record["object_description"],
                            "chunkid":record["object_chunkid"],
                        }
                    ])
                    relation_results_knowdge.extend([
                        {
                            "subject": record["subject"],
                            "object": record["object"],
                            "description": record["rel_description"],
                            "predicate": record["predicate"],
                            "strength": record["rel_strength"],
                            "chunkid": record["rel_chunkid"]
                        }
                    ])
                # 查询该实体作为宾语的三元组
                object_relations = session.run(
                    """MATCH (s:Entity)-[r]->(o:Entity {name: $name, description: $description})
                    RETURN s.name as subject, s.type as subject_type, s.description as subject_description,
                           s.chunk_id as subject_chunkid, r.predicate as predicate,
                           o.name as object, o.type as object_type, o.description as object_description,
                           o.chunk_id as object_chunkid, r.type as rel_type,
                           r.strength as rel_strength, r.description as rel_description,
                           r.chunk_id as rel_chunkid""",
                    name=entity_name,
                    description=entity_description
                )
                for record in object_relations:
                    entity_results_knowdge.extend([
                        {
                            "name":record["subject"],
                            "type":record["subject_type"],
                            "description":record["subject_description"],
                            "chunkid":record["subject_chunkid"],
                        }
                    ])
                    relation_results_knowdge.extend([
                        {
                            "subject": record["subject"],
                            "object": record["object"],
                            "description": record["rel_description"],
                            "predicate": record["predicate"],
                            "strength": record["rel_strength"],
                            "chunkid": record["rel_chunkid"]
                        }
                    ])

                
                # # 按type分类关系
                # type_groups = {}
                # for record in subject_relations:
                #     rel_type = record.get("type", "default")
                #     if rel_type not in type_groups:
                #         type_groups[rel_type] = []
                #     type_groups[rel_type].append(f"{record['subject']} {record['predicate']} {record['object']}")
                
                # for record in object_relations:
                #     rel_type = record.get("type", "default")
                #     if rel_type not in type_groups:
                #         type_groups[rel_type] = []
                #     type_groups[rel_type].append(f"{record['subject']} {record['predicate']} {record['object']}")
                
                # # 为每个type生成一个结果项
                # for rel_type, desc_list in type_groups.items():
                #     # combined_text = "\n".join(desc_list)
                #     desc_list = list(set(desc_list))
                #     for desc in desc_list:
                #         # combined_text = f"{entity_name} {rel_type} {desc}"
                #         results.append({
                #             "type": "entity",
                #             "name": entity_name,
                #             "relation_type": rel_type,
                #             # "score": 1 - r["distance"],
                #             "description": desc
                #         })
                    # results.append({
                    #     "type": "entity",
                    #     "name": entity_name,
                    #     "relation_type": rel_type,
                    #     "score": 1 - r["distance"],
                    #     "description": combined_text
                    # })
        
        # 处理关系结果
        # for r in relation_results:
        #     relation_text = r["relation"]
        #     parts = relation_text.split()
        #     if len(parts) >= 3:
        #         combined_text = f"{relation_text}"
        #         results.append({
        #             "type": "relation",
        #             "content": relation_text,
        #             "score": 1 - r["distance"],
        #             "description": combined_text
        #         })
        
        return entity_results+entity_results_knowdge, relation_results+relation_results_knowdge

    def extract_entities_relations_batch(self, texts: List[str]) -> Dict:
        """批量提取实体和关系(顺序处理版本)"""
        entities = []
        relations = []
        
        for text in texts:
            try:
                result = self.extract_entities_relations(text)
                entities.extend(result["entities"])
                relations.extend(result["relations"])
            except Exception as e:
                print(f"处理文本时出错: {e}")
        
        # 保存提取结果到文件
        os.makedirs("/root/ty/data/kg_extractions", exist_ok=True)
        with open("/root/ty/data/kg_extractions/latest_extraction.json", "w", encoding="utf-8") as f:
            json.dump({
                "entities": entities,
                "relations": relations,
                "source_texts": texts
            }, f, ensure_ascii=False)
        
        return {
            "entities": entities,
            "relations": relations
        }

    def get_all_entities(self) -> List[Dict]:
        """
        获取知识图谱中的所有实体
        
        Returns:
            包含实体名称和类型的字典列表
        """
        entities = []
        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e.name as name, e.type as type")
            for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"]
                })
        return entities

    def vector_entity_retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        向量实体检索: 使用嵌入向量查找相似实体
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似实体列表
        """
        query_embedding = np.array(self.embedding_handler.embed_texts([query])).astype('float32')
        distances, indices = self.entity_index.search(query_embedding, k)
        
        return [
            {
                "entity": self.entity_texts[idx],
                "distance": float(dist)
            }
            for idx, dist in zip(indices[0], distances[0])
        ]
    
    def vector_relation_retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        向量关系检索: 使用嵌入向量查找相似关系
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似关系列表
        """
        query_embedding = np.array(self.embedding_handler.embed_texts([query])).astype('float32')
        distances, indices = self.relation_index.search(query_embedding, k)
        
        return [
            {
                "relation": self.relation_texts[idx],
                "distance": float(dist)
            }
            for idx, dist in zip(indices[0], distances[0])
        ]

    def store_to_neo4j(self, kg_data):
        """
        将实体和关系存储到Neo4j数据库并进行向量化
        """
        # 准备向量化文本
        entity_texts = []
        relation_texts = []
        raw_entities = []  # 新增：存储原始实体数据
        raw_relations = []  # 新增：存储原始关系数据
        
        for data in kg_data:
            # 处理实体
            for entity in data["entity"]:
                entity_texts.append(f"{entity[0]}: {entity[2]}")
                # 新增：保存原始实体数据并附加chunkid
                raw_entities.append({
                    "name": entity[0],
                    "type": entity[1],
                    "description": entity[2],
                    "chunkid": data["chunkid"]
                })
            
            # 处理关系
            for rel in data["relationship"]:
                relation_texts.append(f"{rel[3]}|{rel[0]}→{rel[1]}: {rel[2]}")
                # 新增：保存原始关系数据并附加chunkid
                raw_relations.append({
                    "subject": rel[0],
                    "object": rel[1],
                    "description": rel[2],
                    "predicate": rel[3],
                    "strength": rel[4],
                    "chunkid": data["chunkid"]
                })
            # entities.extend(data["entity"])
            # relations.extend(data["relationship"])
        # 生成向量
        if entity_texts:
            self.entity_embeddings = self.embedding_handler.embed_texts(entity_texts)
            self.entity_texts = raw_entities
            # 创建实体索引
            dim = len(self.entity_embeddings[0]) if self.entity_embeddings else 768
            self.entity_index = faiss.IndexFlatIP(dim)
            if len(self.entity_embeddings) > 0:
                self.entity_index.add(np.array(self.entity_embeddings).astype('float32'))
                print(f"创建实体索引，包含 {self.entity_index.ntotal} 个向量")
        
        if relation_texts:
            self.relation_embeddings = self.embedding_handler.embed_texts(relation_texts)
            self.relation_texts = raw_relations
            # 创建关系索引
            dim = len(self.relation_embeddings[0]) if self.relation_embeddings else 768
            self.relation_index = faiss.IndexFlatIP(dim)
            if len(self.relation_embeddings) > 0:
                self.relation_index.add(np.array(self.relation_embeddings).astype('float32'))
                print(f"创建关系索引，包含 {self.relation_index.ntotal} 个向量")
        
        # 存储到Neo4j
        with self.driver.session() as session:
            # 存储实体
            for data in kg_data:
                for entity in data["entity"]:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, 
                            e.description = $description,
                            e.chunk_id = $chunk_id
                        """,
                        name=entity[0],
                        type=entity[1],
                        description=entity[2],
                        chunk_id=data["chunkid"]  # 添加chunk_id字段
                    )
                
                # 存储关系
                for relation in data["relationship"]:
                    session.run(
                        """
                        MATCH (s:Entity {name: $subject}), (o:Entity {name: $object})
                        MERGE (s)-[r:RELATIONSHIP {predicate: $predicate}]->(o)
                        SET r.strength = $strength,
                            r.description = $description,
                            r.chunk_id = $chunk_id
                        """,
                        subject=relation[0],
                        object=relation[1],
                        predicate=relation[3],
                        description=relation[2],
                        strength=relation[4],
                        chunk_id=data["chunkid"]  # 添加chunk_id字段
                    )
                

if __name__ == "__main__":
    # 测试代码
    kg = KGBuilder("bolt://localhost:7687", "neo4j", "taidineo4j")
    test_text = """第七届全国青少年人工智能创新挑战赛3D编程模型创新设计专项赛由中国少年儿童发展服务中心主办。"""
    # 设置skip_neo4j=True跳过Neo4j存储步骤
    kg.process_text(test_text, skip_neo4j=True)


