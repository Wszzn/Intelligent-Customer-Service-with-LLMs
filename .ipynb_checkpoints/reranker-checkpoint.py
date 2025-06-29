from typing import List, Dict, Optional
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import FlagReranker
import torch

class Reranker:
    def __init__(self, model_type: str = "bge"):
        """
        初始化结果重排序器
        
        Args:
            model_type: 模型类型，可选"gte"或"bge"
        """
        self.model_type = model_type
        if model_type == "gte":
            self.model_name = "/root/autodl-tmp/modelscope/models/gte_rerank"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model.eval()
        elif model_type == "bge":
            self.reranker = FlagReranker('/root/autodl-tmp/modelscope/models/bge_reranker_v2_m3', use_fp16=True)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def rerank(self, results: List[Dict], query: str, normalize: bool = True) -> List[Dict]:
        """
        对检索结果进行重新排序
        
        Args:
            results: 检索结果列表
            query: 查询文本
            normalize: 是否对分数进行归一化(0-1)，仅对bge模型有效
            
        Returns:
            重排序后的结果列表
        """
        if not results:
            return []
            
        if self.model_type == "gte":
            # 准备输入对
            pairs = [
                [query, result.get("document") or result.get("description")]
                for result in results
            ]
            
            # 获取排序分数
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=8192
                )
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
            
            # 将分数添加到结果中并排序
            final_results = []
            for i, result in enumerate(results):
                if scores[i] < 0.3:  # 分数阈值
                    continue
                final_results.append({
                    "document": result.get("document") or result.get("description"),
                    "rerank_score": scores[i]
                })
                
            return sorted(final_results, key=lambda x: x['rerank_score'], reverse=True)
        else:  # bge模型
            pairs = [[query, result.get("document") or result.get("description")] 
                    for result in results]
            scores = self.reranker.compute_score(pairs, normalize=normalize)
            print(pairs)
            final_results = []
            for i, result in enumerate(results):
                if scores[i] < (0.3 if normalize else 0):  # 根据是否归一化调整阈值
                    continue
                final_results.append({
                    "document": result.get("document") or result.get("description"),
                    "rerank_score": float(scores[i])
                })
                
            return sorted(final_results, key=lambda x: x['rerank_score'], reverse=True)
