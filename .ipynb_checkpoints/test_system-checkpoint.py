import json
import os
import asyncio
import aiohttp
from typing import List, Dict, Optional
import time
from main import ChatSystem

class AsyncLimiter:
    """令牌桶限流器"""
    def __init__(self, max_rate: int, time_period: float = 1.0):
        self.max_rate = max_rate
        self.time_period = time_period
        self.tokens = max_rate
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            
            # 计算新增的令牌数
            new_tokens = elapsed * (self.max_rate / self.time_period)
            self.tokens = min(self.max_rate, self.tokens + new_tokens)
            self.updated_at = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                # 计算需要等待的时间
                wait_time = (1 - self.tokens) * (self.time_period / self.max_rate)
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.updated_at = time.monotonic()
                return True

class QAEvaluator:
    def __init__(self, max_rate: int = 10, time_period: float = 1.0):
        self.chat_system = ChatSystem()
        self.qa_data_dir = "/root/autodl-tmp/qadata"
        self.api_key = "sk-4c48568e7bcb45b386cda7f51f58386a"
        self.base_url = "https://api.deepseek.com"
        self.rate_limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
        self._session_lock = asyncio.Lock()
        self.session = None  # 延迟初始化
    
    async def get_session(self) -> aiohttp.ClientSession:
        """延迟初始化并返回会话"""
        async with self._session_lock:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
            return self.session
    
    def load_qa_data(self) -> List[Dict]:
        """加载所有QA测试数据"""
        qa_pairs = []
        for filename in os.listdir(self.qa_data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.qa_data_dir, filename), 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            qa_pairs.extend(data)
                        else:
                            qa_pairs.append(data)
                    except json.JSONDecodeError:
                        print(f"警告：无法解析文件 {filename}")
        return qa_pairs
    
    async def evaluate_answer(self, question: str, expected_answer: str, actual_answer: str) -> Optional[Dict]:
        """使用大模型异步评估答案质量"""
        session = await self.get_session()
        
        prompt = f"""请评估以下问答对的质量，从准确性、完整性和相关性三个方面进行评分（1-5分），并给出简要解释：

问题：{question}
标准答案：{expected_answer}
系统回答：{actual_answer}

请按以下格式输出：
准确性评分：x/5
完整性评分：x/5
相关性评分：x/5
解释：xxx"""

        try:
            # 应用速率限制
            await self.rate_limiter.acquire()
            
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一个专业的问答系统评估专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                }
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"API请求失败: {error}")
                    return {
                        "evaluation": "API请求失败",
                        "raw_response": actual_answer
                    }
                
                data = await response.json()
                evaluation = data["choices"][0]["message"]["content"]
                return {
                    "evaluation": evaluation,
                    "raw_response": actual_answer
                }
        except Exception as e:
            print(f"评估答案时出错: {str(e)}")
            return {
                "evaluation": f"评估错误: {str(e)}",
                "raw_response": actual_answer
            }
    
    async def process_question(self, qa: Dict, index: int, total: int) -> Dict:
        """异步处理单个问题"""
        question = qa["question"]
        expected_answer = qa["answer"]
        
        start_time = time.time()
        try:
            result = self.chat_system.ask(question)
            actual_answer = result["answer"]
            end_time = time.time()
            
            # 使用大模型评估答案
            evaluation = await self.evaluate_answer(question, expected_answer, actual_answer)
            
            print(f"\n问题 {index}/{total}:")
            print(f"问题: {question}")
            print(f"系统回答: {actual_answer}")
            if evaluation is not None:
                print(f"评估结果: {evaluation['evaluation']}")
            print(f"耗时: {end_time - start_time:.2f}秒")
            
            return {
                "question": question,
                "expected_answer": expected_answer,
                "evaluation": evaluation,
                "time": end_time - start_time
            }
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")
            return {
                "question": question,
                "error": str(e)
            }
    
    async def evaluate(self):
        """异步执行评估"""
        qa_pairs = self.load_qa_data()
        total = len(qa_pairs)
        evaluations = []
        
        print(f"开始评估，共 {total} 个问题...")
        print(f"速率限制: {self.rate_limiter.max_rate} 请求/秒")
        start_time = time.time()
        
        tasks = []
        for i, qa in enumerate(qa_pairs, 1):
            task = asyncio.create_task(self.process_question(qa, i, total))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        evaluations = [result for result in results if "error" not in result]
        
        # 关闭会话
        if self.session and not self.session.closed:
            await self.session.close()
        
        # 保存评估结果
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        
        # 输出总体评估结果
        total_time = time.time() - start_time
        avg_time = total_time / len(evaluations) if evaluations else 0
        print("\n评估完成！")
        print(f"总问题数: {total}")
        print(f"实际评估问题数: {len(evaluations)}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均响应时间: {avg_time:.2f}秒")
        print("详细评估结果已保存到 evaluation_results.json")

async def main():
    # 在这里设置速率限制，例如每秒最多5个请求
    evaluator = QAEvaluator(max_rate=1, time_period=5.0)
    await evaluator.evaluate()

if __name__ == "__main__":
    asyncio.run(main())