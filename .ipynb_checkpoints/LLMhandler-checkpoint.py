from __future__ import annotations
import aiohttp
import json
import re
from typing import Optional, Dict, Any
from typing import Any
from openai import OpenAI
from prompts import *
import json
import re
from typing import List, Dict, Any
import asyncio
from aiolimiter import AsyncLimiter
GRAPH_FIELD_SEP = "<SEP>"
class AsyncLLMHandler:
    def __init__(self):
        self.api_key = "sk-4c48568e7bcb45b386cda7f51f58386a"
        # self.api_key = "sk-hjexxiqvkahyxeznmxskiqhwxvjanwdsfodxrtuvcntzqjio"
        self.base_url = "https://api.deepseek.com"
        # self.base_url = "https://api.siliconflow.cn"
        self.client = None  # 将在__aenter__中初始化
        self._session = None  # 使用下划线前缀的统一命名
        self._session_lock = asyncio.Lock()
        self.rate_limiter = AsyncLimiter(max_rate=10, time_period=5)  # 每秒最多5个请求
    async def ensure_session(self):
        """确保session已初始化"""
        if self._session is None:  # 现在这个检查可以正常工作了
            async with self._session_lock:
                if self._session is None:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=600)  # 设置60秒超时
                    )
        return self._session

    async def close(self):
        """手动关闭session"""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """支持上下文管理器"""
        await self.ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭"""
        await self.close()

    async def _make_request(self, model: str, messages: list, temperature: float = 0.0) -> str:
        """自动处理session初始化"""
        async with self.rate_limiter:  # 限流控制
            session = await self.ensure_session()

            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": False
            }


            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data['choices'][0]['message']['content']

    async def entity_extraction(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": ENTITY_EXTRACTION.format(
                    language="中文",
                    examples="\n".join([example for example in ENTITY_EXTRACTION_EXAMPLES])
                )
            },
            {
                "role": "user",
                "content": f"""
                #############################
                ---真实数据---
                ######################
                Entity_types(实体类型):[{DEFAULT_ENTITY_TYPES}]
                Text(文本):'''
                {text}
                '''"""
            }
        ]
        return await self._make_request("deepseek-chat", messages)
        # return await self._make_request("deepseek-ai/DeepSeek-V3", messages)


    async def entity_continue_extraction(self, text: str, first_extraction_result: str) -> str:
        messages = [
            {
                "role": "system",
                "content": ENTITY_EXTRACTION.format(
                    language="中文",
                    tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
                    record_delimiter=DEFAULT_RECORD_DELIMITER,
                    completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
                    examples="\n".join([example for example in ENTITY_EXTRACTION_EXAMPLES])
                )
            },
            {
                "role": "user",
                "content": f"""
                #############################
                ---真实数据---
                ######################
                Entity_types(实体类型):[{DEFAULT_ENTITY_TYPES}]
                Text(文本):
                {text}
                ######################
                Output(输出):"""
            },
            {
                "role": "assistant",
                "content": first_extraction_result
            },
            {
                "role": "user",
                "content": entity_continue_extraction.format(
                    language="中文",
                    tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
                    record_delimiter=DEFAULT_RECORD_DELIMITER,
                    completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
                    entity_types=DEFAULT_ENTITY_TYPES
                ) + text
            }
        ]
        return await self._make_request("deepseek-chat", messages)
        # return await self._make_request("deepseek-ai/DeepSeek-V3", messages)

    async def entity_if_loop_extraction(self, text: str, first_extraction_result: str, second_extraction_result: str) -> str:
        messages = [
            {
                "role": "system",
                "content": ENTITY_EXTRACTION.format(
                    language="中文",
                    tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
                    record_delimiter=DEFAULT_RECORD_DELIMITER,
                    completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
                    examples="\n".join([example for example in ENTITY_EXTRACTION_EXAMPLES]) + text
                )
            },
            {
                "role": "user",
                "content": f"""
                #############################
                ---真实数据---
                ######################
                Entity_types(实体类型):[{DEFAULT_ENTITY_TYPES}]
                Text(文本):
                {text}
                ######################
                Output(输出):"""
            },
            {
                "role": "assistant",
                "content": first_extraction_result
            },
            {
                "role": "user",
                "content": entity_continue_extraction.format(
                    language="中文",
                    tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
                    record_delimiter=DEFAULT_RECORD_DELIMITER,
                    completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
                    entity_types=DEFAULT_ENTITY_TYPES
                )
            },
            {
                "role": "assistant",
                "content": second_extraction_result
            },
            {
                "role": "user",
                "content": entity_if_loop_extraction.format(language="中文")
            }
        ]
        return await self._make_request("deepseek-chat", messages)

    async def get_second_extraction_result(self, text: str) -> Dict[str, Any]:
        """
        异步执行两次实体提取并返回第二次的结果
        
        Args:
            text: 输入文本
            
        Returns:
            第二次实体提取的结果
        """
        # 第一次提取
        first_result = await self.entity_extraction(text)
        
        # 第二次提取
        second_result = await self.entity_continue_extraction(text, first_result)
        print(second_result)
        
        # 尝试从三重引号中提取JSON
        json_match = re.search(r"```(?:json)?(.*?)```", second_result, re.DOTALL)
        
        if json_match:
            # 从三重引号中提取内容
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("无法解析提取的JSON内容，尝试直接解析整个响应")
        
        # 如果没有三重引号或解析失败，尝试直接解析整个内容
        try:
            return json.loads(second_result)
        except json.JSONDecodeError:
            print("无法直接解析响应为JSON，返回空结果")
            return {"entities": [], "relations": [], "content_keywords": []}