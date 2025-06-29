from __future__ import annotations
from typing import Any
from openai import OpenAI
from prompts import *
import json
import re
from typing import List, Dict, Any
GRAPH_FIELD_SEP = "<SEP>"
class LLMHandler:
    def __init__(self):
        self.client = OpenAI(api_key="sk-4c48568e7bcb45b386cda7f51f58386a", base_url="https://api.deepseek.com")
    def entity_extraction(self, text):
        # 调用大模型进行实体提取
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": ENTITY_EXTRACTION.format(
                        language="中文",
                        examples="\n".join([example for example in ENTITY_EXTRACTION_EXAMPLES
                        ])
                    )}, 
                {
                    "role": "user",
                    "content": """ 
                            #############################
                            ---真实数据---
                            ######################
                            Entity_types(实体类型):[{entity_types}]
                            Text(文本):'''
                            {input_text}
                            '''""".format(
                                entity_types=DEFAULT_ENTITY_TYPES,
                                input_text=text,
                            )
                }
            ],
            temperature=0.0,
            stream=False
        )
        # print(ENTITY_EXTRACTION.format(
        #                 language="中文",
        #                 tuple_delimiter= DEFAULT_TUPLE_DELIMITER,
        #                 record_delimiter=DEFAULT_RECORD_DELIMITER,
        #                 completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
        #                 examples="\n".join([example for example in ENTITY_EXTRACTION_EXAMPLES
        #                 ]))
        # )
        return response.choices[0].message.content
    def entity_continue_extraction(self, text, first_extraction_result):
        # 调用大模型进行实体继续提取
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": ENTITY_EXTRACTION.format(
                        language="中文",
                        tuple_delimiter= DEFAULT_TUPLE_DELIMITER,
                        record_delimiter=DEFAULT_RECORD_DELIMITER,
                        completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
                        examples="\n".join([example for example in ENTITY_EXTRACTION_EXAMPLES
                        ])
                    )
                }, 
                {
                    "role": "user",
                    "content": """ #############################
                            ---真实数据---
                            ######################
                            Entity_types(实体类型):[{entity_types}]
                            Text(文本):
                            {input_text}
                            ######################
                            Output(输出):""".format(
                                entity_types=DEFAULT_ENTITY_TYPES,
                                input_text=text,
                            )
                },
                {
                    "role": "assistant",
                    "content": first_extraction_result
                },
                {
                    "role": "user",
                    "content": entity_continue_extraction.format(
                        language="中文",
                        tuple_delimiter= DEFAULT_TUPLE_DELIMITER,
                        record_delimiter=DEFAULT_RECORD_DELIMITER,
                        completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
                        entity_types=DEFAULT_ENTITY_TYPES
                    ) + text
                }
            ],
            temperature=0.0,
            stream=False
        )
        return response.choices[0].message.content
    def entity_if_loop_extraction(self, text, first_extraction_result, second_extraction_result):
        # 调用大模型进行实体循环提取判断
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {                    
                    "role": "system",
                    "content": ENTITY_EXTRACTION.format(
                        language="中文",
                        tuple_delimiter= DEFAULT_TUPLE_DELIMITER,
                        record_delimiter=DEFAULT_RECORD_DELIMITER,
                        completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
                        examples="\n".join([example for example in ENTITY_EXTRACTION_EXAMPLES
                        ]) + text
                    )
                },
                {
                    "role": "user",
                    "content": """ #############################
                            ---真实数据---
                            ######################
                            Entity_types(实体类型):[{entity_types}]
                            Text(文本):
                            {input_text}
                            ######################
                            Output(输出):""".format(
                                entity_types=DEFAULT_ENTITY_TYPES,
                                input_text=text,
                            )
                },
                {
                    "role": "assistant",
                    "content": first_extraction_result
                },
                {
                    "role": "user",
                    "content": entity_continue_extraction.format(
                        language="中文",
                        tuple_delimiter= DEFAULT_TUPLE_DELIMITER,
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
                    "content": entity_if_loop_extraction.format(
                        language="中文",
                    )
                }
            ],
            temperature=0.0,
            stream=False
        )
        return response.choices[0].message.content

    def get_second_extraction_result(self, text):
        """
        执行两次实体提取并返回第二次的结果
        
        Args:
            text: 输入文本
            
        Returns:
            第二次实体提取的结果
        """
        # 第一次提取
        first_result = self.entity_extraction(text)
        
        # 第二次提取
        second_result = self.entity_continue_extraction(text, first_result)
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
        # return second_result
    def keywords_extraction(self, query):
        """
        从查询中提取关键词

        Args:
            query: 查询文本

        Returns:
            关键词列表
        """
        # 调用大模型进行关键词提取
        # print(KEYWORDS_EXTRACTION_EXAMPLES)
        content = KEYWORDS_EXTRACTION.format(examples="\n".join(KEYWORDS_EXTRACTION_EXAMPLES))
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system","content": content},
                {"role": "user",
                    "content": f"""
                            ---实际数据---
                            当前查询:{query}
                            输出:"""
                }
            ],
            temperature=0.3,
            stream=False
        )
        result = response.choices[0].message.content
        print("关键词生成结果---------------------\n\n",result)
        json_match = re.search(r"```(?:json)?(.*?)```", result, re.DOTALL)
        
        if json_match:
            # 从三重引号中提取内容
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("无法解析提取的JSON内容，尝试直接解析整个响应")
        
        # 如果没有三重引号或解析失败，尝试直接解析整个内容
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            print("无法直接解析响应为JSON，返回空结果")
            return {"high_level_keywords": [], "low_level_keywords": []}
    def naive_rag_response(self, query, content_data):
        """
        使用Naive RAG方法生成响应
        Args:
            query: 查询文本
            knowledge_base: 知识库
        Returns:
            生成的响应文本
        """
        # 提取关键词
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": NAIVE_RAG_RESPONSE.format(
                        content_data=content_data
                    )
                },
                {
                    "role": "user",
                    "content": f"用户问题为：{query}"
                }
            ]
        )
        return response.choices[0].message.content
    def mix_rag_response(self, query, vector_context, kg_context):
        """
        使用混合RAG方法生成响应
        Args:
            query: 查询文本
            vector_context: 向量上下文
            kg_context: 知识图谱上下文
        Returns:
            生成的响应文本
        """
        # 调用大模型进行混合RAG
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": MIX_RAG_RESPONSE.format(
                        vector_context=vector_context,
                        kg_context=kg_context
                    )
                },
                {
                    "role": "user",
                    "content": f"用户问题为：{query}"
                }
            ]
        )
        return response.choices[0].message.content
    def mix_rag_response_t(self, query, vector_context, kg_context):
        """
        使用混合RAG方法生成响应
        Args:
            query: 查询文本
            vector_context: 向量上下文
            kg_context: 知识图谱上下文
        Returns:
            生成的响应文本
        """
        # 调用大模型进行混合RAG
        prompt = f"""你是一个专业的比赛赛事问答助手，请根据提供的上下文信息，用简洁准确的语言回答用户的问题。
        
        上下文信息文档：
        {vector_context}
        
        知识图谱实体关系：
        {kg_context}

        用户问题：{query}
        若上下文没有提供相关信息，请明确告知无法回答，不能胡编乱造。
        语言要简介，但是信息一定要完整。
        请直接给出答案，不要包含无关的解释或说明。"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": MIX_RAG_RESPONSE.format(
                        vector_context=vector_context,
                        kg_context=kg_context
                    )
                },
                {
                    "role": "user",
                    "content": f"用户问题为：{query}"
                }
            ]
        )
        return response.choices[0].message.content
if __name__ == "__main__":
    # 初始化LLM工具
    llm_tool = LLMHandler()
    
    # 测试文本
    test_text = """
    这是关于3D编程模型创新设计专项赛的相关信息:\
    #三、选拔赛参与办法1.选拔赛报名。参加活动的青少年通过访问“人工智能创新挑战赛”网站https://aiic.china61.org.cn/，在首页点击“选拔赛报名”进行在线报名，详细登记相关信息和报名赛项、组别。\
    2.参加选拔赛。根据各地区报名实际情况，本赛项选拔赛为线上选拔赛，线上申报参赛作品，并由专家对作品进行盲评的形式举办。3.报名时间:2024年4月15日-5月15日，选拔赛时间为2024年5月16日-7月1日(具体时间另行通知)。\
    参加选拔赛的青少年需通过“人工智能创新挑战赛”网站点击“参加选拔赛”链接，选择“3D编程模型创新设计专项赛”了解选拔赛详细信息。\
    4.主办单位将根据线上选拔赛的成绩，甄选部分优秀选手入围全国挑战赛决赛。5.选拔赛成绩可以在2024年7月15日后，登录“人工智能创新挑战赛”网站进行查询，入围决赛的选手可以参加全国决赛。\
    #四、选拔赛规则#(一)线上选拔赛规则#1.线上选拔赛简介线上选拔赛以线上申报参赛作品，并由专家对作品进行盲评的形式举办。参加活动的青少年需通过“人工智能创新挑战赛”网站点击“参加选拔赛”链接，选择“3D编程模型创新设计专项赛”并凭报名信息进入线上竞赛系统参赛。\
    参赛学生团队使用三维编程设计平台，融合编程语言、计算算法，结合所在学段组别对应参赛命题，将作品可编辑程序源代码文件、作品设计说明文件(PPT格式)等上传至赛事网站进行参赛。#2.选拔赛参赛命题(1)小学低年级组、小学高年级组、初中组参赛命题:运用算法知识和编程逻辑设计三维模型设计作品“伞”。\
    (2)高中组、中职职高组参赛。运用三角函数、阵列、立体几何等算法知识，设计“波浪线形”栅栏设计作品(参考如下图)#3.线上选拔赛流程线上选拔赛分为报名阶段、作品上传阶段、作品评审阶段。具体比赛日期详见后续通知。阶段环节报名阶段参赛学生自行组队，在赛事平台上完成个人信息注册与组队报名。\
    作品上传阶段参赛学生需在该阶段，根据作品设计主题，按照赛事平台要求完成作品可编辑程序源代码文件、作品设计说明文件(PPT格式)的上传。参赛学生提交作品将由专家评审委员会进行作品作品评审阶段评审，并甄选出部分优秀参赛学生入围全国挑战赛。
    """
    
    # 测试实体提取
    print("=== 测试实体提取 ===")
    extraction_result = llm_tool.entity_extraction(test_text)
    print("提取结果:\n", extraction_result)
    
    # 测试继续提取
    print("\n=== 测试继续提取 ===")
    continue_result = llm_tool.entity_continue_extraction(test_text,extraction_result)
    print("继续提取结果:\n", continue_result)
    
    # 测试是否需要继续提取
    print("\n=== 测试是否需要继续提取 ===")
    if_loop_result = llm_tool.entity_if_loop_extraction(test_text,extraction_result,continue_result)
    print("是否需要继续提取:", if_loop_result)
