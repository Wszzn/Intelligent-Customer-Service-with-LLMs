# 知识图谱构建与更新系统

这是一个基于PDF文档的知识图谱构建和更新系统，支持自动处理PDF文件、构建知识图谱、向量检索和增量更新等功能。

## 系统架构

系统主要由以下几个核心组件构成：

### 1. 文档处理模块
- `process_pdfs.py`: PDF文件处理模块，负责将PDF转换为可处理的文本格式
- `chunk_processor.py`: 文本分块处理模块，将长文本分割成适合处理的块

### 2. 知识图谱构建模块
- `kg_builder.py`: 知识图谱构建器，负责实体和关系的提取、存储和检索
- `graph_matcher.py`: 图匹配模块，用于知识图谱的匹配和查询

### 3. 向量处理模块
- `embedding_handler.py`: 向量处理模块，负责文本的向量化
- `retriever.py`: 向量检索模块，支持相似度搜索
- `reranker.py`: 重排序模块，优化检索结果

### 4. 大语言模型接口
- `LLM_tool.py`: 大语言模型工具类
- `LLMhandler.py`: 异步大语言模型处理接口
- `prompts.py`: 提示词模板管理

### 5. 系统管理模块
- `prepare_system.py`: 系统初始化模块
- `knowledge_update.py`: 知识库更新模块
- `main.py`: 主程序入口

## 功能特性

1. **PDF文档处理**
   - 支持批量PDF文件处理
   - 自动文本提取和分块
   - 增量更新支持

2. **知识图谱构建**
   - 实体和关系自动提取
   - Neo4j图数据库存储
   - 向量化索引支持

3. **智能检索**
   - 基于向量的语义搜索
   - 知识图谱关系查询
   - 结果重排序优化

4. **增量更新**
   - 文件变更检测
   - 自动知识图谱更新
   - 向量索引维护

## 安装部署

### 环境要求
- Python 3.10+
- Neo4j数据库
- 必要的Python包（requirements.txt）

### 安装步骤
1. 克隆项目
```bash
git clone [项目地址]
cd [项目目录]
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

4. 初始化系统
```bash
python prepare_system.py
```

## 使用说明

### 1. 系统初始化
```python
from prepare_system import SystemPreparer

preparer = SystemPreparer()
await preparer.run_full_preparation()
```

### 2. 知识库更新
```python
from knowledge_update import KnowledgeBaseUpdater

updater = KnowledgeBaseUpdater()
await updater.update_knowledge_base()
```

### 3. 知识检索
```python
from retriever import Retriever

retriever = Retriever()
results = await retriever.search("查询内容")
```

## 目录结构
```python
.
├── data/                 # 数据存储
│   ├── embeddings/       # 向量索引（FAISS格式）
│   ├── pdfs/             # 原始PDF文件
│   └── update_log.json   # 更新记录
├── docs/                 # 文档
├── src/
│   ├── kg_builder.py     # 知识图谱构建核心
│   ├── retriever.py      # 混合检索模块
│   ├── reranker.py       # 重排序模块
│   └── LLMhandler.py     # 大模型接口
└── requirements.txt      # 依赖清单
```

## 目录结构
```python
graph TD
    A[PDF文件] --> B{process_pdfs.py}
    B --> C[Markdown文本]
    C --> D{chunk_processor.py}
    D --> E[文本分块]
    E --> F[kg_builder.py]
    F --> G[Neo4j知识图谱]
    F --> H[FAISS向量索引]
    G --> I[graph_matcher.py]
    H --> J[retriever.py]
    J --> K[reranker.py]
    K --> L[最终结果]
```