# 基于RAG的企业内部制度问答系统
这是一个使用FastAPI和LangChain构建的企业内部制度智能问答系统，支持基于文档的问答功能。

## 项目特点
- 使用FastAPI构建RESTful API
- 基于LangChain实现RAG（检索增强生成）
- 使用FAISS进行向量相似性搜索
- 支持多种LLM模型（默认使用DeepSeek）
- 支持企业内部制度文档的智能问答

## 本地运行

### 1. 配置API密钥
最简单的方法是使用`.env`文件。在项目根目录创建`.env`文件，配置以下环境变量：

```
# DeepSeek API配置
OPENAI_API_KEY=你的DeepSeek API密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1

# 阿里云通义千问Embedding配置（可选）
EMBEDDING_API_KEY=你的阿里云API密钥
EMBEDDING_BASE_URL=https://ark.cn-beijing.aliyuncs.com/api/v3
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 准备企业制度文档
将企业制度文档放在`./documents`目录下，支持`.txt`格式的文档。目前系统已包含以下示例文档：
- 员工手册.txt
- 考勤制度.txt
- 休假制度.txt
- 报销流程.txt
- 薪酬福利.txt

### 4. 生成向量数据库
运行以下命令生成向量数据库：
```bash
python rag.py --repopulate
```

### 5. 启动API服务
```bash
python main.py
```

### 6. 测试API
使用以下命令测试API：
```bash
# PowerShell
Invoke-WebRequest -Uri http://localhost:8000/question -Method POST -ContentType "application/json" -Body '{"input": "员工手册包含哪些内容？", "detailed": false}'

# curl
curl -X POST http://localhost:8000/question -H "Content-Type: application/json" -d '{"input": "员工手册包含哪些内容？", "detailed": false}'
```

## API接口

### 1. 问答接口
- **URL**: `/question`
- **方法**: POST
- **请求体**:
  ```json
  {
    "input": "你的问题",
    "detailed": false
  }
  ```
- **响应**:
  - `detailed=false`: 只返回回答内容
  - `detailed=true`: 返回完整的回答和检索到的文档

### 2. 首页
- **URL**: `/`
- **方法**: GET
- **响应**: 返回系统首页，包含API测试页面链接

## Docker部署

### 构建镜像
```bash
docker build . -t rag
```

### 运行容器
```bash
docker run -p 8000:8000 rag
```

## 技术栈
- **后端框架**: FastAPI
- **RAG框架**: LangChain
- **向量数据库**: FAISS
- **LLM模型**: DeepSeek
- **Embedding**: 支持多种Embedding模型

## 注意事项
- 确保API密钥有足够的余额
- 文档内容应清晰、结构化，以获得更好的问答效果
- 对于大型文档，建议适当分割以提高检索效率

## 示例问题
- 员工手册包含哪些内容？
- 考勤制度有哪些规定？
- 如何申请休假？
- 报销流程是怎样的？
- 公司的薪酬福利有哪些？
