# 基于 RAG 的企业内部制度问答系统

这是一个基于 FastAPI、LangChain 和 FAISS 构建的企业内部知识问答系统。项目以企业制度文档为知识库，通过 RAG 检索增强生成能力，为用户提供带权限控制的智能问答服务。

系统当前集成了：
- DeepSeek 聊天模型，用于生成最终回答
- 阿里云 DashScope Embedding，用于向量化检索
- FAISS 向量数据库，用于本地相似度搜索
- 基于角色的权限控制，用于限制敏感制度内容访问
- 审计日志，用于记录查询行为和接口执行结果

## 项目功能

- 支持本地制度文档导入与分块
- 支持基于向量检索的 RAG 问答
- 支持用户登录与 Token 鉴权
- 支持基于角色的访问控制
- 支持问答审计日志记录
- 支持通过 FastAPI 自动生成接口文档

## 技术栈

- 后端框架：FastAPI
- LLM：DeepSeek
- Embedding：阿里云 DashScope OpenAI 兼容接口
- 检索框架：LangChain
- 向量数据库：FAISS
- 鉴权：JWT
- 配置管理：python-dotenv

## 项目结构

```text
rag-graduation-project/
├── documents/               # 企业制度文档
├── vectors/                 # FAISS 向量库
├── main.py                  # FastAPI 应用入口
├── rag.py                   # RAG 检索与向量库逻辑
├── auth.py                  # 角色权限控制
├── audit.py                 # 审计日志
├── auth_data.json           # 默认用户与角色数据
├── test_api.py              # LLM 连接测试脚本
├── requirements.txt         # 依赖列表
├── .env.example             # 环境变量示例
└── README.md
```

## 默认账号

项目内置了 3 个演示账号：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| `admin` | `admin123` | 管理员 |
| `hr` | `hr123` | 人力资源 |
| `employee` | `employee123` | 普通员工 |

不同角色可查看的内容不同：
- `admin`：可访问全部文档，可查看日志
- `hr`：可访问全部文档，可查看日志
- `employee`：仅可访问普通员工权限范围内的内容

## 环境要求

- Python 3.9 及以上
- Windows、macOS 或 Linux
- 可正常访问 DeepSeek 和阿里云 DashScope 接口

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 环境变量配置

先复制示例配置：

```bash
cp .env.example .env
```

Windows PowerShell 可手动新建 `.env` 文件，内容参考如下：

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1

EMBEDDING_API_KEY=your_dashscope_api_key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
```

### 配置说明

- `OPENAI_API_KEY`
  DeepSeek 的 API Key
- `OPENAI_BASE_URL`
  DeepSeek OpenAI 兼容接口地址，默认使用 `https://api.deepseek.com/v1`
- `EMBEDDING_API_KEY`
  阿里云 DashScope 的 API Key
- `EMBEDDING_BASE_URL`
  阿里云 Embedding 接口地址
- `EMBEDDING_MODEL`
  Embedding 模型名称，当前推荐 `text-embedding-v4`

### 阿里云 Embedding 注意事项

- 不需要单独申请“Embedding 专用 API Key”，阿里云 Model Studio 的 API Key 可以直接用于 Embedding
- 中国区通常使用：
  `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 国际站通常使用：
  `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- `API Key`、`地区`、`Base URL`、`模型名` 必须匹配

## 准备知识库文档

把企业制度文档放到 `documents/` 目录下，推荐使用 UTF-8 编码的 `.txt` 文件。

当前项目内已包含示例文档：
- `员工手册.txt`
- `考勤制度.txt`
- `休假制度.txt`
- `报销流程.txt`
- `薪酬福利.txt`

## 构建向量库

首次运行前，建议先生成向量库：

```bash
python rag.py --repopulate
```

说明：
- `--repopulate` 会强制重建 `vectors/` 下的索引
- 当你修改了文档内容或更换了 Embedding 模型后，也建议重新生成向量库

## 启动项目

```bash
python main.py
```

启动后默认访问地址：

- 首页：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 接口说明

### 1. 登录接口

- 路径：`POST /login`
- 作用：获取访问令牌

请求示例：

```json
{
  "username": "admin",
  "password": "admin123"
}
```

响应示例：

```json
{
  "access_token": "your_token",
  "token_type": "bearer"
}
```

### 2. 问答接口

- 路径：`POST /question`
- 作用：基于知识库进行问答
- 鉴权：需要在请求头中携带 `Bearer Token`

请求体：

```json
{
  "input": "公司的考勤制度是什么？",
  "detailed": false
}
```

说明：
- `input`：用户问题
- `detailed=false`：仅返回最终回答
- `detailed=true`：返回完整链路结果

### 3. 审计日志接口

- 路径：`GET /logs`
- 作用：查看审计日志
- 鉴权：需要登录
- 权限：仅 `admin` 和 `hr` 可访问

## 调用示例

### PowerShell

先登录：

```powershell
$loginBody = @{
  username = "admin"
  password = "admin123"
} | ConvertTo-Json

$token = (Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/login `
  -ContentType "application/json" `
  -Body $loginBody).access_token
```

再提问：

```powershell
$headers = @{
  Authorization = "Bearer $token"
}

$body = @{
  input = "公司的考勤制度是什么？"
  detailed = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/question `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

### curl

```bash
curl -X POST "http://127.0.0.1:8000/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"admin\",\"password\":\"admin123\"}"
```

拿到 token 后：

```bash
curl -X POST "http://127.0.0.1:8000/question" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d "{\"input\":\"公司的考勤制度是什么？\",\"detailed\":false}"
```

## 审计日志

系统会将问答请求记录到本地日志文件中：

- 文件：`audit_logs.json`

日志字段包括：
- 请求时间
- 用户名
- 问题内容
- 响应内容
- 执行状态
- 执行耗时
- IP 地址

## 常见问题

### 1. 阿里云提示 API Key 错误

请重点检查：
- 是否使用了正确地区的 `EMBEDDING_BASE_URL`
- 是否使用了正确的模型名，如 `text-embedding-v4`
- 是否重启了服务，让新的 `.env` 生效
- 是否把中国区 Key 配到了国际站地址，或反过来

### 2. 连接阿里云时报 `Connection error`

这通常不是 Key 错误，而是网络问题。常见原因包括：
- 当前网络无法访问阿里云接口
- 代理配置异常
- TLS / SSL 握手失败
- 公司网络策略限制了外部请求

### 3. 向量库加载失败

如果 `vectors/` 中的索引和当前 Embedding 配置不一致，建议执行：

```bash
python rag.py --repopulate
```

### 4. 中文显示乱码

请确认：
- 文档文件使用 UTF-8 编码
- 终端或编辑器使用 UTF-8 编码打开

## Docker 运行

构建镜像：

```bash
docker build -t rag-app .
```

启动容器：

```bash
docker run -p 8000:8000 --env-file .env rag-app
```

## 后续可扩展方向

- 支持 PDF、Word 等更多文档格式
- 支持更细粒度的权限控制
- 支持管理后台
- 支持前端问答页面
- 支持多知识库切换
- 支持日志筛选与导出

## 许可证

本项目当前未单独声明开源许可证，如需公开分发，建议补充 LICENSE 文件。
