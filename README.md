# 基于 RAG 的企业内部制度问答系统

这是一个基于 FastAPI、LangChain 和 FAISS 构建的企业内部制度问答系统。项目围绕毕业设计场景实现了知识库问答、用户注册与审批、RBAC 权限控制、知识库管理、审计日志、会话管理和前端展示页面，适合用于课程演示、中期检查和毕业答辩。

## 项目亮点

- 支持基于企业制度知识库的 RAG 问答
- 支持 `txt`、`md`、`pdf`、`docx` 文档类型
- 支持递归读取 `documents/` 下的子文件夹
- 支持用户注册、管理员审批和角色管理
- 支持基于用户身份和权限的检索范围限制
- 支持会话上下文保留和多轮对话
- 支持知识库管理页面、文档清单展示和索引重建
- 支持问答审计日志查看与可视化
- 支持来源引用展示和富文本响应
- 支持系统健康检查和状态监控

## 技术栈

- 后端框架：FastAPI
- 问答模型：DeepSeek Chat（OpenAI Compatible API）
- 向量模型：DashScope Embedding
- 检索框架：LangChain
- 向量数据库：FAISS
- 权限认证：JWT + RBAC
- 前端：Jinja2 模板 + 原生 JavaScript

## 当前目录结构

```text
rag-graduation-project/
├─ documents/                    # 知识库文档目录
│  ├─ 行政管理制度/
│  ├─ 财务管理制度/
│  ├─ 人力资源制度/
│  └─ 知识库说明/
├─ static/                       # 前端静态资源
│  └─ app.js
├─ templates/                    # 前端模板
│  ├─ index.html
│  └─ login.html
├─ tests/                        # 自动化测试
│  ├─ test_main_api.py
│  └─ test_rag_documents.py
├─ vectors/                      # FAISS 向量索引
│  ├─ index.faiss
│  └─ index.pkl
├─ main.py                       # FastAPI 应用入口
├─ rag.py                        # 文档加载、索引构建、检索链路
├─ auth.py                       # RBAC 权限控制
├─ audit.py                      # 审计日志逻辑
├─ auth_data.json                # 用户和角色数据
├─ requirements.txt              # 依赖列表
├─ .env.example                  # 环境变量示例
└─ README.md
```

## 默认账号

系统内置了 3 个演示账号：

| 用户名 | 密码 | 角色 | 说明 |
| --- | --- | --- | --- |
| `admin` | `admin123` | 管理员 | 可查看全部文档，可查看日志，可重建知识库，可管理用户 |
| `hr` | `hr123` | 人力资源 | 可查看全部文档，可查看日志，可重建知识库 |
| `employee` | `employee123` | 普通员工 | 仅可检索普通制度文档 |

## 环境要求

- Python 3.9 及以上
- Windows / macOS / Linux
- 可访问 DeepSeek 和 DashScope 接口

## 安装依赖

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

如果只想补装文档解析依赖，也可以执行：

```bash
python -m pip install pypdf docx2txt
```

## 环境变量配置

先复制示例配置：

```bash
cp .env.example .env
```

Windows PowerShell 下也可以手动创建 `.env` 文件，示例内容如下：

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1

EMBEDDING_API_KEY=your_dashscope_api_key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
```

## 知识库文档说明

系统当前支持以下文档格式：

- `txt`
- `md`
- `pdf`
- `docx`

知识库文档统一放在 `documents/` 目录下，支持使用子文件夹按业务主题组织，例如：

- `行政管理制度`
- `财务管理制度`
- `人力资源制度`
- `知识库说明`

### 关于 PDF 和 DOCX

- 如果环境已安装 `pypdf`，系统可以读取 PDF 文档
- 如果环境已安装 `docx2txt`，系统可以读取 DOCX 文档
- 对于提取质量较差的 PDF，项目支持使用同目录的"文本版"文档辅助建立索引

## 构建或重建向量库

新增、删除或修改知识库文档后，建议重建向量索引：

```bash
python rag.py --repopulate
```

这一步很重要。只有重建索引后，新文档才能真正参与问答检索。

## 启动项目

```bash
python main.py
```

启动后可访问：

- 认证入口页面：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- 系统首页：[http://127.0.0.1:8000/app](http://127.0.0.1:8000/app)
- Swagger 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 健康检查：[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## 主要接口

### 用户管理

#### `POST /register`
用户注册，需要管理员审批后才能登录。

请求示例：

```json
{
  "username": "newuser",
  "password": "password123",
  "display_name": "新用户"
}
```

响应示例：

```json
{
  "username": "newuser",
  "status": "pending",
  "message": "注册成功，等待管理员审批"
}
```

#### `POST /login`
用户登录并获取 Bearer Token。

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
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### `GET /me`
返回当前登录用户的角色、权限和功能访问能力。

响应示例：

```json
{
  "username": "admin",
  "display_name": "管理员",
  "status": "approved",
  "roles": ["admin"],
  "permissions": ["read_all", "write_logs", "manage_users"],
  "can_view_logs": true,
  "can_manage_knowledge_base": true,
  "can_manage_users": true
}
```

### 管理员功能

#### `GET /admin/users`
管理员查看用户列表，需要 `manage_users` 权限。

响应示例：

```json
{
  "users": [
    {
      "username": "admin",
      "display_name": "管理员",
      "status": "approved",
      "roles": ["admin"],
      "created_at": "2024-01-01T00:00:00Z",
      "reviewed_at": "2024-01-01T00:00:00Z",
      "approved_by": "system"
    }
  ],
  "status_counts": {
    "pending": 0,
    "approved": 3,
    "rejected": 0,
    "disabled": 0
  },
  "roles": ["admin", "hr", "employee"],
  "permissions": {
    "read_employee": "员工公开制度",
    "read_all": "管理层/HR敏感制度",
    "write_logs": "查看日志",
    "manage_users": "管理用户"
  }
}
```

#### `PATCH /admin/users/{username}/review`
管理员审批用户，需要 `manage_users` 权限。

请求示例：

```json
{
  "action": "approve",
  "roles": ["employee"],
  "review_note": "审批通过"
}
```

#### `PATCH /admin/users/{username}/roles`
管理员更新用户角色，需要 `manage_users` 权限。

请求示例：

```json
{
  "roles": ["hr", "employee"]
}
```

### 问答功能

#### `POST /question`
基于知识库进行问答。

请求示例：

```json
{
  "input": "公司的考勤制度是怎样的？",
  "detailed": false,
  "return_rich_response": true,
  "session_id": "session123",
  "reset_history": false
}
```

参数说明：
- `input`：用户问题（必填）
- `detailed`：是否返回详细链路结果（默认 false）
- `return_rich_response`：是否返回富结构化响应（默认 false）
- `session_id`：会话 ID（可选，系统会自动生成）
- `reset_history`：提问前是否重置历史（默认 false）

响应示例（富文本响应）：

```json
{
  "answer": "根据公司考勤制度，标准工作时间为...",
  "session_id": "session123",
  "sources": [
    {
      "filename": "考勤制度.txt",
      "document_type": "考勤制度",
      "snippet": "标准工作时间：周一至周五 9:00-18:00，午休时间 12:00-13:30..."
    }
  ],
  "history": [
    {
      "role": "user",
      "content": "公司的考勤制度是怎样的？",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "根据公司考勤制度...",
      "timestamp": "2024-01-01T12:00:01Z"
    }
  ],
  "execution_time": 1.234
}
```

### 会话管理

#### `GET /conversation/{session_id}`
获取指定会话的历史记录。

响应示例：

```json
{
  "session_id": "session123",
  "history": [
    {
      "role": "user",
      "content": "公司的考勤制度是怎样的？",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "根据公司考勤制度...",
      "timestamp": "2024-01-01T12:00:01Z"
    }
  ],
  "turns": 2
}
```

#### `DELETE /conversation/{session_id}`
清空指定会话的历史记录。

响应示例：

```json
{
  "session_id": "session123",
  "cleared": true
}
```

### 知识库管理

#### `GET /knowledge-base`
返回知识库管理视图。

响应示例：

```json
{
  "total_documents": 5,
  "accessible_documents": 3,
  "restricted_documents": 2,
  "supported_types": ["txt", "md", "pdf", "docx"],
  "documents_by_type": {
    "考勤制度": 1,
    "休假制度": 1,
    "薪酬福利": 1,
    "报销流程": 1,
    "员工手册": 1
  },
  "documents_by_permission": {
    "read_employee": 3,
    "read_all": 2
  },
  "documents_by_category": {
    "行政管理制度": 2,
    "人力资源制度": 2,
    "财务管理制度": 1
  },
  "allowed_permissions": ["read_employee"],
  "vector_store_ready": true,
  "can_rebuild": false,
  "items": [
    {
      "filename": "考勤制度.txt",
      "relative_path": "行政管理制度/考勤制度.txt",
      "category": "行政管理制度",
      "document_type": "考勤制度",
      "file_type": "txt",
      "required_permission": "read_employee",
      "permission_label": "员工公开制度",
      "accessible": true,
      "size_bytes": 2048,
      "updated_at": 1704067200.0
    }
  ]
}
```

#### `POST /knowledge-base/rebuild`
重建知识库索引。仅管理员和 HR 可使用。

### 审计日志

#### `GET /logs`
查看审计日志。仅管理员和 HR 可访问。

支持查询参数：
- `limit`：限制返回数量（默认 100）
- `username`：按用户名筛选
- `keyword`：按关键词筛选
- `status_filter`：按状态筛选（success/failed）

响应示例：

```json
{
  "logs": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "username": "admin",
      "query": "公司的考勤制度是怎样的？",
      "response": "根据公司考勤制度...",
      "status": "success",
      "execution_time": 1.234,
      "ip_address": "127.0.0.1"
    }
  ],
  "total": 1
}
```

### 系统监控

#### `GET /health`
查看系统健康状态、索引状态与配置状态。

响应示例：

```json
{
  "status": "ok",
  "documents_count": 5,
  "document_files": [
    "行政管理制度/考勤制度.txt",
    "人力资源制度/休假制度.txt",
    "财务管理制度/薪酬福利.txt",
    "财务管理制度/报销流程.txt",
    "知识库说明/员工手册.txt"
  ],
  "documents_by_type": {
    "考勤制度": 1,
    "休假制度": 1,
    "薪酬福利": 1,
    "报销流程": 1,
    "员工手册": 1
  },
  "documents_by_permission": {
    "read_employee": 3,
    "read_all": 2
  },
  "documents_by_category": {
    "行政管理制度": 2,
    "人力资源制度": 2,
    "财务管理制度": 1
  },
  "supported_types": ["docx", "md", "pdf", "txt"],
  "vector_store_ready": true,
  "embedding_configured": true,
  "chat_configured": true,
  "embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "embedding_model": "text-embedding-v4",
  "warnings": []
}
```

## 用户状态说明

系统支持以下用户状态：

- **pending**：待审批（注册后默认状态，无法登录）
- **approved**：已批准（可以正常使用系统）
- **rejected**：已拒绝（无法登录）
- **disabled**：已禁用（无法登录）

## 权限说明

系统支持以下权限：

- **read_employee**：查看员工公开制度
- **read_all**：查看管理层/HR敏感制度
- **write_logs**：查看审计日志
- **manage_users**：管理用户（审批、禁用、修改角色）

## 角色说明

系统支持以下角色：

- **admin**：管理员，拥有所有权限
- **hr**：人力资源，拥有文档查看和日志查看权限
- **employee**：普通员工，仅拥有基本文档查看权限

## 测试

运行测试：

```bash
python -m pytest tests/test_main_api.py tests/test_rag_documents.py
```

## 演示建议

答辩时建议按以下流程演示：

1. 打开认证入口页面，展示系统状态
2. 演示用户注册功能
3. 使用 `admin` 登录，演示用户管理功能（审批新用户、修改角色）
4. 使用 `employee` 登录，演示普通权限下的检索范围
5. 使用 `admin` 或 `hr` 登录，演示更高权限下可访问的知识库内容
6. 提问制度相关问题，展示来源引用和会话历史
7. 演示会话管理功能（获取历史、清空会话）
8. 打开知识库管理面板，展示文档分类和权限分布
9. 打开日志可视化面板，展示审计记录
10. 展示系统健康检查功能

## 常见问题

### 1. 新增文档后系统答不出来

通常是因为没有重建向量库。请执行：

```bash
python rag.py --repopulate
```

### 2. PDF 在知识库里能看到，但问答命中不到

常见原因：

- 没有安装 `pypdf`
- 索引没有重建
- PDF 文本提取质量差

建议：

- 安装依赖：`python -m pip install pypdf`
- 重建索引：`python rag.py --repopulate`
- 为重要 PDF 准备同目录的文本整理版

### 3. 问答时报连接错误

如果报 `Connection error`、`APIConnectionError` 等错误，通常是：

- DeepSeek 接口不可用
- DashScope Embedding 接口不可用
- 网络代理或 TLS 配置异常

### 4. 中文出现乱码

请确认：

- 文档使用 UTF-8 编码
- 终端和编辑器使用 UTF-8 打开

### 5. 注册后无法登录

注册后用户状态为 `pending`，需要管理员审批后才能登录。请使用 `admin` 账号登录，然后在用户管理页面审批新用户。

### 6. 会话历史不显示

请确保在请求中提供了正确的 `session_id`，或者让系统自动生成会话 ID。

## 后续可扩展方向

- 增加知识库上传、删除和编辑能力
- 增加更细粒度的权限控制
- 增加更多企业制度文档样例
- 增加更完整的后台管理能力
- 增加更丰富的答辩展示图表
- 增加用户自助修改密码功能
- 增加邮件通知功能（用户注册、审批等）
- 增加数据导出功能（日志、用户列表等）
- 增加系统监控和告警功能

## 许可证

本项目当前未单独声明开源许可证，如需公开分发，建议补充 LICENSE 文件。
