# 基于 RAG 的企业内部制度问答系统

这是一个基于 FastAPI、LangChain 和 FAISS 构建的企业内部制度问答系统。项目围绕毕业设计场景实现了知识库问答、RBAC 权限控制、知识库管理、审计日志和前端展示页面，适合用于课程演示、中期检查和毕业答辩。

## 项目亮点

- 支持基于企业制度知识库的 RAG 问答
- 支持 `txt`、`md`、`pdf`、`docx` 文档类型
- 支持递归读取 `documents/` 下的子文件夹
- 支持基于用户身份和权限的检索范围限制
- 支持知识库管理页面、文档清单展示和索引重建
- 支持问答审计日志查看与可视化
- 支持会话上下文保留与来源引用展示

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
├─ templates/                    # 前端模板
├─ tests/                        # 自动化测试
├─ vectors/                      # FAISS 向量索引
├─ main.py                       # FastAPI 应用入口
├─ rag.py                        # 文档加载、索引构建、检索链路
├─ auth.py                       # RBAC 权限控制
├─ audit.py                      # 审计日志逻辑
├─ requirements.txt              # 依赖列表
├─ .env.example                  # 环境变量示例
└─ README.md
```

## 默认账号

系统内置了 3 个演示账号：

| 用户名 | 密码 | 角色 | 说明 |
| --- | --- | --- | --- |
| `admin` | `admin123` | 管理员 | 可查看全部文档，可查看日志，可重建知识库 |
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
- 对于提取质量较差的 PDF，项目支持使用同目录的“文本版”文档辅助建立索引

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

- 首页演示页面：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 健康检查：[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## 主要接口

### `POST /login`

用户登录并获取 Bearer Token。

请求示例：

```json
{
  "username": "admin",
  "password": "admin123"
}
```

### `GET /me`

返回当前登录用户的角色、权限和功能访问能力。

### `POST /question`

基于知识库进行问答。

请求示例：

```json
{
  "input": "公司的考勤制度是怎样的？",
  "detailed": false,
  "return_rich_response": true
}
```

### `GET /knowledge-base`

返回知识库管理视图，包括：

- 文档总数
- 可访问文档数
- 受限文档数
- 文档类型分布
- 目录分布
- 文档清单

### `POST /knowledge-base/rebuild`

重建知识库索引。仅管理员和 HR 可使用。

### `GET /logs`

查看审计日志。仅管理员和 HR 可访问。

### `GET /health`

查看系统健康状态、索引状态与配置状态。

## 测试

运行测试：

```bash
python -m pytest tests/test_main_api.py tests/test_rag_documents.py
```

## 演示建议

答辩时建议按以下流程演示：

1. 打开首页，展示系统状态和知识库管理面板
2. 使用 `employee` 登录，演示普通权限下的检索范围
3. 使用 `admin` 或 `hr` 登录，演示更高权限下可访问的知识库内容
4. 提问制度相关问题，展示来源引用和会话历史
5. 打开日志可视化面板，展示审计记录

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

- 安装依赖
- 重建索引
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

## 后续可扩展方向

- 增加知识库上传、删除和编辑能力
- 增加更细粒度的权限控制
- 增加更多企业制度文档样例
- 增加更完整的后台管理能力
- 增加更丰富的答辩展示图表
