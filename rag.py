import argparse
import os
from os import listdir, path
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENT_KEYWORDS = {
    "员工手册": ["员工手册", "公司制度", "员工权利", "员工义务"],
    "考勤制度": ["考勤", "工作时间", "打卡", "迟到", "早退", "加班"],
    "休假制度": ["休假", "请假", "年假", "病假", "事假"],
    "报销流程": ["报销", "费用", "流程", "审批"],
    "薪酬福利": ["薪酬", "工资", "福利", "奖金", "社保"],
}
VECTOR_INDEX_FILES = ("index.faiss", "index.pkl")
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md"}


class RAGServiceError(RuntimeError):
    """对外暴露的 RAG 服务异常。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _default_embedding_model(base_url: str) -> str:
    """根据接口地址推断默认的 Embedding 模型。"""
    if "dashscope.aliyuncs.com" in base_url:
        return "text-embedding-v4"
    return "text-embedding-v3"


def _vector_index_exists(db_path: str = "vectors") -> bool:
    base_path = Path(db_path)
    return all((base_path / file_name).exists() for file_name in VECTOR_INDEX_FILES)


def _classify_service_error(exc: Exception) -> RAGServiceError:
    error_text = str(exc).lower()
    if "embedding_api_key" in error_text or "embedding_base_url" in error_text:
        return RAGServiceError("Embedding 配置缺失，请检查 .env 中的 EMBEDDING_* 配置。")
    if "openai_api_key" in error_text:
        return RAGServiceError("聊天模型配置缺失，请检查 .env 中的 OPENAI_API_KEY。")
    if "connection error" in error_text or "connecterror" in error_text:
        return RAGServiceError(
            "外部模型服务当前连接失败，请检查网络、代理或 TLS 配置后重试。"
        )
    if "timed out" in error_text or "timeout" in error_text:
        return RAGServiceError("外部模型服务请求超时，请稍后重试。")
    return RAGServiceError(f"知识库服务暂时不可用：{exc}")


def get_embeddings():
    """创建阿里云 DashScope 的 Embedding 客户端。"""
    load_dotenv(override=False)

    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = (os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/")
    model = os.getenv("EMBEDDING_MODEL")

    if not api_key:
        raise ValueError("EMBEDDING_API_KEY 未配置，请检查 .env 文件。")
    if not base_url:
        raise ValueError("EMBEDDING_BASE_URL 未配置，请检查 .env 文件。")
    if not model:
        model = _default_embedding_model(base_url)

    print(f"Using embedding API: {base_url}")
    print(f"Embedding model: {model}")
    print(f"API key set: {api_key is not None}")

    if "dashscope-intl.aliyuncs.com" in base_url:
        print(
            "Warning: you are using the international DashScope endpoint. "
            "If this key was created in Alibaba Cloud Model Studio (China), "
            "switch EMBEDDING_BASE_URL to https://dashscope.aliyuncs.com/compatible-mode/v1 ."
        )

    try:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            check_embedding_ctx_length=False,
        )
        print("Embedding client created successfully")
        return embeddings
    except Exception as exc:
        print(f"Error creating embedding client: {exc}")
        from langchain_core.embeddings import Embeddings
        import numpy as np

        class SimpleEmbeddings(Embeddings):
            """当远程 Embedding 初始化失败时的本地兜底实现。"""

            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]

            def embed_query(self, text):
                vector = np.zeros(100)
                for index, char in enumerate(text[:100]):
                    vector[index] = ord(char) % 100
                return vector.tolist()

        print("Falling back to local embedding")
        return SimpleEmbeddings()


def load_api_key() -> None:
    """从环境变量或 .env 中加载聊天模型和 Embedding 的配置。"""
    load_dotenv(override=False)

    api_key_var = "OPENAI_API_KEY"
    base_url_var = "OPENAI_BASE_URL"

    env_var = os.getenv(api_key_var)
    if env_var and env_var != "":
        os.environ[api_key_var] = env_var
        print("API key taken from environment.")
    else:
        print("Loaded API key from .env file")

    base_url = os.getenv(base_url_var)
    if base_url:
        os.environ[base_url_var] = base_url
        print(f"Using custom base URL: {base_url}")

    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    if embedding_base_url:
        os.environ["EMBEDDING_BASE_URL"] = embedding_base_url
        print(f"Using embedding base URL: {embedding_base_url}")


def load_documents(folder: str = "./documents") -> list[Document]:
    """加载本地制度文档，并补充关键词和权限元数据。"""
    docs = []
    folder_path = Path(folder)
    if not folder_path.exists():
        raise RAGServiceError(f"文档目录不存在：{folder}")

    for item in sorted(listdir(folder)):
        full_path = Path(folder) / item
        if not full_path.is_file() or full_path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
            continue

        loader = TextLoader(path.join(folder, item), encoding="utf-8")
        loaded_docs = loader.load()

        for doc in loaded_docs:
            keywords = []
            for name, values in DOCUMENT_KEYWORDS.items():
                if name in item:
                    keywords = values
                    break

            doc.metadata = {
                "filename": item,
                "document_type": item.split(".")[0],
                "required_permission": (
                    "read_all"
                    if "薪酬福利" in item or "人力资源" in item
                    else "read_employee"
                ),
                "keywords": keywords,
            }
            doc.page_content = "关键词：" + ", ".join(keywords) + "\n\n" + doc.page_content
            docs.append(doc)

    if not docs:
        raise RAGServiceError("documents/ 目录下没有可用于构建知识库的文档。")

    return docs


def populate_vector_db(docs: list[Document], db_path: str = "vectors") -> VectorStore:
    """构建并保存 FAISS 向量库。"""
    print("Loading embedding model...")
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    print(f"Split into {len(documents)} document chunks")
    print("Creating vector database...")
    vector = FAISS.from_documents(documents, embeddings)
    Path(db_path).mkdir(parents=True, exist_ok=True)
    vector.save_local(db_path)
    print(f"Vector database saved to {db_path}")
    return vector


def _permission_filter(username: str) -> Callable[[dict[str, Any]], bool]:
    """为检索器构造元数据权限过滤器，避免重复重建用户专属向量库。"""
    from auth import rbac

    def _matches(metadata: dict[str, Any]) -> bool:
        required_permission = metadata.get("required_permission", "read_employee")
        return rbac.has_permission(username, required_permission)

    return _matches


def get_system_status(db_path: str = "vectors", document_folder: str = "./documents") -> dict[str, Any]:
    """返回系统关键依赖的静态状态，供前端展示和健康检查。"""
    load_dotenv(override=False)

    document_files = []
    documents_path = Path(document_folder)
    if documents_path.exists():
        document_files = [
            file.name
            for file in sorted(documents_path.iterdir())
            if file.is_file() and file.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
        ]

    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL") or (
        _default_embedding_model(embedding_base_url)
        if embedding_base_url
        else None
    )

    warnings = []
    if not _vector_index_exists(db_path):
        warnings.append("向量索引尚未就绪，首次问答可能会触发构建。")
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY 未配置。")
    if not os.getenv("EMBEDDING_API_KEY"):
        warnings.append("EMBEDDING_API_KEY 未配置。")

    return {
        "status": "ok" if not warnings else "degraded",
        "documents_count": len(document_files),
        "document_files": document_files,
        "vector_store_ready": _vector_index_exists(db_path),
        "embedding_configured": bool(os.getenv("EMBEDDING_API_KEY") and embedding_base_url),
        "chat_configured": bool(os.getenv("OPENAI_API_KEY")),
        "embedding_base_url": embedding_base_url,
        "embedding_model": embedding_model,
        "warnings": warnings,
    }


def load_vector_db(db_path: str = "vectors") -> VectorStore:
    """加载已有向量库，如果失败则自动重建。"""
    embeddings = get_embeddings()

    try:
        db = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Vector database loaded successfully")
        return db
    except Exception as exc:
        print(f"Error loading vector database: {exc}")
        print("Regenerating vector database...")

    try:
        docs = load_documents()
        return populate_vector_db(docs, db_path)
    except Exception as rebuild_exc:
        if _vector_index_exists(db_path):
            raise _classify_service_error(rebuild_exc) from rebuild_exc

        raise RAGServiceError(
            "知识库索引不可用，且自动重建失败。请检查 Embedding 配置和网络后执行 "
            "`python rag.py --repopulate`。"
        ) from rebuild_exc


def content_filter(response: str, username: str) -> str:
    """对生成结果做敏感信息过滤。"""
    from auth import rbac

    sensitive_keywords = [
        "薪酬",
        "工资",
        "奖金",
        "福利",
        "绩效",
        "考核",
        "个人信息",
        "隐私",
        "保密",
        "机密",
        "内部资料",
    ]

    if not rbac.has_permission(username, "read_all"):
        for keyword in sensitive_keywords:
            if keyword in response:
                return "根据您的权限，无法查看此信息。"

    return response


def _normalize_chain_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    """补齐问答链所需字段，兼容旧调用方式。"""
    input_text = str(payload.get("input", "") or "").strip()
    question = str(payload.get("question", "") or input_text).strip()
    chat_history = str(payload.get("chat_history", "") or "").strip() or "无"
    return {
        "input": input_text,
        "question": question,
        "chat_history": chat_history,
    }


def get_retrieval_chain(username: Optional[str] = None) -> Runnable:
    """创建带权限过滤和内容审查的 RAG 检索链。"""
    load_api_key()
    if not os.getenv("OPENAI_API_KEY"):
        raise RAGServiceError("OPENAI_API_KEY 未配置，无法调用问答模型。")

    prompt = ChatPromptTemplate.from_template(
        """请严格根据下面提供的上下文回答问题。

如果上下文中没有明确答案，请直接回答“根据当前知识库内容，无法确定该问题的答案”。
不要编造内容，也不要补充上下文之外的信息。

<对话历史>
{chat_history}
</对话历史>

<上下文>
{context}
</上下文>

当前问题：{question}

回答："""
    )

    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model="deepseek-chat",
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    vector = load_vector_db()
    search_kwargs: dict[str, Any] = {"k": 5}
    if username:
        search_kwargs["filter"] = _permission_filter(username)
    retriever = vector.as_retriever(search_kwargs=search_kwargs)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    from langchain_core.runnables import RunnableLambda

    def filter_response(output):
        if "answer" in output and username:
            output["answer"] = content_filter(output["answer"], username)
        return output

    return (
        RunnableLambda(_normalize_chain_inputs)
        | retrieval_chain
        | RunnableLambda(filter_response)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default="vectors")
    parser.add_argument("--document-folder", type=str, default="./documents")
    parser.add_argument("--repopulate", action="store_true", default=False)

    args = parser.parse_args()
    if path.exists(args.db_path) and args.repopulate is False:
        print(
            f"The vector DB path '{args.db_path}' already exists. "
            "Run with option --repopulate to force repopulation."
        )
        exit()

    print("Loading documents...")
    docs = load_documents(args.document_folder)
    print("Populating vector DB...")
    populate_vector_db(docs, args.db_path)
