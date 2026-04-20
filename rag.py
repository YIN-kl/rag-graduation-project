import argparse
import os
from collections import Counter
from os import path
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
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
SENSITIVE_DOCUMENT_HINTS = (
    "薪酬",
    "福利",
    "人力资源",
    "招聘",
    "绩效",
    "员工档案",
    "隐私",
    "保密",
    "合同",
)
PERMISSION_LABELS = {
    "read_employee": "员工公开制度",
    "read_all": "管理层/HR敏感制度",
}


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


def _iter_document_paths(folder_path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in folder_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    )


def _load_text_document_file(file_path: Path) -> list[Document]:
    try:
        return TextLoader(str(file_path), encoding="utf-8").load()
    except UnicodeDecodeError:
        return TextLoader(str(file_path), encoding="gbk").load()


def _find_companion_text_path(file_path: Path) -> Optional[Path]:
    if file_path.suffix.lower() not in {".pdf", ".docx"}:
        return None

    candidates = [
        file_path.with_name(f"{file_path.stem}-文本版.md"),
        file_path.with_name(f"{file_path.stem}-文本版.txt"),
        file_path.with_suffix(".md"),
        file_path.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _is_companion_text_only_file(file_path: Path) -> bool:
    if file_path.suffix.lower() not in {".md", ".txt"}:
        return False
    if not file_path.stem.endswith("-文本版"):
        return False

    base_stem = file_path.stem.removesuffix("-文本版")
    for binary_suffix in (".pdf", ".docx"):
        if file_path.with_name(f"{base_stem}{binary_suffix}").exists():
            return True
    return False


def _text_quality_is_poor(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True

    meaningful_chars = sum(1 for char in normalized if char.isalnum() or "\u4e00" <= char <= "\u9fff")
    question_mark_ratio = normalized.count("?") / max(len(normalized), 1)
    return meaningful_chars < 12 or question_mark_ratio > 0.2


def _guess_keywords(relative_path: str, file_name: str) -> list[str]:
    keywords: list[str] = []
    normalized_path = relative_path.replace("\\", "/")
    stem = Path(file_name).stem
    for name, values in DOCUMENT_KEYWORDS.items():
        if name in normalized_path or any(keyword in normalized_path for keyword in values):
            keywords = values
            break

    if not keywords:
        folder_names = [part for part in Path(normalized_path).parts[:-1] if part]
        keywords = [stem, *folder_names[-2:]]

    return list(dict.fromkeys(keyword for keyword in keywords if keyword))


def _infer_required_permission(relative_path: str) -> str:
    normalized_path = relative_path.replace("\\", "/")
    if any(keyword in normalized_path for keyword in SENSITIVE_DOCUMENT_HINTS):
        return "read_all"
    return "read_employee"


def _load_document_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _load_text_document_file(file_path)

    if suffix == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loaded_docs = PyPDFLoader(str(file_path)).load()
            extracted_text = "\n".join(doc.page_content for doc in loaded_docs)
            if _text_quality_is_poor(extracted_text):
                companion_text_path = _find_companion_text_path(file_path)
                if companion_text_path:
                    companion_docs = _load_text_document_file(companion_text_path)
                    for doc in companion_docs:
                        doc.page_content = (
                            "以下内容来自与 PDF 同目录的文本整理版，用于提升检索质量。\n\n"
                            + doc.page_content
                        )
                    return companion_docs
            return loaded_docs
        except (ImportError, ModuleNotFoundError) as exc:
            raise RAGServiceError(
                "检测到 PDF 文档，但当前环境缺少 `pypdf` 依赖，无法将 PDF 加入知识库。"
            ) from exc

    if suffix == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            return Docx2txtLoader(str(file_path)).load()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RAGServiceError(
                "检测到 DOCX 文档，但当前环境缺少 `docx2txt` 依赖，无法将 DOCX 加入知识库。"
            ) from exc

    return []


def _build_document_metadata(file_path: Path, folder_path: Path) -> dict[str, Any]:
    relative_path = file_path.relative_to(folder_path).as_posix()
    category = file_path.parent.relative_to(folder_path).as_posix() if file_path.parent != folder_path else "根目录"
    return {
        "filename": file_path.name,
        "relative_path": relative_path,
        "category": category,
        "document_type": file_path.stem,
        "file_type": file_path.suffix.lower().lstrip("."),
        "required_permission": _infer_required_permission(relative_path),
    }


def get_knowledge_base_snapshot(
    folder: str = "./documents",
    username: Optional[str] = None,
) -> dict[str, Any]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise RAGServiceError(f"文档目录不存在：{folder}")

    items: list[dict[str, Any]] = []
    parse_warnings: list[str] = []
    permission_filter = _permission_filter(username) if username else None
    type_counter: Counter[str] = Counter()
    permission_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for file_path in _iter_document_paths(folder_path):
        metadata = _build_document_metadata(file_path, folder_path)
        accessible = permission_filter(metadata) if permission_filter else True
        stat = file_path.stat()
        try:
            _load_document_file(file_path)
        except RAGServiceError as exc:
            parse_warnings.append(f"{metadata['relative_path']}: {exc.message}")
        type_counter[metadata["file_type"]] += 1
        permission_counter[metadata["required_permission"]] += 1
        category_counter[metadata["category"]] += 1
        items.append(
            {
                **metadata,
                "accessible": accessible,
                "permission_label": PERMISSION_LABELS.get(
                    metadata["required_permission"],
                    metadata["required_permission"],
                ),
                "size_bytes": stat.st_size,
                "updated_at": stat.st_mtime,
            }
        )

    accessible_items = [item for item in items if item["accessible"]]
    allowed_permissions = (
        sorted(
            {
                item["required_permission"]
                for item in items
                if item["accessible"]
            }
        )
        if username
        else sorted(permission_counter.keys())
    )

    return {
        "total_documents": len(items),
        "accessible_documents": len(accessible_items),
        "restricted_documents": max(len(items) - len(accessible_items), 0),
        "supported_types": sorted(file_type.lstrip(".") for file_type in SUPPORTED_DOCUMENT_SUFFIXES),
        "documents_by_type": dict(sorted(type_counter.items())),
        "documents_by_permission": dict(sorted(permission_counter.items())),
        "documents_by_category": dict(sorted(category_counter.items())),
        "allowed_permissions": allowed_permissions,
        "vector_store_ready": _vector_index_exists(),
        "parse_warnings": parse_warnings,
        "items": items,
    }


def rebuild_vector_store(
    db_path: str = "vectors",
    document_folder: str = "./documents",
) -> dict[str, Any]:
    docs = load_documents(document_folder)
    populate_vector_db(docs, db_path)
    snapshot = get_knowledge_base_snapshot(document_folder)
    snapshot["indexed_chunks"] = len(docs)
    return snapshot


def load_documents(folder: str = "./documents") -> list[Document]:
    """加载本地制度文档，并补充关键词和权限元数据。"""
    docs = []
    folder_path = Path(folder)
    if not folder_path.exists():
        raise RAGServiceError(f"文档目录不存在：{folder}")

    for full_path in _iter_document_paths(folder_path):
        if _is_companion_text_only_file(full_path):
            print(f"Skipping companion text file: {full_path.relative_to(folder_path).as_posix()}")
            continue

        relative_path = full_path.relative_to(folder_path).as_posix()
        loaded_docs = _load_document_file(full_path)

        for doc in loaded_docs:
            metadata = _build_document_metadata(full_path, folder_path)
            keywords = _guess_keywords(relative_path, full_path.name)

            print(f"Loading document: {relative_path}")
            print(f"Keywords: {keywords}")

            doc.metadata = {
                **metadata,
                "keywords": keywords,
            }
            doc.page_content = (
                "关键词：" + ", ".join(keywords)
                + f"\n目录：{metadata['category']}"
                + f"\n文件类型：{metadata['file_type']}"
                + "\n\n"
                + doc.page_content
            )
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
        # 如果用户有 read_all 权限，直接返回 True
        if rbac.has_permission(username, "read_all"):
            return True
        
        required_permission = metadata.get("required_permission", "read_employee")
        has_perm = rbac.has_permission(username, required_permission)
        print(f"Filter check: username={username}, required_permission={required_permission}, has_perm={has_perm}")
        return has_perm

    return _matches


def get_system_status(db_path: str = "vectors", document_folder: str = "./documents") -> dict[str, Any]:
    """返回系统关键依赖的静态状态，供前端展示和健康检查。"""
    load_dotenv(override=False)

    document_files: list[str] = []
    snapshot: dict[str, Any] = {
        "documents_by_type": {},
        "documents_by_permission": {},
        "documents_by_category": {},
        "supported_types": sorted(file_type.lstrip(".") for file_type in SUPPORTED_DOCUMENT_SUFFIXES),
    }
    documents_path = Path(document_folder)
    if documents_path.exists():
        snapshot = get_knowledge_base_snapshot(document_folder)
        document_files = [item["relative_path"] for item in snapshot["items"]]

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
    warnings.extend(snapshot.get("parse_warnings", []))

    return {
        "status": "ok" if not warnings else "degraded",
        "documents_count": len(document_files),
        "document_files": document_files,
        "documents_by_type": snapshot["documents_by_type"],
        "documents_by_permission": snapshot["documents_by_permission"],
        "documents_by_category": snapshot["documents_by_category"],
        "supported_types": snapshot["supported_types"],
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
        """请根据下面提供的上下文回答问题。

上下文：
{context}

问题：{question}

回答："""
    )

    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model="deepseek-chat",
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    vector = load_vector_db()
    
    # 创建一个自定义的检索函数，包含权限过滤
    def retrieve_with_permission(query):
        # 检查 query 是否是字典，如果是，提取 question 或 input 字段
        if isinstance(query, dict):
            actual_query = query.get('question', '') or query.get('input', '')
            print(f"Actual query: {actual_query}")
            query = actual_query
        # 获取所有相关文档
        docs = vector.similarity_search(query, k=10)
        print(f"Found {len(docs)} documents for query: {query}")
        # 打印前3个文档的内容，用于调试
        for i, doc in enumerate(docs[:3]):
            print(f"Document {i+1} metadata: {doc.metadata}")
            print(f"Document {i+1} content: {doc.page_content[:100]}...")
        # 应用权限过滤
        if username:
            filter_func = _permission_filter(username)
            filtered_docs = [doc for doc in docs if filter_func(doc.metadata)]
            print(f"Filtered docs: {len(filtered_docs)} out of {len(docs)}")
            return filtered_docs[:5]  # 只返回前5个
        return docs[:5]
    
    # 创建一个简单的检索器
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda
    
    class SimpleRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):
            # 检查 query 是否是字典，如果是，提取 question 或 input 字段
            if isinstance(query, dict):
                actual_query = query.get('question', '') or query.get('input', '')
                print(f"Retrieving docs for actual query: {actual_query}")
                return retrieve_with_permission(actual_query)
            # 否则直接使用 query 作为检索字符串
            print(f"Retrieving docs for query: {query}")
            return retrieve_with_permission(query)
    
    # 使用自定义检索器
    retriever = SimpleRetriever()
    
    # 创建检索链
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 添加一个中间步骤来查看 document_chain 的输出
    def debug_output(output):
        print(f"Document chain output: {output}")
        return output
    
    retrieval_chain = retrieval_chain | RunnableLambda(debug_output)
    
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
