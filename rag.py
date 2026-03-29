import argparse
import os
from os import listdir, path

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


def _default_embedding_model(base_url: str) -> str:
    if "dashscope.aliyuncs.com" in base_url:
        return "text-embedding-v4"
    return "text-embedding-v3"


def get_embeddings():
    """
    Use Alibaba Cloud DashScope embeddings via the OpenAI-compatible API.
    """
    load_dotenv(override=False)

    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = (os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/")
    model = os.getenv("EMBEDDING_MODEL")

    if not api_key:
        raise ValueError("EMBEDDING_API_KEY is not set. Please check your .env file.")
    if not base_url:
        raise ValueError("EMBEDDING_BASE_URL is not set. Please check your .env file.")
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
    """
    Load chat and embedding API settings from environment variables or .env.
    """
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
    docs = []
    for item in listdir(folder):
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
    return docs


def populate_vector_db(docs: list[Document], db_path: str = "vectors") -> VectorStore:
    """
    Build the vector database.
    """
    print("Loading embedding model...")
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    print(f"Split into {len(documents)} document chunks")
    print("Creating vector database...")
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(db_path)
    print(f"Vector database saved to {db_path}")
    return vector


def filter_documents_by_permission(docs: list[Document], username: str) -> list[Document]:
    """
    Filter documents according to the user's permissions.
    """
    from auth import rbac

    filtered_docs = []
    for doc in docs:
        required_permission = doc.metadata.get("required_permission", "read_employee")
        if rbac.has_permission(username, required_permission):
            filtered_docs.append(doc)
    return filtered_docs


def load_vector_db(db_path: str = "vectors", username: str = None) -> VectorStore:
    """
    Load the vector database and rebuild it if needed.
    """
    embeddings = get_embeddings()

    try:
        db = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Vector database loaded successfully")
    except Exception as exc:
        print(f"Error loading vector database: {exc}")
        print("Regenerating vector database...")
        docs = load_documents()
        db = populate_vector_db(docs, db_path)

    if username:
        try:
            docs = []
            for doc_id in db.index_to_docstore_id.values():
                try:
                    doc = db.docstore.search(doc_id)
                    if doc:
                        docs.append(doc)
                except Exception:
                    pass

            filtered_docs = filter_documents_by_permission(docs, username)
            if filtered_docs:
                db = FAISS.from_documents(filtered_docs, embeddings)
                print(f"Vector database filtered for user: {username}")
        except Exception as exc:
            print(f"Error during permission filtering: {exc}")

    return db


def content_filter(response: str, username: str) -> str:
    """
    Filter generated content based on the user's permissions.
    """
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


def get_retrieval_chain(username: str = None) -> Runnable:
    """
    Create the RAG retrieval chain.
    """
    load_api_key()
    prompt = ChatPromptTemplate.from_template(
        """请根据以下上下文回答问题，确保答案完全基于上下文内容，不要添加任何外部信息。

        <上下文>
        {context}
        </上下文>

        问题: {input}

        回答:"""
    )
    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model="deepseek-chat",
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    vector = load_vector_db(username=username)
    retriever = vector.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    from langchain_core.runnables import RunnableLambda

    def filter_response(output):
        if "answer" in output:
            output["answer"] = content_filter(output["answer"], username)
        return output

    return retrieval_chain | RunnableLambda(filter_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default="vectors")
    parser.add_argument("--document-folder", type=str, default="./documents")
    parser.add_argument("--repopulate", action="store_true", default=False)

    args = parser.parse_args()
    if path.exists(args.db_path) and args.repopulate is False:
        print(
            f"The vector DB path '{args.db_path}' already exists. Run with option --repopulate to force repopulation."
        )
        exit()

    print("Loading documents...")
    docs = load_documents(args.document_folder)
    print("Populating vector DB...")
    populate_vector_db(docs, args.db_path)
