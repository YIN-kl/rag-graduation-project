import os
from typing import Optional

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag import get_embeddings


def mask_key(key: Optional[str]) -> str:
    """Mask API keys in terminal output."""
    if not key:
        return "<missing>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _require_live_api_tests() -> None:
    if os.getenv("RUN_LIVE_API_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_API_TESTS=1 to run live model API tests.")


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if value:
        return value
    pytest.skip(f"{var_name} is not configured for live API tests.")


def run_chat_api_test() -> None:
    """Test whether the chat model is reachable."""
    print("Testing DeepSeek chat API...")
    print(f"OPENAI_API_KEY: {mask_key(os.getenv('OPENAI_API_KEY'))}")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")

    llm = ChatOpenAI(
        openai_api_key=_require_env("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model="deepseek-chat",
    )
    response = llm.invoke("请用一句中文回复：连接测试成功。")
    print("Chat API test succeeded.")
    print(f"Response: {response.content}")


def run_embedding_api_test() -> None:
    """Test whether the embedding model is reachable."""
    print("\nTesting embedding API...")
    print(f"EMBEDDING_API_KEY: {mask_key(os.getenv('EMBEDDING_API_KEY'))}")
    print(f"EMBEDDING_BASE_URL: {os.getenv('EMBEDDING_BASE_URL')}")
    print(f"EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")

    _require_env("EMBEDDING_API_KEY")
    embeddings = get_embeddings()
    vector = embeddings.embed_query("这是一条用于测试 embedding 的文本。")
    print("Embedding API test succeeded.")
    print(f"Embedding dimension: {len(vector)}")


@pytest.mark.integration
def test_chat_api():
    load_dotenv(override=True)
    _require_live_api_tests()
    run_chat_api_test()


@pytest.mark.integration
def test_embedding_api():
    load_dotenv(override=True)
    _require_live_api_tests()
    run_embedding_api_test()


if __name__ == "__main__":
    load_dotenv(override=True)

    try:
        run_chat_api_test()
    except Exception as exc:
        print(f"Chat API test failed: {exc}")

    try:
        run_embedding_api_test()
    except Exception as exc:
        print(f"Embedding API test failed: {exc}")
