from fastapi.testclient import TestClient
from langchain_core.documents.base import Document

import main


client = TestClient(main.app)


def auth_headers(username: str) -> dict[str, str]:
    token = main.create_access_token({"sub": username})
    return {"Authorization": f"Bearer {token}"}


class StubChain:
    def __init__(self, output: dict):
        self.output = output
        self.last_payload = None

    def invoke(self, payload: dict) -> dict:
        self.last_payload = payload
        return self.output


def test_login_returns_bearer_token():
    response = client.post(
        "/login",
        json={"username": "admin", "password": "admin123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_rejects_invalid_password():
    response = client.post(
        "/login",
        json={"username": "admin", "password": "wrong-password"},
    )

    assert response.status_code == 401


def test_get_me_returns_profile_for_current_user():
    response = client.get("/me", headers=auth_headers("employee"))

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "employee"
    assert "employee" in body["roles"]
    assert body["can_view_logs"] is False


def test_question_returns_answer_and_writes_audit_log(monkeypatch):
    chain = StubChain({"answer": "公司实行标准考勤制度。"})
    audit_calls = []

    monkeypatch.setattr(main, "get_retrieval_chain", lambda username=None: chain)
    monkeypatch.setattr(
        main.audit_logger,
        "log_query",
        lambda **kwargs: audit_calls.append(kwargs),
    )

    response = client.post(
        "/question",
        headers=auth_headers("employee"),
        json={"input": "公司的考勤制度是什么？", "detailed": False},
    )

    assert response.status_code == 200
    assert response.json() == "公司实行标准考勤制度。"
    assert chain.last_payload["question"] == "公司的考勤制度是什么？"
    assert chain.last_payload["chat_history"] == "无"
    assert "当前问题" not in chain.last_payload["input"]
    assert audit_calls[0]["username"] == "employee"
    assert audit_calls[0]["status"] == "success"


def test_question_returns_sanitized_detailed_output(monkeypatch):
    chain_output = {
        "answer": "报销需要提交单据并审批。",
        "context": [
            Document(
                page_content="关键词：报销\n\n报销流程包括填写申请、主管审批和财务复核。",
                metadata={"filename": "报销流程.txt", "document_type": "报销流程"},
            )
        ],
    }
    chain = StubChain(chain_output)

    monkeypatch.setattr(main, "get_retrieval_chain", lambda username=None: chain)
    monkeypatch.setattr(main.audit_logger, "log_query", lambda **kwargs: None)

    response = client.post(
        "/question",
        headers=auth_headers("admin"),
        json={"input": "报销流程是什么？", "detailed": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == chain_output["answer"]
    assert body["context"][0]["metadata"]["filename"] == "报销流程.txt"
    assert "填写申请" in body["context"][0]["snippet"]


def test_question_rejects_blank_input(monkeypatch):
    called = False

    def fake_chain(username=None):
        nonlocal called
        called = True
        return StubChain({"answer": "unused"})

    monkeypatch.setattr(main, "get_retrieval_chain", fake_chain)

    response = client.post(
        "/question",
        headers=auth_headers("employee"),
        json={"input": "   ", "detailed": False},
    )

    assert response.status_code == 422
    assert called is False


def test_question_returns_503_when_rag_service_is_unavailable(monkeypatch):
    audit_calls = []

    def fail_chain(username=None):
        raise main.RAGServiceError("知识库服务暂时不可用")

    monkeypatch.setattr(main, "get_retrieval_chain", fail_chain)
    monkeypatch.setattr(
        main.audit_logger,
        "log_query",
        lambda **kwargs: audit_calls.append(kwargs),
    )

    response = client.post(
        "/question",
        headers=auth_headers("employee"),
        json={"input": "公司福利有哪些？", "detailed": False},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "知识库服务暂时不可用"
    assert audit_calls[0]["status"] == "failed"


def test_rich_question_response_contains_sources_and_session_history(monkeypatch):
    main.conversation_store._sessions.clear()
    chain = StubChain(
        {
            "answer": "员工每个工作日需要按时打卡。",
            "context": [
                Document(
                    page_content="关键词：考勤\n\n员工应在工作日按规定时间打卡，上下班均需记录。",
                    metadata={"filename": "考勤制度.txt", "document_type": "考勤制度"},
                )
            ],
        }
    )

    monkeypatch.setattr(main, "get_retrieval_chain", lambda username=None: chain)
    monkeypatch.setattr(main.audit_logger, "log_query", lambda **kwargs: None)

    first_response = client.post(
        "/question",
        headers=auth_headers("employee"),
        json={
            "input": "员工需要打卡吗？",
            "return_rich_response": True,
        },
    )

    assert first_response.status_code == 200
    first_body = first_response.json()
    assert first_body["answer"] == "员工每个工作日需要按时打卡。"
    assert first_body["session_id"]
    assert first_body["sources"][0]["filename"] == "考勤制度.txt"
    assert "按规定时间打卡" in first_body["sources"][0]["snippet"]
    assert len(first_body["history"]) == 2

    second_response = client.post(
        "/question",
        headers=auth_headers("employee"),
        json={
            "input": "那迟到会怎样？",
            "return_rich_response": True,
            "session_id": first_body["session_id"],
        },
    )

    assert second_response.status_code == 200
    second_body = second_response.json()
    assert second_body["session_id"] == first_body["session_id"]
    assert len(second_body["history"]) == 4
    assert chain.last_payload["question"] == "那迟到会怎样？"
    assert "员工需要打卡吗？" in chain.last_payload["input"]
    assert "员工每个工作日需要按时打卡。" in chain.last_payload["chat_history"]


def test_get_and_clear_conversation_history(monkeypatch):
    main.conversation_store._sessions.clear()
    chain = StubChain({"answer": "可以。"})

    monkeypatch.setattr(main, "get_retrieval_chain", lambda username=None: chain)
    monkeypatch.setattr(main.audit_logger, "log_query", lambda **kwargs: None)

    ask_response = client.post(
        "/question",
        headers=auth_headers("employee"),
        json={
            "input": "可以请年假吗？",
            "return_rich_response": True,
            "session_id": "demo-session",
        },
    )
    assert ask_response.status_code == 200

    history_response = client.get(
        "/conversation/demo-session",
        headers=auth_headers("employee"),
    )
    assert history_response.status_code == 200
    assert history_response.json()["turns"] == 2

    clear_response = client.delete(
        "/conversation/demo-session",
        headers=auth_headers("employee"),
    )
    assert clear_response.status_code == 200
    assert clear_response.json()["cleared"] is True

    history_after_clear = client.get(
        "/conversation/demo-session",
        headers=auth_headers("employee"),
    )
    assert history_after_clear.status_code == 200
    assert history_after_clear.json()["turns"] == 0


def test_logs_endpoint_requires_log_permission():
    response = client.get("/logs", headers=auth_headers("employee"))

    assert response.status_code == 403


def test_logs_endpoint_returns_filtered_logs_for_admin(monkeypatch):
    captured = {}
    fake_logs = [{"username": "admin", "status": "success"}]

    def fake_get_logs(**kwargs):
        captured.update(kwargs)
        return fake_logs

    monkeypatch.setattr(main.audit_logger, "get_logs", fake_get_logs)

    response = client.get(
        "/logs?limit=20&username=admin&keyword=考勤&status_filter=success",
        headers=auth_headers("admin"),
    )

    assert response.status_code == 200
    assert response.json() == fake_logs
    assert captured == {
        "limit": 20,
        "username": "admin",
        "keyword": "考勤",
        "status_filter": "success",
    }


def test_health_returns_system_status(monkeypatch):
    expected = {
        "status": "ok",
        "documents_count": 5,
        "document_files": ["员工手册.txt"],
        "vector_store_ready": True,
        "embedding_configured": True,
        "chat_configured": True,
        "embedding_base_url": "https://example.com",
        "embedding_model": "text-embedding-v4",
        "warnings": [],
    }
    monkeypatch.setattr(main, "get_system_status", lambda: expected)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == expected
