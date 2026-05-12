import os
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from audit import audit_logger
from auth import rbac
from rag import (
    RAGServiceError,
    get_knowledge_base_snapshot,
    get_retrieval_chain,
    get_system_status,
    rebuild_vector_store,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MAX_CONVERSATION_TURNS = 6
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_QUESTIONS = [
    "公司的考勤制度是怎样的？",
    "报销流程需要经过哪些步骤？",
    "年假和病假的申请规则是什么？",
    "薪酬福利相关内容有哪些注意事项？",
]

app = FastAPI(
    title="基于 RAG 的企业制度问答系统",
    description="毕业设计演示系统",
    version="2.0",
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
security = HTTPBearer()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QuestionRequest(BaseModel):
    input: str = Field(..., min_length=1, description="用户问题")
    detailed: bool = Field(default=False, description="是否返回详细链路结果")
    return_rich_response: bool = Field(default=False, description="是否返回富结构化响应")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    reset_history: bool = Field(default=False, description="提问前是否重置历史")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=64)
    display_name: str = Field(..., min_length=1, max_length=64)


class RegisterResponse(BaseModel):
    username: str
    status: str
    message: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserProfile(BaseModel):
    username: str
    display_name: str
    status: str
    roles: list[str]
    permissions: list[str]
    can_view_logs: bool
    can_manage_knowledge_base: bool
    can_manage_users: bool


class UserReviewRequest(BaseModel):
    action: Literal["approve", "reject", "disable"]
    roles: list[str] = Field(default_factory=list)
    review_note: str = Field(default="", max_length=200)


class UserRoleUpdateRequest(BaseModel):
    roles: list[str] = Field(..., min_length=1)


class UserApprovalResponse(BaseModel):
    username: str
    status: str
    roles: list[str]
    reviewed_at: Optional[str] = None
    approved_by: Optional[str] = None
    review_note: str = ""


class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: str


class SourceItem(BaseModel):
    filename: str
    document_type: str
    snippet: str


class KnowledgeBaseItem(BaseModel):
    filename: str
    relative_path: str
    category: str
    document_type: str
    file_type: str
    required_permission: str
    permission_label: str
    accessible: bool
    size_bytes: int
    updated_at: float


class KnowledgeBaseResponse(BaseModel):
    total_documents: int
    accessible_documents: int
    restricted_documents: int
    supported_types: list[str]
    documents_by_type: dict[str, int]
    documents_by_permission: dict[str, int]
    documents_by_category: dict[str, int]
    allowed_permissions: list[str]
    vector_store_ready: bool
    can_rebuild: bool
    parse_warnings: list[str] = Field(default_factory=list)
    items: list[KnowledgeBaseItem] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: list[ConversationTurn] = Field(default_factory=list)
    turns: int


class ConversationClearResponse(BaseModel):
    session_id: str
    cleared: bool


class RichQuestionResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceItem] = Field(default_factory=list)
    history: list[ConversationTurn] = Field(default_factory=list)
    execution_time: float
    detailed_result: Optional[dict[str, Any]] = None


class ConversationStore:
    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS):
        self.max_turns = max_turns
        self._sessions: dict[str, list[ConversationTurn]] = {}

    def _key(self, username: str, session_id: str) -> str:
        return f"{username}:{session_id}"

    def get_history(self, username: str, session_id: str) -> list[ConversationTurn]:
        key = self._key(username, session_id)
        return [turn.model_copy(deep=True) for turn in self._sessions.get(key, [])]

    def append_exchange(self, username: str, session_id: str, question: str, answer: str) -> list[ConversationTurn]:
        key = self._key(username, session_id)
        history = self.get_history(username, session_id)
        now = datetime.utcnow().isoformat()
        history.extend([
            ConversationTurn(role="user", content=question, timestamp=now),
            ConversationTurn(role="assistant", content=answer, timestamp=now),
        ])
        max_items = self.max_turns * 2
        if len(history) > max_items:
            history = history[-max_items:]
        self._sessions[key] = history
        return self.get_history(username, session_id)

    def clear(self, username: str, session_id: str) -> None:
        self._sessions.pop(self._key(username, session_id), None)


conversation_store = ConversationStore()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _get_client_ip(request: Request) -> Optional[str]:
    return request.client.host if request.client else None


def _build_user_profile(username: str) -> UserProfile:
    user = rbac.get_user(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist")
    permissions = sorted(rbac.get_user_permissions(username))
    roles = rbac.get_user_roles(username)
    return UserProfile(
        username=username,
        display_name=str(user.get("display_name") or username),
        status=str(user.get("status") or "pending"),
        roles=roles,
        permissions=permissions,
        can_view_logs="write_logs" in permissions,
        can_manage_knowledge_base="read_all" in permissions,
        can_manage_users="manage_users" in permissions,
    )


def _require_user_management_access(current_user: TokenData) -> None:
    if not rbac.has_permission(current_user.username, "manage_users"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to manage users",
        )


def _format_admin_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "username": user.get("username"),
        "display_name": user.get("display_name"),
        "status": user.get("status"),
        "roles": user.get("roles", []),
        "created_at": user.get("created_at"),
        "reviewed_at": user.get("reviewed_at"),
        "approved_at": user.get("approved_at"),
        "approved_by": user.get("approved_by"),
        "review_note": user.get("review_note", ""),
    }


def _build_user_status_counts(users: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"pending": 0, "approved": 0, "rejected": 0, "disabled": 0}
    for user in users:
        status_key = str(user.get("status") or "pending")
        counts[status_key] = counts.get(status_key, 0) + 1
    return counts


def _ensure_session_id(session_id: Optional[str]) -> str:
    normalized = (session_id or "").strip()
    return normalized or uuid.uuid4().hex[:12]


def _format_history_for_prompt(history: list[ConversationTurn]) -> str:
    if not history:
        return "无"
    return "\n".join(
        f"{'用户' if turn.role == 'user' else '助手'}：{turn.content}"
        for turn in history[-6:]
    )


def _build_search_query(question: str, history: list[ConversationTurn]) -> str:
    if not history:
        return question
    recent_turns = history[-4:]
    history_text = "\n".join(
        f"{'用户' if turn.role == 'user' else '助手'}：{turn.content}"
        for turn in recent_turns
    )
    return f"{history_text}\n当前问题：{question}"


def _normalize_context_item(item: Any) -> tuple[dict[str, Any], str]:
    if hasattr(item, "metadata") and hasattr(item, "page_content"):
        return dict(getattr(item, "metadata", {}) or {}), str(getattr(item, "page_content", "") or "")
    if isinstance(item, dict):
        metadata = dict(item.get("metadata", {}) or {})
        for key in ("filename", "document_type", "source"):
            if key in item and key not in metadata:
                metadata[key] = item[key]
        content = item.get("page_content") or item.get("content") or ""
        return metadata, str(content)
    return {}, str(item or "")


def _make_snippet(content: str, limit: int = 180) -> str:
    cleaned = (content or "").replace("\r", "").strip()
    if cleaned.startswith("关键词："):
        parts = cleaned.split("\n\n", 1)
        cleaned = parts[1] if len(parts) == 2 else "\n".join(cleaned.splitlines()[1:])
    cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return f"{cleaned[:limit].rstrip()}..." if len(cleaned) > limit else cleaned


def _extract_sources(output: dict[str, Any]) -> list[SourceItem]:
    sources: list[SourceItem] = []
    seen: set[str] = set()
    for item in output.get("context", []) or []:
        metadata, content = _normalize_context_item(item)
        filename = str(metadata.get("filename") or metadata.get("source") or metadata.get("document_type") or "未命名文档")
        if filename in seen:
            continue
        sources.append(
            SourceItem(
                filename=filename,
                document_type=str(metadata.get("document_type") or Path(filename).stem),
                snippet=_make_snippet(content) or "该文档片段未提供可展示内容。",
            )
        )
        seen.add(filename)
    return sources


def _serialize_context(output: dict[str, Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in output.get("context", []) or []:
        metadata, content = _normalize_context_item(item)
        serialized.append({"metadata": metadata, "snippet": _make_snippet(content)})
    return serialized


def _sanitize_detailed_output(output: dict[str, Any]) -> dict[str, Any]:
    detailed_output: dict[str, Any] = {}
    for key, value in output.items():
        if key == "context":
            detailed_output[key] = _serialize_context(output)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            detailed_output[key] = value
        elif isinstance(value, (dict, list)):
            detailed_output[key] = value
        else:
            detailed_output[key] = str(value)
    return detailed_output


def _build_rich_response(
    session_id: str,
    answer: str,
    history: list[ConversationTurn],
    output: dict[str, Any],
    execution_time: float,
    detailed: bool,
) -> RichQuestionResponse:
    return RichQuestionResponse(
        answer=answer,
        session_id=session_id,
        sources=_extract_sources(output),
        history=history,
        execution_time=execution_time,
        detailed_result=_sanitize_detailed_output(output) if detailed else None,
    )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc

    user = rbac.get_user(username)
    if not user:
        raise credentials_exception
    if user.get("status") != "approved":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Current account is not approved")
    return TokenData(username=username)


@app.get("/", summary="认证入口")
def auth_home(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "status_data": get_system_status()},
    )


@app.get("/app", summary="系统首页")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "status_data": get_system_status(), "sample_questions": SAMPLE_QUESTIONS},
    )

@app.post("/login", response_model=Token, summary="用户登录")
def login(login_data: LoginRequest):
    authenticated, message = rbac.authenticate_user(login_data.username, login_data.password)
    if not authenticated:
        status_code = status.HTTP_401_UNAUTHORIZED
        existing_user = rbac.get_user(login_data.username)
        if existing_user and existing_user.get("status") != "approved":
            status_code = status.HTTP_403_FORBIDDEN
        raise HTTPException(status_code=status_code, detail=message, headers={"WWW-Authenticate": "Bearer"})
    access_token = create_access_token(
        data={"sub": login_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register", response_model=RegisterResponse, summary="用户注册")
def register(register_data: RegisterRequest):
    try:
        new_user = rbac.register_user(
            username=register_data.username,
            password=register_data.password,
            display_name=register_data.display_name,
        )
    except ValueError as exc:
        text = str(exc)
        normalized_username = register_data.username.strip()
        conflict = rbac.user_exists(normalized_username)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if conflict else status.HTTP_400_BAD_REQUEST,
            detail=text,
        ) from exc
    return {"username": new_user["username"], "status": new_user["status"], "message": "注册成功，等待管理员审批"}


@app.get("/me", response_model=UserProfile, summary="当前用户信息")
def get_me(current_user: TokenData = Depends(get_current_user)):
    return _build_user_profile(current_user.username)


@app.get("/admin/users", summary="管理员查看用户列表")
def admin_list_users(current_user: TokenData = Depends(get_current_user)):
    _require_user_management_access(current_user)
    users = [_format_admin_user(user) for user in rbac.list_users()]
    return {
        "users": users,
        "status_counts": _build_user_status_counts(users),
        "roles": rbac.list_roles(),
        "permissions": rbac.permissions,
    }


@app.patch("/admin/users/{username}/review", response_model=UserApprovalResponse, summary="管理员审批用户")
def admin_review_user(
    username: str,
    payload: UserReviewRequest,
    current_user: TokenData = Depends(get_current_user),
):
    _require_user_management_access(current_user)
    try:
        reviewed = rbac.review_user(
            username=username,
            actor=current_user.username,
            action=payload.action,
            roles=payload.roles or None,
            review_note=payload.review_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {
        "username": reviewed["username"],
        "status": reviewed["status"],
        "roles": reviewed.get("roles", []),
        "reviewed_at": reviewed.get("reviewed_at"),
        "approved_by": reviewed.get("approved_by"),
        "review_note": reviewed.get("review_note", ""),
    }


@app.patch("/admin/users/{username}/roles", response_model=UserApprovalResponse, summary="管理员更新用户角色")
def admin_update_user_roles(
    username: str,
    payload: UserRoleUpdateRequest,
    current_user: TokenData = Depends(get_current_user),
):
    _require_user_management_access(current_user)
    try:
        updated = rbac.set_user_roles(username=username, roles=payload.roles, actor=current_user.username)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {
        "username": updated["username"],
        "status": updated["status"],
        "roles": updated.get("roles", []),
        "reviewed_at": updated.get("reviewed_at"),
        "approved_by": updated.get("approved_by"),
        "review_note": updated.get("review_note", ""),
    }


@app.get("/conversation/{session_id}", response_model=ConversationHistoryResponse, summary="获取会话历史")
def get_conversation_history(session_id: str, current_user: TokenData = Depends(get_current_user)):
    history = conversation_store.get_history(current_user.username, session_id)
    return {"session_id": session_id, "history": history, "turns": len(history)}


@app.delete("/conversation/{session_id}", response_model=ConversationClearResponse, summary="清空会话历史")
def clear_conversation_history(session_id: str, current_user: TokenData = Depends(get_current_user)):
    conversation_store.clear(current_user.username, session_id)
    return {"session_id": session_id, "cleared": True}


@app.post("/question", summary="知识库问答")
def answer_question(query: QuestionRequest, request: Request, current_user: TokenData = Depends(get_current_user)):
    start_time = time.time()
    normalized_question = query.input.strip()
    client_ip = _get_client_ip(request)

    if not normalized_question:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="问题不能为空")

    session_id = _ensure_session_id(query.session_id)
    if query.reset_history:
        conversation_store.clear(current_user.username, session_id)
    history_before = conversation_store.get_history(current_user.username, session_id)

    try:
        chain = get_retrieval_chain(username=current_user.username)
        output = chain.invoke({
            "input": _build_search_query(normalized_question, history_before),
            "question": normalized_question,
            "chat_history": _format_history_for_prompt(history_before),
        })
        answer = output["answer"]
        execution_time = time.time() - start_time
        history_after = conversation_store.append_exchange(current_user.username, session_id, normalized_question, answer)

        audit_logger.log_query(
            username=current_user.username,
            query=normalized_question,
            response=answer,
            status="success",
            execution_time=execution_time,
            ip_address=client_ip,
        )

        if query.return_rich_response:
            return _build_rich_response(session_id, answer, history_after, output, execution_time, query.detailed)
        if query.detailed:
            return _sanitize_detailed_output(output)
        return answer
    except RAGServiceError as exc:
        execution_time = time.time() - start_time
        audit_logger.log_query(
            username=current_user.username,
            query=normalized_question,
            response=exc.message,
            status="failed",
            execution_time=execution_time,
            ip_address=client_ip,
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message) from exc
    except Exception as exc:
        print(traceback.format_exc())
        execution_time = time.time() - start_time
        audit_logger.log_query(
            username=current_user.username,
            query=normalized_question,
            response=str(exc),
            status="failed",
            execution_time=execution_time,
            ip_address=client_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="系统处理该问题时发生未预期错误，请稍后重试。",
        ) from exc


@app.get("/health", summary="系统健康检查")
def health():
    return get_system_status()


@app.get("/knowledge-base", response_model=KnowledgeBaseResponse, summary="知识库管理视图")
def get_knowledge_base(current_user: TokenData = Depends(get_current_user)):
    snapshot = get_knowledge_base_snapshot(username=current_user.username)
    snapshot["can_rebuild"] = rbac.has_permission(current_user.username, "read_all")
    return snapshot


@app.post("/knowledge-base/rebuild", response_model=KnowledgeBaseResponse, summary="重建知识库索引")
def rebuild_knowledge_base(current_user: TokenData = Depends(get_current_user)):
    if not rbac.has_permission(current_user.username, "read_all"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to rebuild the knowledge base",
        )
    snapshot = rebuild_vector_store()
    snapshot["can_rebuild"] = True
    return snapshot


@app.get("/logs", summary="获取审计日志")
def get_audit_logs(
    limit: int = Query(default=100, ge=1, le=500),
    username: Optional[str] = Query(default=None),
    keyword: Optional[str] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, pattern="^(success|failed)$"),
    current_user: TokenData = Depends(get_current_user),
):
    if not rbac.has_permission(current_user.username, "write_logs"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access audit logs",
        )
    return audit_logger.get_logs(
        limit=limit,
        username=username,
        keyword=keyword,
        status_filter=status_filter,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

