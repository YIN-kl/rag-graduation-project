import os
import time
from datetime import datetime, timedelta
from typing import Optional

# 设置环境变量解决OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from rag import get_retrieval_chain
from auth import rbac
from audit import audit_logger

# JWT配置
SECRET_KEY = "your-secret-key"  # 在生产环境中应该使用更安全的密钥
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="基于RAG的企业内部制度问答系统",
    description="尹雯清202231060411",
    version="1.0"
)

security = HTTPBearer()


class Query(BaseModel):
    input: str
    detailed: bool  # If True, return whole output rather than just the answer


class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    创建访问令牌
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    获取当前用户
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data


@app.post("/login", response_model=Token)
def login(login_data: LoginRequest):
    """
    用户登录接口
    """
    if rbac.authenticate(login_data.username, login_data.password):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": login_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.post("/question", summary="知识库问答接口")
def answer_question(query: Query, current_user: TokenData = Depends(get_current_user)):
    """
    知识库问答接口，需要用户认证
    """
    start_time = time.time()
    try:
        # 创建带有用户权限的检索链
        chain = get_retrieval_chain(username=current_user.username)
        output = chain.invoke({"input": query.input})
        
        # 记录审计日志
        execution_time = time.time() - start_time
        audit_logger.log_query(
            username=current_user.username,
            query=query.input,
            response=output["answer"],
            status="success",
            execution_time=execution_time
        )
        
        if query.detailed:
            return output
        else:
            return output["answer"]
    except Exception as e:
        # 记录详细错误日志
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in answer_question: {error_traceback}")
        
        # 记录错误日志
        execution_time = time.time() - start_time
        audit_logger.log_query(
            username=current_user.username,
            query=query.input,
            response=str(e),
            status="failed",
            execution_time=execution_time
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>智能问答系统</title>
    </head>
    <body>
        <h1>📚 基于 RAG 的智能问答系统</h1>
        <p>毕业设计演示系统</p>
        <p>接口测试地址：</p>
        <a href="/docs">进入接口测试页面</a>
        <p>使用说明：</p>
        <ul>
            <li>1. 首先使用 /login 接口获取访问令牌</li>
            <li>2. 在 /question 接口中使用 Bearer token 进行认证</li>
            <li>3. 发送问题进行智能问答</li>
        </ul>
    </body>
    </html>
    """


@app.get("/logs", summary="获取审计日志")
def get_audit_logs(limit: int = 100, current_user: TokenData = Depends(get_current_user)):
    """
    获取审计日志，需要管理员权限
    """
    if not rbac.has_permission(current_user.username, "write_logs"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access audit logs"
        )
    return audit_logger.get_logs(limit=limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)