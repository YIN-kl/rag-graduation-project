import os
from rag import load_vector_db

# 加载向量数据库
vector = load_vector_db()

# 测试查询
query = "公司的考勤制度是怎样的？"
print(f"Testing query: {query}")

# 检索文档
docs = vector.similarity_search(query, k=10)
print(f"Found {len(docs)} documents for query: {query}")

# 打印所有文档的信息
for i, doc in enumerate(docs):
    print(f"Document {i+1} metadata: {doc.metadata}")
    print(f"Document {i+1} content: {doc.page_content[:200]}...")
    print()