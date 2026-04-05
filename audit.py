import json
import os
from datetime import datetime
from typing import Optional


class AuditLogger:
    """记录问答请求的审计日志。"""

    def __init__(self, log_file: str = "./audit_logs.json"):
        self.log_file = log_file
        self.ensure_log_file()

    def ensure_log_file(self):
        """确保日志文件存在。"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as file:
                json.dump([], file, ensure_ascii=False, indent=2)

    def _load_logs(self) -> list:
        with open(self.log_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def log_query(
        self,
        username: str,
        query: str,
        response: str,
        status: str,
        execution_time: float,
        ip_address: Optional[str] = None,
    ):
        """写入一条查询日志。"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "query": query,
            "response": response,
            "status": status,
            "execution_time": execution_time,
            "ip_address": ip_address,
        }

        logs = self._load_logs()

        logs.append(log_entry)
        if len(logs) > 1000:
            logs = logs[-1000:]

        with open(self.log_file, "w", encoding="utf-8") as file:
            json.dump(logs, file, ensure_ascii=False, indent=2)

    def get_logs(
        self,
        limit: int = 100,
        username: Optional[str] = None,
        keyword: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list:
        """获取最近的日志记录。"""
        logs = self._load_logs()

        if username:
            logs = [log for log in logs if log.get("username") == username]
        if keyword:
            logs = [
                log
                for log in logs
                if keyword in log.get("query", "") or keyword in log.get("response", "")
            ]
        if status_filter:
            logs = [log for log in logs if log.get("status") == status_filter]

        return logs[-limit:]

    def search_logs(self, keyword: str, limit: int = 50) -> list:
        """按关键字搜索日志。"""
        return self.get_logs(limit=limit, keyword=keyword)

    def clear_logs(self):
        """清空全部审计日志。"""
        with open(self.log_file, "w", encoding="utf-8") as file:
            json.dump([], file, ensure_ascii=False, indent=2)


audit_logger = AuditLogger()
