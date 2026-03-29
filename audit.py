import os
import json
import time
from datetime import datetime
from typing import Dict, Optional

class AuditLogger:
    """
    审计日志记录器
    """
    
    def __init__(self, log_file: str = "./audit_logs.json"):
        """
        初始化审计日志记录器
        :param log_file: 日志文件路径
        """
        self.log_file = log_file
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """
        确保日志文件存在
        """
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def log_query(self, username: str, query: str, response: str, status: str, 
                 execution_time: float, ip_address: Optional[str] = None):
        """
        记录查询日志
        :param username: 用户名
        :param query: 查询内容
        :param response: 响应内容
        :param status: 状态（success/failed）
        :param execution_time: 执行时间（秒）
        :param ip_address: IP地址
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "query": query,
            "response": response,
            "status": status,
            "execution_time": execution_time,
            "ip_address": ip_address
        }
        
        # 读取现有日志
        with open(self.log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        # 添加新日志
        logs.append(log_entry)
        
        # 保存日志（保留最近1000条）
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    
    def get_logs(self, limit: int = 100, username: Optional[str] = None) -> list:
        """
        获取日志
        :param limit: 限制条数
        :param username: 用户名过滤
        :return: 日志列表
        """
        with open(self.log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        if username:
            logs = [log for log in logs if log.get('username') == username]
        
        return logs[-limit:]
    
    def search_logs(self, keyword: str, limit: int = 50) -> list:
        """
        搜索日志
        :param keyword: 搜索关键词
        :param limit: 限制条数
        :return: 日志列表
        """
        with open(self.log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        results = []
        for log in logs:
            if keyword in log.get('query', '') or keyword in log.get('response', ''):
                results.append(log)
        
        return results[-limit:]
    
    def clear_logs(self):
        """
        清空日志
        """
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

# 全局审计日志记录器实例
audit_logger = AuditLogger()
