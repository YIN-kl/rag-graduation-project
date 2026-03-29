import os
import json
from typing import Dict, List, Optional, Tuple

class RBAC:
    """
    基于角色的访问控制（RBAC）模型
    """
    
    def __init__(self, data_file: str = "./auth_data.json"):
        """
        初始化RBAC模型
        :param data_file: 存储权限数据的文件路径
        """
        self.data_file = data_file
        self.load_data()
    
    def load_data(self):
        """
        从文件加载权限数据
        """
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.users = data.get('users', {})
                self.roles = data.get('roles', {})
                self.permissions = data.get('permissions', {})
                self.role_permissions = data.get('role_permissions', {})
                self.user_roles = data.get('user_roles', {})
        else:
            # 初始化默认数据
            self.users = {
                "admin": {"password": "admin123", "name": "管理员"},
                "hr": {"password": "hr123", "name": "人力资源"},
                "employee": {"password": "employee123", "name": "普通员工"}
            }
            self.roles = {
                "admin": "管理员",
                "hr": "人力资源",
                "employee": "普通员工"
            }
            self.permissions = {
                "read_all": "查看所有文档",
                "read_employee": "查看员工相关文档",
                "read_hr": "查看人力资源相关文档",
                "write_logs": "写入审计日志"
            }
            self.role_permissions = {
                "admin": ["read_all", "write_logs"],
                "hr": ["read_all", "write_logs"],
                "employee": ["read_employee"]
            }
            self.user_roles = {
                "admin": ["admin"],
                "hr": ["hr"],
                "employee": ["employee"]
            }
            self.save_data()
    
    def save_data(self):
        """
        保存权限数据到文件
        """
        data = {
            'users': self.users,
            'roles': self.roles,
            'permissions': self.permissions,
            'role_permissions': self.role_permissions,
            'user_roles': self.user_roles
        }
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        验证用户身份
        :param username: 用户名
        :param password: 密码
        :return: 是否验证成功
        """
        if username in self.users:
            return self.users[username]['password'] == password
        return False
    
    def get_user_roles(self, username: str) -> List[str]:
        """
        获取用户的角色
        :param username: 用户名
        :return: 角色列表
        """
        return self.user_roles.get(username, [])
    
    def get_role_permissions(self, role: str) -> List[str]:
        """
        获取角色的权限
        :param role: 角色名称
        :return: 权限列表
        """
        return self.role_permissions.get(role, [])
    
    def get_user_permissions(self, username: str) -> List[str]:
        """
        获取用户的所有权限
        :param username: 用户名
        :return: 权限列表
        """
        roles = self.get_user_roles(username)
        permissions = []
        for role in roles:
            role_perms = self.get_role_permissions(role)
            permissions.extend(role_perms)
        return list(set(permissions))  # 去重
    
    def has_permission(self, username: str, permission: str) -> bool:
        """
        检查用户是否有指定权限
        :param username: 用户名
        :param permission: 权限名称
        :return: 是否有该权限
        """
        user_permissions = self.get_user_permissions(username)
        return permission in user_permissions
    
    def add_user(self, username: str, password: str, name: str):
        """
        添加新用户
        :param username: 用户名
        :param password: 密码
        :param name: 用户名
        """
        self.users[username] = {"password": password, "name": name}
        self.user_roles[username] = ["employee"]  # 默认角色
        self.save_data()
    
    def add_role(self, role: str, description: str):
        """
        添加新角色
        :param role: 角色名称
        :param description: 角色描述
        """
        self.roles[role] = description
        self.role_permissions[role] = []
        self.save_data()
    
    def add_permission(self, permission: str, description: str):
        """
        添加新权限
        :param permission: 权限名称
        :param description: 权限描述
        """
        self.permissions[permission] = description
        self.save_data()
    
    def assign_role(self, username: str, role: str):
        """
        为用户分配角色
        :param username: 用户名
        :param role: 角色名称
        """
        if username not in self.user_roles:
            self.user_roles[username] = []
        if role not in self.user_roles[username]:
            self.user_roles[username].append(role)
        self.save_data()
    
    def assign_permission(self, role: str, permission: str):
        """
        为角色分配权限
        :param role: 角色名称
        :param permission: 权限名称
        """
        if role not in self.role_permissions:
            self.role_permissions[role] = []
        if permission not in self.role_permissions[role]:
            self.role_permissions[role].append(permission)
        self.save_data()

# 全局RBAC实例
rbac = RBAC()
