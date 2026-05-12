import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Optional


class RBAC:
    """Simple file-based user and role management with approval workflow."""

    USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")

    DEFAULT_PERMISSIONS = {
        "read_all": "查看全部制度文档",
        "read_employee": "查看员工公开制度",
        "read_hr": "查看人力资源制度",
        "write_logs": "查看审计日志",
        "manage_users": "管理用户与权限",
    }

    DEFAULT_ROLES = {
        "admin": {
            "label": "系统管理员",
            "permissions": ["read_all", "read_employee", "read_hr", "write_logs", "manage_users"],
        },
        "hr": {
            "label": "人力资源",
            "permissions": ["read_all", "read_employee", "read_hr", "write_logs"],
        },
        "employee": {
            "label": "普通员工",
            "permissions": ["read_employee"],
        },
    }

    VALID_STATUSES = {"pending", "approved", "rejected", "disabled"}

    def __init__(self, data_file: str = "./auth_data.json"):
        self.data_file = str(Path(data_file))
        self._lock = RLock()
        self.schema_version = 2
        self.users: dict[str, dict[str, Any]] = {}
        self.roles: dict[str, dict[str, Any]] = {}
        self.permissions: dict[str, str] = {}
        self.load_data()

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _seed_default_admin(self) -> None:
        if self.users:
            return
        now = self._now_iso()
        self.users["admin"] = {
            "display_name": "系统管理员",
            "password_hash": self._hash_password("admin123"),
            "status": "approved",
            "roles": ["admin"],
            "created_at": now,
            "reviewed_at": now,
            "approved_at": now,
            "approved_by": "system",
            "review_note": "初始化管理员",
        }

    def _normalize_role(self, role_key: str, role_data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        role_data = role_data or {}
        label = str(role_data.get("label") or role_data.get("description") or role_key)
        perms = role_data.get("permissions") or []
        perms = [str(item) for item in perms if str(item)]
        return {"label": label, "permissions": list(dict.fromkeys(perms))}

    def _normalize_user(self, username: str, data: dict[str, Any]) -> dict[str, Any]:
        status = str(data.get("status") or "pending").lower()
        if status not in self.VALID_STATUSES:
            status = "pending"

        roles = [str(item) for item in (data.get("roles") or []) if str(item)]
        roles = list(dict.fromkeys(roles))

        password_hash = str(data.get("password_hash") or "")
        if not password_hash and data.get("password"):
            password_hash = self._hash_password(str(data["password"]))

        created_at = str(data.get("created_at") or self._now_iso())
        reviewed_at = data.get("reviewed_at")
        approved_at = data.get("approved_at")
        approved_by = data.get("approved_by")

        if status == "approved":
            reviewed_at = reviewed_at or self._now_iso()
            approved_at = approved_at or reviewed_at

        return {
            "display_name": str(data.get("display_name") or data.get("name") or username),
            "password_hash": password_hash,
            "status": status,
            "roles": roles,
            "created_at": created_at,
            "reviewed_at": reviewed_at,
            "approved_at": approved_at,
            "approved_by": approved_by,
            "review_note": str(data.get("review_note") or ""),
        }

    def _migrate_legacy_data(self, data: dict[str, Any]) -> None:
        legacy_users = data.get("users", {}) or {}
        legacy_roles = data.get("roles", {}) or {}
        legacy_permissions = data.get("permissions", {}) or {}
        legacy_role_permissions = data.get("role_permissions", {}) or {}
        legacy_user_roles = data.get("user_roles", {}) or {}

        self.permissions = dict(self.DEFAULT_PERMISSIONS)
        for key, label in legacy_permissions.items():
            self.permissions[str(key)] = str(label)

        self.roles = {
            role_key: {
                "label": str(role_label),
                "permissions": [str(item) for item in legacy_role_permissions.get(role_key, [])],
            }
            for role_key, role_label in legacy_roles.items()
        }

        for role_key, role_data in self.DEFAULT_ROLES.items():
            if role_key not in self.roles:
                self.roles[role_key] = {"label": role_data["label"], "permissions": role_data["permissions"]}

        migrated_users: dict[str, dict[str, Any]] = {}
        now = self._now_iso()
        for username, info in legacy_users.items():
            user_info = dict(info or {})
            roles = legacy_user_roles.get(username, [])
            migrated_users[str(username)] = self._normalize_user(
                str(username),
                {
                    "display_name": user_info.get("name") or username,
                    "password": user_info.get("password") or "",
                    "status": "approved",
                    "roles": roles,
                    "created_at": now,
                    "reviewed_at": now,
                    "approved_at": now,
                    "approved_by": "legacy_migration",
                    "review_note": "从旧版数据迁移",
                },
            )

        self.users = migrated_users
        self._seed_default_admin()

    def _load_v2_data(self, data: dict[str, Any]) -> None:
        raw_permissions = data.get("permissions", {}) or {}
        raw_roles = data.get("roles", {}) or {}
        raw_users = data.get("users", {}) or {}

        self.permissions = dict(self.DEFAULT_PERMISSIONS)
        self.permissions.update({str(key): str(value) for key, value in raw_permissions.items()})

        self.roles = {
            str(role_key): self._normalize_role(str(role_key), dict(role_data or {}))
            for role_key, role_data in raw_roles.items()
        }
        for role_key, role_data in self.DEFAULT_ROLES.items():
            if role_key not in self.roles:
                self.roles[role_key] = {"label": role_data["label"], "permissions": list(role_data["permissions"])}
            else:
                current_perms = [str(item) for item in self.roles[role_key].get("permissions", []) if str(item)]
                default_perms = [str(item) for item in role_data.get("permissions", []) if str(item)]
                merged_perms = list(dict.fromkeys(current_perms + default_perms))
                self.roles[role_key]["permissions"] = merged_perms
                if not self.roles[role_key].get("label"):
                    self.roles[role_key]["label"] = role_data["label"]

        self.users = {
            str(username): self._normalize_user(str(username), dict(user_data or {}))
            for username, user_data in raw_users.items()
        }
        self._seed_default_admin()

    def load_data(self) -> None:
        with self._lock:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if data.get("schema_version") == 2:
                    self._load_v2_data(data)
                else:
                    self._migrate_legacy_data(data)
                self.save_data()
                return

            self.permissions = dict(self.DEFAULT_PERMISSIONS)
            self.roles = {
                key: {"label": value["label"], "permissions": list(value["permissions"])}
                for key, value in self.DEFAULT_ROLES.items()
            }
            self.users = {}
            self._seed_default_admin()
            self.save_data()

    def save_data(self) -> None:
        with self._lock:
            payload = {
                "schema_version": self.schema_version,
                "users": self.users,
                "roles": self.roles,
                "permissions": self.permissions,
            }
            with open(self.data_file, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)

    def get_user(self, username: str) -> Optional[dict[str, Any]]:
        user = self.users.get(username)
        if not user:
            return None
        return {"username": username, **dict(user)}

    def user_exists(self, username: str) -> bool:
        return username in self.users

    def verify_password(self, username: str, password: str) -> bool:
        user = self.users.get(username)
        if not user:
            return False
        return user.get("password_hash") == self._hash_password(password)

    def authenticate(self, username: str, password: str) -> bool:
        ok, _ = self.authenticate_user(username, password)
        return ok

    def authenticate_user(self, username: str, password: str) -> tuple[bool, str]:
        user = self.users.get(username)
        if not user:
            return False, "用户名或密码错误"
        if user.get("password_hash") != self._hash_password(password):
            return False, "用户名或密码错误"

        status = user.get("status")
        if status == "pending":
            return False, "账号待管理员审批"
        if status == "rejected":
            return False, "注册申请已被拒绝，请联系管理员"
        if status == "disabled":
            return False, "账号已停用，请联系管理员"
        if status != "approved":
            return False, "账号状态异常，请联系管理员"

        return True, "ok"

    def register_user(self, username: str, password: str, display_name: str) -> dict[str, Any]:
        normalized_username = username.strip()
        if not normalized_username:
            raise ValueError("用户名不能为空")
        if len(normalized_username) < 3:
            raise ValueError("用户名至少 3 位")
        if len(normalized_username) > 32:
            raise ValueError("用户名最多 32 位")
        if "@" in normalized_username:
            raise ValueError("用户名不支持邮箱格式，请使用字母、数字或符号（._-）")
        if not self.USERNAME_PATTERN.fullmatch(normalized_username):
            raise ValueError("用户名仅支持字母、数字、下划线（_）、短横线（-）和点号（.）")
        if len(password) < 6:
            raise ValueError("密码至少 6 位")

        with self._lock:
            if normalized_username in self.users:
                raise ValueError("用户名已存在")

            now = self._now_iso()
            self.users[normalized_username] = {
                "display_name": display_name.strip() or normalized_username,
                "password_hash": self._hash_password(password),
                "status": "pending",
                "roles": [],
                "created_at": now,
                "reviewed_at": None,
                "approved_at": None,
                "approved_by": None,
                "review_note": "",
            }
            self.save_data()
            return self.get_user(normalized_username) or {}

    def get_user_roles(self, username: str) -> list[str]:
        user = self.users.get(username)
        if not user:
            return []
        return [role for role in user.get("roles", []) if role in self.roles]

    def get_role_permissions(self, role: str) -> list[str]:
        role_data = self.roles.get(role, {})
        return [perm for perm in role_data.get("permissions", []) if perm in self.permissions]

    def get_user_permissions(self, username: str) -> list[str]:
        user = self.users.get(username)
        if not user or user.get("status") != "approved":
            return []

        merged: list[str] = []
        for role in self.get_user_roles(username):
            merged.extend(self.get_role_permissions(role))
        return sorted(set(merged))

    def has_permission(self, username: str, permission: str) -> bool:
        return permission in self.get_user_permissions(username)

    def list_roles(self) -> dict[str, dict[str, Any]]:
        return {
            role_key: {
                "label": role_data.get("label", role_key),
                "permissions": list(role_data.get("permissions", [])),
            }
            for role_key, role_data in self.roles.items()
        }

    def list_users(self) -> list[dict[str, Any]]:
        return [
            {"username": username, **dict(user)}
            for username, user in sorted(self.users.items(), key=lambda item: item[0].lower())
        ]

    def set_user_roles(self, username: str, roles: list[str], actor: str) -> dict[str, Any]:
        if username not in self.users:
            raise ValueError("用户不存在")

        normalized_roles = []
        for role in roles:
            role_key = str(role).strip()
            if not role_key:
                continue
            if role_key not in self.roles:
                raise ValueError(f"角色不存在: {role_key}")
            normalized_roles.append(role_key)

        if not normalized_roles:
            raise ValueError("至少分配一个角色")

        user = self.users[username]
        user["roles"] = list(dict.fromkeys(normalized_roles))
        user["reviewed_at"] = self._now_iso()
        user["approved_by"] = actor
        if user.get("status") == "approved" and not user.get("approved_at"):
            user["approved_at"] = user["reviewed_at"]

        self.save_data()
        return self.get_user(username) or {}

    def review_user(
        self,
        username: str,
        actor: str,
        action: str,
        roles: Optional[list[str]] = None,
        review_note: str = "",
    ) -> dict[str, Any]:
        if username not in self.users:
            raise ValueError("用户不存在")

        normalized_action = action.strip().lower()
        if normalized_action not in {"approve", "reject", "disable"}:
            raise ValueError("不支持的审核动作")

        user = self.users[username]
        reviewed_at = self._now_iso()
        user["reviewed_at"] = reviewed_at
        user["approved_by"] = actor
        user["review_note"] = review_note.strip()

        if normalized_action == "approve":
            assign_roles = roles if roles is not None else (user.get("roles") or ["employee"])
            normalized_roles = []
            for role in assign_roles:
                role_key = str(role).strip()
                if role_key not in self.roles:
                    raise ValueError(f"角色不存在: {role_key}")
                normalized_roles.append(role_key)
            if not normalized_roles:
                normalized_roles = ["employee"]
            user["roles"] = list(dict.fromkeys(normalized_roles))
            user["status"] = "approved"
            user["approved_at"] = reviewed_at
        elif normalized_action == "reject":
            user["status"] = "rejected"
            user["approved_at"] = None
        else:
            user["status"] = "disabled"

        self.save_data()
        return self.get_user(username) or {}

    def add_role(self, role: str, description: str):
        role_key = role.strip()
        if not role_key:
            raise ValueError("角色名不能为空")
        if role_key in self.roles:
            raise ValueError("角色已存在")
        self.roles[role_key] = {"label": description.strip() or role_key, "permissions": []}
        self.save_data()

    def add_permission(self, permission: str, description: str):
        permission_key = permission.strip()
        if not permission_key:
            raise ValueError("权限名不能为空")
        self.permissions[permission_key] = description.strip() or permission_key
        self.save_data()

    def assign_role(self, username: str, role: str):
        user = self.users.get(username)
        if not user:
            raise ValueError("用户不存在")
        if role not in self.roles:
            raise ValueError("角色不存在")

        roles = user.get("roles", [])
        if role not in roles:
            roles.append(role)
            user["roles"] = roles
            self.save_data()

    def assign_permission(self, role: str, permission: str):
        if role not in self.roles:
            raise ValueError("角色不存在")
        if permission not in self.permissions:
            raise ValueError("权限不存在")

        permissions = self.roles[role].get("permissions", [])
        if permission not in permissions:
            permissions.append(permission)
            self.roles[role]["permissions"] = permissions
            self.save_data()


rbac = RBAC()
