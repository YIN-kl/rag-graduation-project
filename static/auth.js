(function () {
    const TOKEN_KEY = "rag_access_token";
    const USERNAME_PATTERN = /^[a-zA-Z0-9_.-]+$/;
    const USERNAME_RULE_HINT = "用户名需 3-32 位，仅支持字母、数字、下划线（_）、短横线（-）和点号（.），不支持邮箱格式。";

    const el = {
        loginForm: document.getElementById("login-form"),
        registerForm: document.getElementById("register-form"),
        loginButton: document.getElementById("login-button"),
        registerButton: document.getElementById("register-button"),
        resetLoginButton: document.getElementById("reset-login-button"),
        loginMessage: document.getElementById("login-message"),
        registerMessage: document.getElementById("register-message"),
        username: document.getElementById("auth-username"),
        password: document.getElementById("auth-password"),
        registerUsername: document.getElementById("register-username"),
        registerDisplayName: document.getElementById("register-display-name"),
        registerPassword: document.getElementById("register-password"),
        registerConfirm: document.getElementById("register-confirm"),
    };

    function setBusy(button, busy, busyText, idleText) {
        if (!button) return;
        button.disabled = busy;
        button.textContent = busy ? busyText : idleText;
    }

    function setMessage(target, text, type) {
        if (!target) return;
        target.textContent = text;
        target.classList.remove("error", "success");
        if (type) target.classList.add(type);
    }

    function extractErrorMessage(data, status) {
        const detail = data?.detail;
        if (typeof detail === "string" && detail.trim()) return detail;
        if (Array.isArray(detail) && detail.length > 0) {
            const first = detail[0];
            if (typeof first === "string" && first.trim()) return first;
            if (first && typeof first.msg === "string" && first.msg.trim()) return first.msg;
        }
        return `请求失败（${status}）`;
    }

    function validateRegisterInput(username, displayName, password, confirm) {
        if (!username || !displayName || !password) return "请完整填写注册信息。";
        if (username.length < 3) return "用户名至少 3 位。";
        if (username.length > 32) return "用户名最多 32 位。";
        if (username.includes("@")) return "用户名不支持邮箱格式，请使用字母、数字或符号（._-）。";
        if (!USERNAME_PATTERN.test(username)) return "用户名仅支持字母、数字、下划线（_）、短横线（-）和点号（.）。";
        if (password.length < 6) return "密码至少 6 位。";
        if (password !== confirm) return "两次输入的密码不一致。";
        return "";
    }

    async function apiPost(path, payload) {
        const response = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload || {}),
        });

        let data = null;
        try {
            data = await response.json();
        } catch {
            data = null;
        }

        if (!response.ok) throw new Error(extractErrorMessage(data, response.status));
        return data;
    }

    async function validateExistingToken() {
        const token = sessionStorage.getItem(TOKEN_KEY);
        if (!token) return false;

        try {
            const response = await fetch("/me", {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (response.ok) {
                window.location.replace("/app");
                return true;
            }
        } catch {
            // Ignore and clear below.
        }
        sessionStorage.removeItem(TOKEN_KEY);
        return false;
    }

    async function login(event) {
        event.preventDefault();
        const username = el.username.value.trim();
        const password = el.password.value;
        if (!username || !password) {
            setMessage(el.loginMessage, "请输入用户名和密码。", "error");
            return;
        }

        setBusy(el.loginButton, true, "登录中...", "登录并进入系统");
        setMessage(el.loginMessage, "正在验证账号，请稍候...", "");
        try {
            const data = await apiPost("/login", { username, password });
            sessionStorage.setItem(TOKEN_KEY, data.access_token);
            setMessage(el.loginMessage, "登录成功，正在跳转系统主页...", "success");
            window.location.replace("/app");
        } catch (error) {
            setMessage(el.loginMessage, `登录失败：${error.message}`, "error");
        } finally {
            setBusy(el.loginButton, false, "登录中...", "登录并进入系统");
        }
    }

    async function register(event) {
        event.preventDefault();
        const username = el.registerUsername.value.trim();
        const displayName = el.registerDisplayName.value.trim();
        const password = el.registerPassword.value;
        const confirm = el.registerConfirm.value;

        const validationError = validateRegisterInput(username, displayName, password, confirm);
        if (validationError) {
            setMessage(el.registerMessage, validationError, "error");
            return;
        }

        setBusy(el.registerButton, true, "提交中...", "提交注册申请");
        setMessage(el.registerMessage, "正在提交注册申请...", "");
        try {
            const data = await apiPost("/register", {
                username,
                display_name: displayName,
                password,
            });
            setMessage(
                el.registerMessage,
                `${data.message}（账号：${data.username}，状态：${data.status}）`,
                "success",
            );
            el.registerForm.reset();
        } catch (error) {
            setMessage(el.registerMessage, `注册失败：${error.message}`, "error");
        } finally {
            setBusy(el.registerButton, false, "提交中...", "提交注册申请");
        }
    }

    if (el.loginForm) {
        el.loginForm.addEventListener("submit", login);
    }
    if (el.registerForm) {
        if (el.registerMessage && !el.registerMessage.textContent?.trim()) {
            setMessage(el.registerMessage, USERNAME_RULE_HINT, "");
        }
        el.registerForm.addEventListener("submit", register);
    }
    if (el.resetLoginButton) {
        el.resetLoginButton.addEventListener("click", () => {
            if (el.username) el.username.value = "";
            if (el.password) el.password.value = "";
            setMessage(el.loginMessage, "已清空登录信息。", "");
        });
    }

    validateExistingToken();
})();
