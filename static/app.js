(function () {
    function cloneById(id) {
        const element = document.getElementById(id);
        const clone = element.cloneNode(true);
        element.replaceWith(clone);
        return clone;
    }

    const loginForm = cloneById("login-form");
    const questionForm = cloneById("question-form");
    const detailToggle = cloneById("detail-toggle");
    const refreshLogsButton = cloneById("refresh-logs-button");
    const refreshHealthButton = cloneById("refresh-health-button");
    const profileButton = document.getElementById("profile-button");
    const clearAnswerButton = document.getElementById("clear-answer-button");
    const clearSessionButton = document.getElementById("clear-session-button");
    const questionButton = document.getElementById("question-button");
    const loginButton = document.getElementById("login-button");
    const logLimit = cloneById("log-limit");
    const logStatus = cloneById("log-status");
    const logUsername = cloneById("log-username");
    const logKeyword = cloneById("log-keyword");

    const initialHealth = typeof initialStatus !== "undefined" ? initialStatus : null;
    const state = {
        token: "",
        detailed: false,
        profile: null,
        health: initialHealth,
        logs: [],
        sessionId: "",
        conversationHistory: [],
        lastAnswer: null,
    };

    const profilePanel = document.getElementById("profile-panel");
    const permissionBadges = document.getElementById("permission-badges");
    const heroStatusText = document.getElementById("hero-status-text");
    const heroStatusDot = document.getElementById("hero-status-dot");
    const healthPanel = document.getElementById("health-panel");
    const logLocked = document.getElementById("log-locked");
    const logDashboard = document.getElementById("log-dashboard");
    const logAccessPill = document.getElementById("log-access-pill");
    const answerPlaceholder = document.getElementById("answer-placeholder");
    const answerRich = document.getElementById("answer-rich");
    const answerMeta = document.getElementById("answer-meta");
    const answerText = document.getElementById("answer-text");
    const sourceList = document.getElementById("source-list");
    const conversationList = document.getElementById("conversation-list");
    const detailResultWrap = document.getElementById("detail-result-wrap");
    const detailResult = document.getElementById("detail-result");
    const sessionPill = document.getElementById("session-pill");

    function setBusy(button, busy, busyText, idleText) {
        button.disabled = busy;
        button.textContent = busy ? busyText : idleText;
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function formatTimestamp(value) {
        if (!value) {
            return "未知时间";
        }
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) {
            return value;
        }
        return parsed.toLocaleString("zh-CN", {
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    }

    function formatDuration(value) {
        const ms = Number(value || 0) * 1000;
        if (!Number.isFinite(ms)) {
            return "0 ms";
        }
        if (ms < 1000) {
            return `${ms.toFixed(0)} ms`;
        }
        return `${(ms / 1000).toFixed(2)} s`;
    }

    function shortText(value, limit = 96) {
        const text = String(value || "");
        return text.length > limit ? `${text.slice(0, limit)}...` : text;
    }

    function updateDetailButton() {
        detailToggle.textContent = state.detailed ? "当前为详细链路结果" : "当前为普通结果";
    }

    function updateSessionPill() {
        sessionPill.textContent = state.sessionId ? `会话 ${state.sessionId}` : "新会话未开始";
    }

    function clearAnswerPanels() {
        answerMeta.innerHTML = "";
        answerText.textContent = "";
        sourceList.innerHTML = "";
        conversationList.innerHTML = "";
        detailResult.textContent = "";
        detailResultWrap.classList.add("hidden");
    }

    function setAnswerNotice(message) {
        answerPlaceholder.textContent = message;
        answerPlaceholder.classList.remove("hidden");
        answerRich.classList.add("hidden");
        clearAnswerPanels();
    }

    function resetConversationState(message) {
        state.sessionId = "";
        state.conversationHistory = [];
        state.lastAnswer = null;
        updateSessionPill();
        setAnswerNotice(message);
    }

    function renderHealth(statusData) {
        if (!statusData) {
            return;
        }
        state.health = statusData;
        heroStatusText.textContent = `系统状态：${statusData.status}`;
        heroStatusDot.style.background = statusData.status === "ok" ? "var(--success)" : "var(--accent)";
        heroStatusDot.style.boxShadow = statusData.status === "ok"
            ? "0 0 0 7px rgba(47, 124, 93, 0.12)"
            : "0 0 0 7px rgba(187, 90, 54, 0.12)";

        document.getElementById("metric-documents").textContent = statusData.documents_count;
        document.getElementById("metric-vector").textContent = statusData.vector_store_ready ? "已就绪" : "未就绪";
        document.getElementById("metric-embedding").textContent = statusData.embedding_model || "未配置";
        document.getElementById("metric-chat").textContent = statusData.chat_configured ? "正常" : "缺失";

        const warnings = Array.isArray(statusData.warnings) && statusData.warnings.length
            ? statusData.warnings.map((item) => `- ${item}`).join("\n")
            : "当前没有额外风险提醒，可以直接用于演示。";
        const files = Array.isArray(statusData.document_files) && statusData.document_files.length
            ? statusData.document_files.join("、")
            : "暂无文档";

        healthPanel.textContent = [
            `文档数量：${statusData.documents_count}`,
            `向量索引：${statusData.vector_store_ready ? "已构建" : "未构建"}`,
            `Embedding 接口：${statusData.embedding_base_url || "未配置"}`,
            `文档清单：${files}`,
            "",
            "提醒：",
            warnings,
        ].join("\n");
    }

    function renderProfile() {
        if (!state.profile) {
            profilePanel.textContent = "尚未登录。";
            permissionBadges.innerHTML = "";
            logAccessPill.textContent = "日志权限未激活";
            return;
        }

        const { username, roles, permissions, can_view_logs } = state.profile;
        profilePanel.textContent = [
            `当前用户：${username}`,
            `角色：${roles.join("、") || "无"}`,
            `权限：${permissions.join("、") || "无"}`,
            `日志查看权限：${can_view_logs ? "已开启" : "未开启"}`,
        ].join("\n");

        permissionBadges.innerHTML = permissions.length
            ? permissions.map((item) => `<span class="mini-pill">${escapeHtml(item)}</span>`).join("")
            : '<span class="mini-pill">暂无权限</span>';

        logAccessPill.textContent = can_view_logs ? "日志权限已激活" : "当前账号无日志权限";
    }

    function renderLogAccess() {
        const canViewLogs = Boolean(state.profile && state.profile.can_view_logs);
        logLocked.classList.toggle("hidden", canViewLogs);
        logDashboard.classList.toggle("hidden", !canViewLogs);
    }

    function renderLogs() {
        renderLogAccess();

        const canViewLogs = Boolean(state.profile && state.profile.can_view_logs);
        if (!canViewLogs) {
            return;
        }

        const logs = [...state.logs];
        const total = logs.length;
        const successCount = logs.filter((item) => item.status === "success").length;
        const failedCount = logs.filter((item) => item.status === "failed").length;
        const successRate = total ? ((successCount / total) * 100).toFixed(1) : "0.0";
        const avgLatency = total
            ? logs.reduce((sum, item) => sum + Number(item.execution_time || 0), 0) / total
            : 0;
        const userCounts = {};
        logs.forEach((item) => {
            const username = item.username || "unknown";
            userCounts[username] = (userCounts[username] || 0) + 1;
        });
        const userEntries = Object.entries(userCounts).sort((a, b) => b[1] - a[1]);
        const maxUserCount = userEntries.length ? userEntries[0][1] : 0;

        document.getElementById("summary-total").textContent = total;
        document.getElementById("summary-success-rate").textContent = `${successRate}%`;
        document.getElementById("summary-latency").textContent = formatDuration(avgLatency);
        document.getElementById("summary-users").textContent = userEntries.length;
        document.getElementById("user-chart-caption").textContent = total ? `${userEntries.length} 个用户` : "暂无数据";
        document.getElementById("status-chart-caption").textContent = total ? `success ${successCount} / failed ${failedCount}` : "暂无数据";
        document.getElementById("timeline-caption").textContent = total ? `最近 ${Math.min(total, 8)} 条` : "暂无日志";

        const userChart = document.getElementById("user-chart");
        userChart.innerHTML = userEntries.length
            ? userEntries.map(([username, count]) => `
                <div class="bar-row">
                    <div class="bar-meta"><span>${escapeHtml(username)}</span><strong>${count}</strong></div>
                    <div class="bar-track"><div class="bar-fill" style="width:${maxUserCount ? (count / maxUserCount) * 100 : 0}%"></div></div>
                </div>
            `).join("")
            : '<div class="empty-state">暂无可视化数据。</div>';

        const statusChart = document.getElementById("status-chart");
        statusChart.innerHTML = total
            ? `
                <div class="bar-row">
                    <div class="bar-meta"><span>success</span><strong>${successCount}</strong></div>
                    <div class="bar-track"><div class="bar-fill success" style="width:${(successCount / total) * 100}%"></div></div>
                </div>
                <div class="bar-row">
                    <div class="bar-meta"><span>failed</span><strong>${failedCount}</strong></div>
                    <div class="bar-track"><div class="bar-fill failed" style="width:${(failedCount / total) * 100}%"></div></div>
                </div>
            `
            : '<div class="empty-state">暂无可视化数据。</div>';

        const timelineList = document.getElementById("timeline-list");
        const recentLogs = logs.slice().reverse().slice(0, 8);
        timelineList.innerHTML = recentLogs.length
            ? recentLogs.map((item) => `
                <article class="timeline-card">
                    <div class="timeline-top">
                        <span class="status-badge ${item.status === "success" ? "status-success" : "status-failed"}">${escapeHtml(item.status || "unknown")}</span>
                        <span>${escapeHtml(item.username || "unknown")} · ${formatTimestamp(item.timestamp)}</span>
                        <span>${formatDuration(item.execution_time)}</span>
                    </div>
                    <div class="query-text">${escapeHtml(shortText(item.query, 120))}</div>
                    <div class="response-preview">${escapeHtml(shortText(item.response, 160))}</div>
                </article>
            `).join("")
            : '<div class="empty-state">没有符合筛选条件的日志。</div>';
    }

    function renderSources(sources) {
        sourceList.innerHTML = Array.isArray(sources) && sources.length
            ? sources.map((item) => `
                <article class="source-card">
                    <div class="source-title">
                        <strong>${escapeHtml(item.filename || "未命名文档")}</strong>
                        <span class="mini-pill">${escapeHtml(item.document_type || "文档片段")}</span>
                    </div>
                    <div class="source-snippet">${escapeHtml(item.snippet || "该片段未提供可展示内容。")}</div>
                </article>
            `).join("")
            : '<div class="empty-state">本次回答没有返回可展示的引用片段。</div>';
    }

    function renderConversation(history) {
        conversationList.innerHTML = Array.isArray(history) && history.length
            ? history.map((item) => `
                <article class="conversation-item">
                    <div class="source-title">
                        <span class="conversation-role">${item.role === "assistant" ? "助手" : "用户"}</span>
                        <span class="muted">${escapeHtml(formatTimestamp(item.timestamp))}</span>
                    </div>
                    <div class="conversation-content">${escapeHtml(item.content || "")}</div>
                </article>
            `).join("")
            : '<div class="empty-state">当前会话还没有历史记录。</div>';
    }

    function renderDetailedResult(data) {
        if (state.detailed && data && data.detailed_result) {
            detailResultWrap.classList.remove("hidden");
            detailResult.textContent = JSON.stringify(data.detailed_result, null, 2);
            return;
        }
        detailResultWrap.classList.add("hidden");
        detailResult.textContent = "";
    }

    function renderRichAnswer(data) {
        state.lastAnswer = data;
        state.sessionId = data.session_id || state.sessionId;
        state.conversationHistory = Array.isArray(data.history) ? data.history : [];

        updateSessionPill();
        answerPlaceholder.classList.add("hidden");
        answerRich.classList.remove("hidden");

        const sources = Array.isArray(data.sources) ? data.sources : [];
        answerMeta.innerHTML = [
            `<span class="mini-pill">会话 ${escapeHtml(state.sessionId || "未分配")}</span>`,
            `<span class="mini-pill">耗时 ${escapeHtml(formatDuration(data.execution_time))}</span>`,
            `<span class="mini-pill">来源 ${sources.length} 份</span>`,
            `<span class="mini-pill">${state.detailed ? "详细模式" : "普通模式"}</span>`,
        ].join("");
        answerText.textContent = data.answer || "系统未返回答案。";
        renderSources(sources);
        renderConversation(state.conversationHistory);
        renderDetailedResult(data);
    }

    async function apiGet(path) {
        const response = await fetch(path, {
            headers: state.token ? { Authorization: `Bearer ${state.token}` } : {},
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "请求失败");
        }
        return data;
    }

    async function apiDelete(path) {
        const response = await fetch(path, {
            method: "DELETE",
            headers: state.token ? { Authorization: `Bearer ${state.token}` } : {},
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "请求失败");
        }
        return data;
    }

    async function fetchHealth() {
        setBusy(refreshHealthButton, true, "刷新中...", "刷新状态");
        try {
            renderHealth(await apiGet("/health"));
        } catch (error) {
            healthPanel.textContent = `状态读取失败：${error.message}`;
        } finally {
            setBusy(refreshHealthButton, false, "刷新中...", "刷新状态");
        }
    }

    async function fetchProfile() {
        if (!state.token) {
            profilePanel.textContent = "请先登录。";
            return;
        }
        setBusy(profileButton, true, "读取中...", "刷新身份信息");
        try {
            state.profile = await apiGet("/me");
            renderProfile();
            renderLogAccess();
        } catch (error) {
            profilePanel.textContent = `身份读取失败：${error.message}`;
        } finally {
            setBusy(profileButton, false, "读取中...", "刷新身份信息");
        }
    }

    async function fetchLogs() {
        const canViewLogs = Boolean(state.profile && state.profile.can_view_logs);
        if (!state.token || !canViewLogs) {
            renderLogAccess();
            return;
        }

        const params = new URLSearchParams();
        params.set("limit", logLimit.value);
        if (logStatus.value.trim()) {
            params.set("status_filter", logStatus.value.trim());
        }
        if (logUsername.value.trim()) {
            params.set("username", logUsername.value.trim());
        }
        if (logKeyword.value.trim()) {
            params.set("keyword", logKeyword.value.trim());
        }

        setBusy(refreshLogsButton, true, "加载中...", "刷新日志");
        try {
            state.logs = await apiGet(`/logs?${params.toString()}`);
            renderLogs();
        } catch (error) {
            logDashboard.classList.remove("hidden");
            logLocked.classList.add("hidden");
            document.getElementById("timeline-list").innerHTML = `<div class="empty-state">日志加载失败：${escapeHtml(error.message)}</div>`;
        } finally {
            setBusy(refreshLogsButton, false, "加载中...", "刷新日志");
        }
    }

    async function login(event) {
        event.preventDefault();
        setBusy(loginButton, true, "登录中...", "登录系统");
        profilePanel.textContent = "正在登录，请稍候...";

        try {
            const response = await fetch("/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    username: document.getElementById("username").value.trim(),
                    password: document.getElementById("password").value,
                }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "登录失败");
            }
            state.token = data.access_token;
            resetConversationState("登录成功，现在可以开始提问，系统会自动保留上下文。");
            await fetchProfile();
            await fetchLogs();
        } catch (error) {
            state.token = "";
            state.profile = null;
            state.logs = [];
            renderProfile();
            renderLogs();
            resetConversationState("登录后就可以开始提问。");
            profilePanel.textContent = `登录失败：${error.message}`;
        } finally {
            setBusy(loginButton, false, "登录中...", "登录系统");
        }
    }

    async function askQuestion(event) {
        event.preventDefault();
        if (!state.token) {
            setAnswerNotice("请先登录获取 Token。");
            return;
        }

        const question = document.getElementById("question-input").value.trim();
        if (!question) {
            setAnswerNotice("请输入一个有效问题。");
            return;
        }

        setBusy(questionButton, true, "提问中...", "开始提问");
        setAnswerNotice("系统正在处理你的问题，请稍候...");

        try {
            const response = await fetch("/question", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${state.token}`,
                },
                body: JSON.stringify({
                    input: question,
                    detailed: state.detailed,
                    return_rich_response: true,
                    session_id: state.sessionId || null,
                }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "问答失败");
            }
            renderRichAnswer(data);
            await fetchLogs();
        } catch (error) {
            setAnswerNotice(`问答失败：${error.message}`);
            await fetchLogs();
        } finally {
            setBusy(questionButton, false, "提问中...", "开始提问");
        }
    }

    async function clearConversation() {
        if (!state.sessionId) {
            resetConversationState("当前还没有可清空的会话。");
            return;
        }

        if (!state.token) {
            resetConversationState("会话已清空。");
            return;
        }

        setBusy(clearSessionButton, true, "清空中...", "清空会话");
        try {
            await apiDelete(`/conversation/${encodeURIComponent(state.sessionId)}`);
            resetConversationState("会话已清空，下一次提问会开始新的上下文。");
        } catch (error) {
            setAnswerNotice(`清空会话失败：${error.message}`);
        } finally {
            setBusy(clearSessionButton, false, "清空中...", "清空会话");
        }
    }

    loginForm.addEventListener("submit", login);
    questionForm.addEventListener("submit", askQuestion);
    clearAnswerButton.addEventListener("click", () => {
        state.lastAnswer = null;
        setAnswerNotice(state.sessionId ? "结果已清空，本轮会话上下文仍保留，可继续追问。" : "结果已清空。");
        updateSessionPill();
    });
    clearSessionButton.addEventListener("click", clearConversation);
    detailToggle.addEventListener("click", () => {
        state.detailed = !state.detailed;
        updateDetailButton();
        if (state.lastAnswer) {
            renderRichAnswer(state.lastAnswer);
        } else {
            setAnswerNotice(
                state.detailed
                    ? "详细链路模式已开启，下一次提问会返回引用、会话和详细结果。"
                    : "已切换为普通结果模式。"
            );
        }
    });
    refreshLogsButton.addEventListener("click", fetchLogs);
    refreshHealthButton.addEventListener("click", fetchHealth);
    profileButton.addEventListener("click", fetchProfile);
    logLimit.addEventListener("change", fetchLogs);
    logStatus.addEventListener("change", fetchLogs);
    logUsername.addEventListener("change", fetchLogs);
    logKeyword.addEventListener("change", fetchLogs);

    document.querySelectorAll(".sample-question").forEach((button) => {
        button.addEventListener("click", () => {
            document.getElementById("question-input").value = button.textContent.trim();
        });
    });
    document.querySelectorAll(".preset-account").forEach((button) => {
        button.addEventListener("click", () => {
            document.getElementById("username").value = button.dataset.username;
            document.getElementById("password").value = button.dataset.password;
            profilePanel.textContent = `已填入 ${button.dataset.username} 账号，点击“登录系统”即可继续。`;
        });
    });

    updateDetailButton();
    updateSessionPill();
    renderHealth(initialHealth);
    renderProfile();
    renderLogAccess();
    setAnswerNotice("登录后就可以开始提问。");
})();
