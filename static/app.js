(function () {
    const TOKEN_KEY = "rag_access_token";
    const state = {
        token: sessionStorage.getItem(TOKEN_KEY) || "",
        detailed: false,
        profile: null,
        health: typeof initialStatus !== "undefined" ? initialStatus : null,
        logs: [],
        knowledgeBase: null,
        sessionId: "",
        conversationHistory: [],
        lastAnswer: null,
        adminUsers: [],
        roles: {},
    };

    const el = {
        questionForm: document.getElementById("question-form"),
        profileButton: document.getElementById("profile-button"),
        logoutButton: document.getElementById("logout-button"),
        questionButton: document.getElementById("question-button"),
        clearAnswerButton: document.getElementById("clear-answer-button"),
        clearSessionButton: document.getElementById("clear-session-button"),
        detailToggle: document.getElementById("detail-toggle"),
        refreshLogsButton: document.getElementById("refresh-logs-button"),
        refreshHealthButton: document.getElementById("refresh-health-button"),
        refreshKnowledgeButton: document.getElementById("refresh-knowledge-button"),
        rebuildKnowledgeButton: document.getElementById("rebuild-knowledge-button"),
        refreshUsersButton: document.getElementById("refresh-users-button"),
        profilePanel: document.getElementById("profile-panel"),
        permissionBadges: document.getElementById("permission-badges"),
        heroStatusText: document.getElementById("hero-status-text"),
        heroStatusDot: document.getElementById("hero-status-dot"),
        healthPanel: document.getElementById("health-panel"),
        answerPlaceholder: document.getElementById("answer-placeholder"),
        answerRich: document.getElementById("answer-rich"),
        answerMeta: document.getElementById("answer-meta"),
        answerText: document.getElementById("answer-text"),
        sourceList: document.getElementById("source-list"),
        conversationList: document.getElementById("conversation-list"),
        detailResultWrap: document.getElementById("detail-result-wrap"),
        detailResult: document.getElementById("detail-result"),
        sessionPill: document.getElementById("session-pill"),
        logLocked: document.getElementById("log-locked"),
        logDashboard: document.getElementById("log-dashboard"),
        logAccessPill: document.getElementById("log-access-pill"),
        logLimit: document.getElementById("log-limit"),
        logStatus: document.getElementById("log-status"),
        logUsername: document.getElementById("log-username"),
        logKeyword: document.getElementById("log-keyword"),
        knowledgeScope: document.getElementById("knowledge-scope"),
        knowledgeList: document.getElementById("knowledge-list"),
        knowledgeAccessPill: document.getElementById("knowledge-access-pill"),
        knowledgeVectorPill: document.getElementById("knowledge-vector-pill"),
        knowledgeTypeBadges: document.getElementById("knowledge-type-badges"),
        knowledgeCategoryBadges: document.getElementById("knowledge-category-badges"),
        userAdminAccessPill: document.getElementById("user-admin-access-pill"),
        userAdminLocked: document.getElementById("user-admin-locked"),
        userAdminDashboard: document.getElementById("user-admin-dashboard"),
        userStatusFilter: document.getElementById("user-status-filter"),
        userRoleFilter: document.getElementById("user-role-filter"),
        userSearch: document.getElementById("user-search"),
        adminUserList: document.getElementById("admin-user-list"),
    };

    function setBusy(button, busy, busyText, idleText) {
        if (!button) return;
        button.disabled = busy;
        button.textContent = busy ? busyText : idleText;
    }

    function clearAuthAndGoToLogin() {
        state.token = "";
        state.profile = null;
        sessionStorage.removeItem(TOKEN_KEY);
        window.location.replace("/");
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
        if (!value) return "未知时间";
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) return String(value);
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
        if (!Number.isFinite(ms)) return "0 ms";
        if (ms < 1000) return `${ms.toFixed(0)} ms`;
        return `${(ms / 1000).toFixed(2)} s`;
    }

    function formatBytes(value) {
        const size = Number(value || 0);
        if (!Number.isFinite(size) || size <= 0) return "0 B";
        if (size < 1024) return `${size} B`;
        if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
        return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    }

    function shortText(value, limit = 96) {
        const text = String(value || "");
        return text.length > limit ? `${text.slice(0, limit)}...` : text;
    }

    function statusClass(status) {
        if (status === "approved" || status === "success") return "status-approved";
        if (status === "pending") return "status-pending";
        if (status === "rejected") return "status-rejected";
        if (status === "disabled" || status === "failed") return "status-disabled";
        return "";
    }

    function clearAnswerPanels() {
        el.answerMeta.innerHTML = "";
        el.answerText.textContent = "";
        el.sourceList.innerHTML = "";
        el.conversationList.innerHTML = "";
        el.detailResult.textContent = "";
        el.detailResultWrap.classList.add("hidden");
    }

    function setAnswerNotice(message) {
        el.answerPlaceholder.textContent = message;
        el.answerPlaceholder.classList.remove("hidden");
        el.answerRich.classList.add("hidden");
        clearAnswerPanels();
    }

    function updateSessionPill() {
        el.sessionPill.textContent = state.sessionId ? `会话 ${state.sessionId}` : "新会话未开始";
    }

    function updateDetailButton() {
        el.detailToggle.textContent = state.detailed ? "当前为详细结果" : "当前为普通结果";
    }

    function resetConversationState(message) {
        state.sessionId = "";
        state.conversationHistory = [];
        state.lastAnswer = null;
        updateSessionPill();
        setAnswerNotice(message);
    }

    function renderHealth(data) {
        if (!data) return;
        state.health = data;
        el.heroStatusText.textContent = `系统状态：${data.status}`;
        el.heroStatusDot.style.background = data.status === "ok" ? "var(--success)" : "var(--warning)";

        document.getElementById("metric-documents").textContent = data.documents_count;
        document.getElementById("metric-vector").textContent = data.vector_store_ready ? "已就绪" : "未就绪";
        document.getElementById("metric-embedding").textContent = data.embedding_model || "未配置";
        document.getElementById("metric-chat").textContent = data.chat_configured ? "正常" : "缺失";

        const warnings = Array.isArray(data.warnings) && data.warnings.length
            ? data.warnings.map((item) => `- ${item}`).join("\n")
            : "当前没有额外风险提醒。";
        const files = Array.isArray(data.document_files) && data.document_files.length
            ? data.document_files.join("、")
            : "暂无文档";

        el.healthPanel.textContent = [
            `文档数量：${data.documents_count}`,
            `向量索引：${data.vector_store_ready ? "已构建" : "未构建"}`,
            `Embedding 接口：${data.embedding_base_url || "未配置"}`,
            `文档清单：${files}`,
            "",
            "提醒：",
            warnings,
        ].join("\n");
    }

    function renderProfile() {
        if (!state.profile) {
            el.profilePanel.textContent = "尚未登录。";
            el.permissionBadges.innerHTML = "";
            el.logAccessPill.textContent = "日志权限未激活";
            el.knowledgeAccessPill.textContent = "登录后读取知识库范围";
            el.userAdminAccessPill.textContent = "管理员权限未激活";
            el.rebuildKnowledgeButton.disabled = true;
            return;
        }

        const p = state.profile;
        el.profilePanel.textContent = [
            `当前用户：${p.username}`,
            `显示名称：${p.display_name}`,
            `审核状态：${p.status}`,
            `角色：${(p.roles || []).join("、") || "无"}`,
            `权限：${(p.permissions || []).join("、") || "无"}`,
            `日志权限：${p.can_view_logs ? "已开启" : "未开启"}`,
            `用户管理权限：${p.can_manage_users ? "已开启" : "未开启"}`,
        ].join("\n");

        el.permissionBadges.innerHTML = p.permissions.length
            ? p.permissions.map((item) => `<span class="mini-pill">${escapeHtml(item)}</span>`).join("")
            : '<span class="mini-pill">暂无权限</span>';

        el.logAccessPill.textContent = p.can_view_logs ? "日志权限已激活" : "当前账号无日志权限";
        el.knowledgeAccessPill.textContent = p.can_manage_knowledge_base
            ? "可查看全部文档并重建索引"
            : "仅展示当前账号可检索范围";
        el.userAdminAccessPill.textContent = p.can_manage_users
            ? "用户管理权限已激活"
            : "当前账号无用户管理权限";
        el.rebuildKnowledgeButton.disabled = !p.can_manage_knowledge_base;
    }

    function renderLogAccess() {
        const canViewLogs = Boolean(state.profile && state.profile.can_view_logs);
        el.logLocked.classList.toggle("hidden", canViewLogs);
        el.logDashboard.classList.toggle("hidden", !canViewLogs);
    }

    function renderLogs() {
        renderLogAccess();
        if (!(state.profile && state.profile.can_view_logs)) return;

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
                <div class="stack">
                    <div class="knowledge-meta"><span>${escapeHtml(username)}</span><strong>${count}</strong></div>
                    <div class="bar-track"><div class="bar-fill" style="width:${maxUserCount ? (count / maxUserCount) * 100 : 0}%"></div></div>
                </div>
            `).join("")
            : '<div class="empty-state">暂无可视化数据。</div>';

        const statusChart = document.getElementById("status-chart");
        statusChart.innerHTML = total
            ? `
                <div class="stack">
                    <div class="knowledge-meta"><span>success</span><strong>${successCount}</strong></div>
                    <div class="bar-track"><div class="bar-fill success" style="width:${(successCount / total) * 100}%"></div></div>
                </div>
                <div class="stack">
                    <div class="knowledge-meta"><span>failed</span><strong>${failedCount}</strong></div>
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

    function renderKnowledgeBadges(container, mapping, formatter) {
        const entries = Object.entries(mapping || {});
        container.innerHTML = entries.length
            ? entries.map(([key, count]) => `<span class="mini-pill">${escapeHtml(formatter(key, count))}</span>`).join("")
            : '<span class="mini-pill">暂无数据</span>';
    }

    function renderKnowledgeBase(data) {
        state.knowledgeBase = data;
        document.getElementById("knowledge-total").textContent = data?.total_documents ?? 0;
        document.getElementById("knowledge-accessible").textContent = data?.accessible_documents ?? 0;
        document.getElementById("knowledge-restricted").textContent = data?.restricted_documents ?? 0;

        const allowedPermissions = Array.isArray(data?.allowed_permissions) && data.allowed_permissions.length
            ? data.allowed_permissions.join("、")
            : "无";

        const scopeLines = [
            `允许检索的权限范围：${allowedPermissions}`,
            `当前可见文档：${data?.accessible_documents ?? 0} / ${data?.total_documents ?? 0}`,
            `受限文档：${data?.restricted_documents ?? 0}`,
            `支持格式：${Array.isArray(data?.supported_types) ? data.supported_types.join(" / ") : "未知"}`,
        ];
        if (Array.isArray(data?.parse_warnings) && data.parse_warnings.length) {
            scopeLines.push("", "解析提醒：");
            scopeLines.push(...data.parse_warnings.map((item) => `- ${item}`));
        }
        el.knowledgeScope.textContent = scopeLines.join("\n");
        el.knowledgeVectorPill.textContent = data?.vector_store_ready ? "索引已构建" : "索引未构建";

        renderKnowledgeBadges(el.knowledgeTypeBadges, data?.documents_by_type, (key, count) => `${String(key).toUpperCase()} ${count} 份`);
        renderKnowledgeBadges(el.knowledgeCategoryBadges, data?.documents_by_category, (key, count) => `${key} ${count} 份`);

        const items = Array.isArray(data?.items) ? data.items : [];
        el.knowledgeList.innerHTML = items.length
            ? items.map((item) => `
                <article class="knowledge-card">
                    <div class="knowledge-top">
                        <strong>${escapeHtml(item.filename || "未命名文档")}</strong>
                        <span class="mini-pill">${escapeHtml((item.file_type || "unknown").toUpperCase())}</span>
                        <span class="mini-pill ${item.accessible ? "status-success" : "status-disabled"}">${item.accessible ? "当前账号可检索" : "当前账号不可检索"}</span>
                    </div>
                    <div class="knowledge-path">${escapeHtml(item.relative_path || "")}</div>
                    <div class="knowledge-meta">
                        <span>目录：${escapeHtml(item.category || "根目录")}</span>
                        <span>权限：${escapeHtml(item.permission_label || item.required_permission || "未知")}</span>
                        <span>大小：${escapeHtml(formatBytes(item.size_bytes))}</span>
                        <span>更新：${escapeHtml(formatTimestamp(item.updated_at ? item.updated_at * 1000 : item.updated_at))}</span>
                    </div>
                </article>
            `).join("")
            : '<div class="empty-state">当前没有可展示的知识库文档。</div>';
    }

    function renderSources(sources) {
        el.sourceList.innerHTML = Array.isArray(sources) && sources.length
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
        el.conversationList.innerHTML = Array.isArray(history) && history.length
            ? history.map((item) => `
                <article class="conversation-item">
                    <div class="source-title">
                        <span class="conversation-role">${item.role === "assistant" ? "助手" : "用户"}</span>
                        <span class="muted">${escapeHtml(formatTimestamp(item.timestamp))}</span>
                    </div>
                    <div>${escapeHtml(item.content || "")}</div>
                </article>
            `).join("")
            : '<div class="empty-state">当前会话还没有历史记录。</div>';
    }

    function renderDetailedResult(data) {
        if (state.detailed && data && data.detailed_result) {
            el.detailResultWrap.classList.remove("hidden");
            el.detailResult.textContent = JSON.stringify(data.detailed_result, null, 2);
            return;
        }
        el.detailResultWrap.classList.add("hidden");
        el.detailResult.textContent = "";
    }

    function renderRichAnswer(data) {
        state.lastAnswer = data;
        state.sessionId = data.session_id || state.sessionId;
        state.conversationHistory = Array.isArray(data.history) ? data.history : [];

        updateSessionPill();
        el.answerPlaceholder.classList.add("hidden");
        el.answerRich.classList.remove("hidden");

        const sources = Array.isArray(data.sources) ? data.sources : [];
        el.answerMeta.innerHTML = [
            `<span class="mini-pill">会话 ${escapeHtml(state.sessionId || "未分配")}</span>`,
            `<span class="mini-pill">耗时 ${escapeHtml(formatDuration(data.execution_time))}</span>`,
            `<span class="mini-pill">来源 ${sources.length} 份</span>`,
            `<span class="mini-pill">${state.detailed ? "详细模式" : "普通模式"}</span>`,
        ].join("");

        el.answerText.textContent = data.answer || "系统未返回答案。";
        renderSources(sources);
        renderConversation(state.conversationHistory);
        renderDetailedResult(data);
    }

    function renderUserAdminAccess() {
        const canManageUsers = Boolean(state.profile && state.profile.can_manage_users);
        el.userAdminLocked.classList.toggle("hidden", canManageUsers);
        el.userAdminDashboard.classList.toggle("hidden", !canManageUsers);
    }

    function populateRoleFilterOptions() {
        const entries = Object.entries(state.roles || {});
        el.userRoleFilter.innerHTML = entries.length
            ? entries.map(([key, role]) => `<option value="${escapeHtml(key)}">${escapeHtml(key)} · ${escapeHtml(role.label || key)}</option>`).join("")
            : '<option value="employee">employee</option>';
        if (!el.userRoleFilter.value) {
            el.userRoleFilter.value = entries.some(([key]) => key === "employee") ? "employee" : (entries[0]?.[0] || "employee");
        }
    }

    function buildRoleOptions(selectedRoles) {
        const entries = Object.entries(state.roles || {});
        const selected = Array.isArray(selectedRoles) && selectedRoles.length ? selectedRoles[0] : "";
        return entries.map(([key, role]) => `<option value="${escapeHtml(key)}" ${key === selected ? "selected" : ""}>${escapeHtml(key)} · ${escapeHtml(role.label || key)}</option>`).join("");
    }

    function renderAdminUsers() {
        renderUserAdminAccess();
        if (!(state.profile && state.profile.can_manage_users)) return;

        const users = Array.isArray(state.adminUsers) ? state.adminUsers : [];
        document.getElementById("pending-count").textContent = users.filter((item) => item.status === "pending").length;
        document.getElementById("approved-count").textContent = users.filter((item) => item.status === "approved").length;
        document.getElementById("rejected-count").textContent = users.filter((item) => item.status === "rejected").length;
        document.getElementById("disabled-count").textContent = users.filter((item) => item.status === "disabled").length;

        populateRoleFilterOptions();

        const statusFilter = el.userStatusFilter.value.trim();
        const keyword = el.userSearch.value.trim().toLowerCase();
        const filtered = users.filter((item) => {
            if (statusFilter && item.status !== statusFilter) return false;
            if (!keyword) return true;
            const haystack = `${item.username || ""} ${item.display_name || ""}`.toLowerCase();
            return haystack.includes(keyword);
        });

        el.adminUserList.innerHTML = filtered.length
            ? filtered.map((item) => `
                <article class="admin-user-card" data-username="${escapeHtml(item.username || "")}">
                    <div class="admin-user-header">
                        <strong>${escapeHtml(item.username || "unknown")}</strong>
                        <span class="mini-pill ${statusClass(item.status)}">${escapeHtml(item.status || "unknown")}</span>
                    </div>
                    <div class="admin-user-meta">
                        <span>显示名：${escapeHtml(item.display_name || item.username || "-")}</span>
                        <span>角色：${escapeHtml((item.roles || []).join("、") || "未分配")}</span>
                        <span>创建：${escapeHtml(formatTimestamp(item.created_at))}</span>
                    </div>
                    <div class="inline-grid">
                        <label>分配角色
                            <select class="user-role-select">${buildRoleOptions(item.roles || [])}</select>
                        </label>
                        <label>审核备注
                            <input class="review-note-input" placeholder="可选：填写审核说明" />
                        </label>
                    </div>
                    <div class="button-row">
                        <button class="secondary-button approve-user" type="button">审批通过</button>
                        <button class="danger-button reject-user" type="button">驳回</button>
                        <button class="neutral-button disable-user" type="button">停用</button>
                        <button class="neutral-button update-role-button" type="button">更新角色</button>
                    </div>
                </article>
            `).join("")
            : '<div class="empty-state">没有符合筛选条件的用户。</div>';
    }

    async function apiRequest(path, options = {}) {
        const headers = new Headers(options.headers || {});
        if (state.token && !headers.has("Authorization")) {
            headers.set("Authorization", `Bearer ${state.token}`);
        }
        const response = await fetch(path, { ...options, headers });
        let data = null;
        try {
            data = await response.json();
        } catch {
            data = null;
        }
        if (!response.ok) throw new Error(data?.detail || `请求失败（${response.status}）`);
        return data;
    }

    const apiGet = (path) => apiRequest(path);
    const apiDelete = (path) => apiRequest(path, { method: "DELETE" });
    const apiPost = (path, payload) => apiRequest(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload ? JSON.stringify(payload) : null,
    });
    const apiPatch = (path, payload) => apiRequest(path, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    async function fetchHealth() {
        setBusy(el.refreshHealthButton, true, "刷新中...", "刷新状态");
        try {
            renderHealth(await apiGet("/health"));
        } catch (error) {
            el.healthPanel.textContent = `状态读取失败：${error.message}`;
        } finally {
            setBusy(el.refreshHealthButton, false, "刷新中...", "刷新状态");
        }
    }

    async function fetchKnowledgeBase() {
        if (!state.token) {
            el.knowledgeScope.textContent = "登录后显示当前账号的检索边界。";
            el.knowledgeList.innerHTML = '<div class="empty-state">登录后显示知识库文档清单。</div>';
            el.knowledgeTypeBadges.innerHTML = '<span class="mini-pill">等待登录</span>';
            el.knowledgeCategoryBadges.innerHTML = '<span class="mini-pill">等待登录</span>';
            el.knowledgeVectorPill.textContent = "索引状态读取中";
            document.getElementById("knowledge-total").textContent = "0";
            document.getElementById("knowledge-accessible").textContent = "0";
            document.getElementById("knowledge-restricted").textContent = "0";
            return;
        }
        setBusy(el.refreshKnowledgeButton, true, "刷新中...", "刷新知识库");
        try {
            renderKnowledgeBase(await apiGet("/knowledge-base"));
        } catch (error) {
            el.knowledgeScope.textContent = `知识库读取失败：${error.message}`;
            el.knowledgeList.innerHTML = `<div class="empty-state">知识库读取失败：${escapeHtml(error.message)}</div>`;
        } finally {
            setBusy(el.refreshKnowledgeButton, false, "刷新中...", "刷新知识库");
        }
    }

    async function rebuildKnowledgeBase() {
        if (!state.token) {
            el.knowledgeScope.textContent = "请先登录后再重建索引。";
            return;
        }
        if (!(state.profile && state.profile.can_manage_knowledge_base)) {
            el.knowledgeScope.textContent = "当前账号没有重建索引权限。";
            return;
        }
        setBusy(el.rebuildKnowledgeButton, true, "重建中...", "重建索引");
        try {
            renderKnowledgeBase(await apiPost("/knowledge-base/rebuild"));
            renderHealth(await apiGet("/health"));
        } catch (error) {
            el.knowledgeScope.textContent = `重建索引失败：${error.message}`;
        } finally {
            setBusy(el.rebuildKnowledgeButton, false, "重建中...", "重建索引");
        }
    }

    async function fetchLogs() {
        if (!(state.token && state.profile && state.profile.can_view_logs)) {
            renderLogAccess();
            return;
        }
        const params = new URLSearchParams();
        params.set("limit", el.logLimit.value);
        if (el.logStatus.value.trim()) params.set("status_filter", el.logStatus.value.trim());
        if (el.logUsername.value.trim()) params.set("username", el.logUsername.value.trim());
        if (el.logKeyword.value.trim()) params.set("keyword", el.logKeyword.value.trim());

        setBusy(el.refreshLogsButton, true, "加载中...", "刷新日志");
        try {
            state.logs = await apiGet(`/logs?${params.toString()}`);
            renderLogs();
        } catch (error) {
            el.logDashboard.classList.remove("hidden");
            el.logLocked.classList.add("hidden");
            document.getElementById("timeline-list").innerHTML = `<div class="empty-state">日志加载失败：${escapeHtml(error.message)}</div>`;
        } finally {
            setBusy(el.refreshLogsButton, false, "加载中...", "刷新日志");
        }
    }

    async function fetchAdminUsers() {
        if (!(state.token && state.profile && state.profile.can_manage_users)) {
            renderUserAdminAccess();
            return;
        }
        setBusy(el.refreshUsersButton, true, "刷新中...", "刷新用户列表");
        try {
            const data = await apiGet("/admin/users");
            state.adminUsers = Array.isArray(data.users) ? data.users : [];
            state.roles = data.roles || {};
            renderAdminUsers();
        } catch (error) {
            el.adminUserList.innerHTML = `<div class="empty-state">用户列表加载失败：${escapeHtml(error.message)}</div>`;
        } finally {
            setBusy(el.refreshUsersButton, false, "刷新中...", "刷新用户列表");
        }
    }

    async function fetchProfile() {
        if (!state.token) {
            el.profilePanel.textContent = "请先登录。";
            return;
        }
        setBusy(el.profileButton, true, "读取中...", "刷新身份信息");
        try {
            state.profile = await apiGet("/me");
            renderProfile();
            renderLogAccess();
            renderUserAdminAccess();
            await Promise.all([fetchKnowledgeBase(), fetchLogs(), fetchAdminUsers()]);
        } catch (error) {
            if (/Could not validate credentials|not approved|401|403/i.test(String(error.message || ""))) {
                clearAuthAndGoToLogin();
                return;
            }
            el.profilePanel.textContent = `身份读取失败：${error.message}`;
        } finally {
            setBusy(el.profileButton, false, "读取中...", "刷新身份信息");
        }
    }

    function logout() {
        clearAuthAndGoToLogin();
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

        setBusy(el.questionButton, true, "提问中...", "开始提问");
        setAnswerNotice("系统正在处理你的问题，请稍候...");
        try {
            const data = await apiPost("/question", {
                input: question,
                detailed: state.detailed,
                return_rich_response: true,
                session_id: state.sessionId || null,
            });
            renderRichAnswer(data);
            await fetchLogs();
        } catch (error) {
            setAnswerNotice(`问答失败：${error.message}`);
            await fetchLogs();
        } finally {
            setBusy(el.questionButton, false, "提问中...", "开始提问");
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
        setBusy(el.clearSessionButton, true, "清空中...", "清空会话");
        try {
            await apiDelete(`/conversation/${encodeURIComponent(state.sessionId)}`);
            resetConversationState("会话已清空，下一次提问会开始新的上下文。");
        } catch (error) {
            setAnswerNotice(`清空会话失败：${error.message}`);
        } finally {
            setBusy(el.clearSessionButton, false, "清空中...", "清空会话");
        }
    }

    async function reviewUser(username, action, roles, note) {
        await apiPatch(`/admin/users/${encodeURIComponent(username)}/review`, {
            action,
            roles,
            review_note: note || "",
        });
        await fetchAdminUsers();
    }

    async function updateUserRoles(username, roles) {
        await apiPatch(`/admin/users/${encodeURIComponent(username)}/roles`, { roles });
        await fetchAdminUsers();
    }

    el.adminUserList.addEventListener("click", async (event) => {
        const card = event.target.closest(".admin-user-card");
        if (!card) return;

        const username = card.dataset.username;
        const roleSelect = card.querySelector(".user-role-select");
        const noteInput = card.querySelector(".review-note-input");
        const selectedRole = roleSelect ? roleSelect.value : (el.userRoleFilter.value || "employee");
        const note = noteInput ? noteInput.value.trim() : "";

        try {
            if (event.target.classList.contains("approve-user")) {
                await reviewUser(username, "approve", [selectedRole || "employee"], note);
            } else if (event.target.classList.contains("reject-user")) {
                await reviewUser(username, "reject", [], note);
            } else if (event.target.classList.contains("disable-user")) {
                await reviewUser(username, "disable", [], note);
            } else if (event.target.classList.contains("update-role-button")) {
                await updateUserRoles(username, [selectedRole || "employee"]);
            }
        } catch (error) {
            el.adminUserList.insertAdjacentHTML("afterbegin", `<div class="empty-state">操作失败：${escapeHtml(error.message)}</div>`);
        }
    });

    el.userStatusFilter.addEventListener("change", renderAdminUsers);
    el.userSearch.addEventListener("input", renderAdminUsers);
    el.userRoleFilter.addEventListener("change", renderAdminUsers);

    el.questionForm.addEventListener("submit", askQuestion);
    el.clearSessionButton.addEventListener("click", clearConversation);

    el.clearAnswerButton.addEventListener("click", () => {
        state.lastAnswer = null;
        setAnswerNotice(state.sessionId ? "结果已清空，本轮会话上下文仍保留，可继续追问。" : "结果已清空。");
        updateSessionPill();
    });

    el.detailToggle.addEventListener("click", () => {
        state.detailed = !state.detailed;
        updateDetailButton();
        if (state.lastAnswer) {
            renderRichAnswer(state.lastAnswer);
        } else {
            setAnswerNotice(state.detailed ? "详细链路模式已开启。" : "已切换为普通结果模式。");
        }
    });

    el.refreshLogsButton.addEventListener("click", fetchLogs);
    el.refreshHealthButton.addEventListener("click", fetchHealth);
    el.refreshKnowledgeButton.addEventListener("click", fetchKnowledgeBase);
    el.rebuildKnowledgeButton.addEventListener("click", rebuildKnowledgeBase);
    el.refreshUsersButton.addEventListener("click", fetchAdminUsers);
    el.profileButton.addEventListener("click", fetchProfile);
    if (el.logoutButton) {
        el.logoutButton.addEventListener("click", logout);
    }

    el.logLimit.addEventListener("change", fetchLogs);
    el.logStatus.addEventListener("change", fetchLogs);
    el.logUsername.addEventListener("change", fetchLogs);
    el.logKeyword.addEventListener("change", fetchLogs);

    document.querySelectorAll(".sample-question").forEach((button) => {
        button.addEventListener("click", () => {
            document.getElementById("question-input").value = button.textContent.trim();
        });
    });

    updateDetailButton();
    updateSessionPill();
    renderHealth(state.health);
    renderProfile();
    renderLogAccess();
    renderUserAdminAccess();
    if (!state.token) {
        clearAuthAndGoToLogin();
        return;
    }
    fetchProfile();
    setAnswerNotice("已登录，可开始提问。");
})();

