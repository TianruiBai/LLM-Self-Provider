"use strict";
/**
 * Account modal — profile, password, API keys, 2FA, admin pane.
 * Mounted by index.html as a defer script. Exposes window.openAccountModal().
 */
(function () {
  const $ = (s, root = document) => root.querySelector(s);

  // -------------------------------------------------------- 401 redirect
  // Wrap fetch so any 401 from /auth/* | /v1/* | /admin/* | /rag/* | /events
  // bounces the user to the login screen instead of leaving the UI broken.
  const origFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const r = await origFetch(input, init);
    if (r.status === 401) {
      const url = typeof input === "string" ? input : (input.url || "");
      if (/^\/(auth|v1|admin|rag|events)/.test(url) && url !== "/auth/me-anonymous"
          && url !== "/auth/login" && url !== "/auth/bootstrap") {
        // Avoid redirect storm during initial probe.
        if (!sessionStorage.getItem("__redirected_to_login")) {
          sessionStorage.setItem("__redirected_to_login", "1");
          window.location.replace("/ui/login.html");
        }
      }
    } else if (r.ok) {
      sessionStorage.removeItem("__redirected_to_login");
    }
    return r;
  };

  // -------------------------------------------------------- DOM
  function ensureModal() {
    if ($("#acct-modal")) return;
    const el = document.createElement("div");
    el.id = "acct-modal";
    el.className = "modal-backdrop";
    el.hidden = true;
    el.innerHTML = `
      <div class="modal acct-modal">
        <header class="modal-head">
          <h2>Account</h2>
          <button class="ghost icon-btn" id="acct-close" type="button" title="Close"><span class="mi">close</span></button>
        </header>
        <nav class="acct-tabs">
          <button class="acct-tab active" data-acct="profile">Profile</button>
          <button class="acct-tab" data-acct="keys">API keys</button>
          <button class="acct-tab" data-acct="2fa">2FA</button>
          <button class="acct-tab" data-acct="admin" id="acct-tab-admin" hidden>Admin</button>
          <button class="acct-tab" data-acct="models" id="acct-tab-models" hidden>Models</button>
        </nav>
        <div class="acct-body">
          <!-- profile -->
          <section class="acct-pane active" data-pane="profile">
            <div class="acct-row"><div class="lbl">Username</div><div class="val" id="acct-username">—</div></div>
            <div class="acct-row"><div class="lbl">Email</div><div class="val" id="acct-email">—</div></div>
            <div class="acct-row"><div class="lbl">Role</div><div class="val" id="acct-role">—</div></div>
            <div class="acct-row"><div class="lbl">Auth via</div><div class="val" id="acct-via">—</div></div>
            <h3>Change password</h3>
            <div class="acct-grid">
              <input id="acct-cpw" type="password" placeholder="Current password" />
              <input id="acct-npw" type="password" placeholder="New password (min 8 chars)" />
              <button class="primary" id="acct-cpw-go" type="button">Update password</button>
            </div>
            <div class="acct-msg" id="acct-cpw-msg"></div>
            <h3 style="margin-top:16px">Session</h3>
            <button class="ghost" id="acct-logout" type="button"><span class="mi">logout</span> Sign out</button>
          </section>

          <!-- API keys -->
          <section class="acct-pane" data-pane="keys">
            <p class="hint">API keys let OpenAI-compatible clients (Continue, Cline, curl…) talk to this provider.
              Use them as <code>Authorization: Bearer sk-prov-…</code>. The full key is shown <strong>once</strong> at creation; only the masked prefix is stored after that.</p>
            <div class="acct-grid">
              <input id="acct-key-name" type="text" placeholder="Key name (e.g. 'laptop continue')" />
              <input id="acct-key-cidr" type="text" placeholder="IP allowlist (CIDR, optional, e.g. 10.0.0.0/8)" />
              <button class="primary" id="acct-key-new" type="button">+ New key</button>
            </div>
            <div class="acct-key-newval" id="acct-key-newval" hidden></div>
            <div class="acct-keys" id="acct-keys"></div>
          </section>

          <!-- 2FA -->
          <section class="acct-pane" data-pane="2fa">
            <div id="acct-2fa-off" hidden>
              <p class="hint">Two-factor authentication is currently <strong>off</strong>. Enrol an authenticator app (Google Authenticator, 1Password, Aegis…) below.</p>
              <button class="primary" id="acct-2fa-begin" type="button">Begin 2FA enrollment</button>
              <div id="acct-2fa-enroll" hidden style="margin-top:12px">
                <div class="acct-row"><div class="lbl">Secret (base32)</div><div class="val mono" id="acct-2fa-secret">—</div></div>
                <div class="acct-row"><div class="lbl">Or scan</div><div class="val"><a id="acct-2fa-link" target="_blank" rel="noopener">otpauth URL</a></div></div>
                <div class="acct-grid" style="margin-top:8px">
                  <input id="acct-2fa-code" type="text" inputmode="numeric" placeholder="6-digit code" maxlength="8" />
                  <button class="primary" id="acct-2fa-finish" type="button">Activate 2FA</button>
                </div>
              </div>
            </div>
            <div id="acct-2fa-on" hidden>
              <p class="hint">Two-factor authentication is <strong>enabled</strong> on this account.</p>
              <h3>Recovery codes</h3>
              <p class="hint">Each code can be used once if you lose your authenticator. Generating new codes invalidates the old set.</p>
              <button class="ghost" id="acct-2fa-recovery" type="button">Generate new recovery codes</button>
              <pre class="acct-recovery" id="acct-recovery-out" hidden></pre>
              <h3 style="margin-top:16px">Disable 2FA</h3>
              <div class="acct-grid">
                <input id="acct-2fa-disable-pw" type="password" placeholder="Current password" />
                <button class="ghost danger" id="acct-2fa-disable" type="button">Disable 2FA</button>
              </div>
            </div>
            <div class="acct-msg" id="acct-2fa-msg"></div>
          </section>

          <!-- admin -->
          <section class="acct-pane" data-pane="admin">
            <h3>Users</h3>
            <div class="acct-admin-actions">
              <button class="ghost" id="acct-admin-newuser" type="button">+ Add user</button>
              <button class="ghost icon-btn" id="acct-admin-refresh" type="button" title="Refresh"><span class="mi">refresh</span></button>
            </div>
            <table class="acct-table" id="acct-users"></table>
            <h3 style="margin-top:16px">All API keys (masked)</h3>
            <table class="acct-table" id="acct-allkeys"></table>
            <h3 style="margin-top:16px">Active sessions</h3>
            <table class="acct-table" id="acct-sessions"></table>
            <h3 style="margin-top:16px">Request audit</h3>
            <div class="acct-admin-actions">
              <select id="acct-audit-window">
                <option value="3600">last hour</option>
                <option value="86400" selected>last 24h</option>
                <option value="604800">last 7d</option>
                <option value="">all</option>
              </select>
              <input id="acct-audit-path" type="text" placeholder="path prefix (e.g. /v1/chat)" />
              <input id="acct-audit-ip"   type="text" placeholder="IP" style="max-width:140px" />
              <input id="acct-audit-status" type="text" placeholder="status (e.g. 401)" style="max-width:120px" />
              <button class="ghost" id="acct-audit-refresh" type="button">Apply</button>
              <span class="hint" id="acct-audit-meta" style="margin-left:auto"></span>
            </div>
            <div id="acct-audit-summary" class="acct-audit-summary"></div>
            <table class="acct-table" id="acct-audit"></table>
            <div class="acct-admin-actions" style="justify-content:flex-end">
              <button class="ghost" id="acct-audit-prev" type="button">‹ Prev</button>
              <button class="ghost" id="acct-audit-next" type="button">Next ›</button>
            </div>
          </section>

          <!-- models -->
          <section class="acct-pane" data-pane="models">
            <div class="acct-admin-actions">
              <span class="hint">Publish, label, and tune every discovered model. Changes take effect on the next swap.</span>
              <button class="ghost icon-btn" id="acct-models-refresh" type="button" title="Refresh"><span class="mi">refresh</span></button>
            </div>
            <div id="acct-models-list"></div>
          </section>
        </div>
      </div>`;
    document.body.appendChild(el);

    // close handlers
    el.addEventListener("click", (ev) => { if (ev.target === el) closeModal(); });
    $("#acct-close").addEventListener("click", closeModal);

    // tab switching
    el.querySelectorAll(".acct-tab").forEach(b => b.addEventListener("click", () => {
      el.querySelectorAll(".acct-tab").forEach(x => x.classList.remove("active"));
      el.querySelectorAll(".acct-pane").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      const which = b.dataset.acct;
      $(`.acct-pane[data-pane="${which}"]`).classList.add("active");
      if (which === "keys")  loadKeys();
      if (which === "2fa")   loadProfile();
      if (which === "admin") loadAdmin();
      if (which === "models") loadModels();
    }));

    // wire actions
    $("#acct-cpw-go").addEventListener("click", changePassword);
    $("#acct-logout").addEventListener("click", logout);
    $("#acct-key-new").addEventListener("click", createKey);
    $("#acct-2fa-begin").addEventListener("click", beginTotp);
    $("#acct-2fa-finish").addEventListener("click", finishTotp);
    $("#acct-2fa-disable").addEventListener("click", disableTotp);
    $("#acct-2fa-recovery").addEventListener("click", regenRecovery);
    $("#acct-admin-refresh").addEventListener("click", loadAdmin);
    $("#acct-models-refresh").addEventListener("click", loadModels);
    $("#acct-admin-newuser").addEventListener("click", adminCreateUser);
    $("#acct-audit-refresh").addEventListener("click", () => { _audit.offset = 0; loadAudit(); });
    $("#acct-audit-prev").addEventListener("click", () => { _audit.offset = Math.max(0, _audit.offset - _audit.limit); loadAudit(); });
    $("#acct-audit-next").addEventListener("click", () => { _audit.offset += _audit.limit; loadAudit(); });
  }

  function openModal() { ensureModal(); $("#acct-modal").hidden = false; loadProfile(); }
  function closeModal() { const m = $("#acct-modal"); if (m) m.hidden = true; }
  window.openAccountModal = openModal;
  window.closeAccountModal = closeModal;

  // -------------------------------------------------------- profile
  let _me = null;
  async function loadProfile() {
    try {
      const r = await fetch("/auth/me");
      if (!r.ok) return;
      _me = (await r.json()).user;
      $("#acct-username").textContent = _me.username;
      $("#acct-email").textContent = _me.email || "—";
      $("#acct-role").textContent = _me.role;
      $("#acct-via").textContent = (await (await fetch("/auth/me")).json()).via || "—";
      $("#acct-tab-admin").hidden = _me.role !== "admin";
      $("#acct-tab-models").hidden = _me.role !== "admin";
      $("#acct-2fa-off").hidden = !!_me.totp_enabled;
      $("#acct-2fa-on").hidden  = !_me.totp_enabled;
      // Update topbar pill if it exists.
      const pill = $("#acct-pill-name"); if (pill) pill.textContent = _me.username;
      const role = $("#acct-pill-role"); if (role) role.textContent = _me.role;
    } catch {}
  }

  async function changePassword() {
    const cur = $("#acct-cpw").value, nxt = $("#acct-npw").value;
    if (!cur || nxt.length < 8) { msg("acct-cpw-msg", "Need current password + 8-char new password.", true); return; }
    const r = await fetch("/auth/change-password", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({current_password: cur, new_password: nxt}),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { msg("acct-cpw-msg", j.detail || "Failed", true); return; }
    msg("acct-cpw-msg", "Password updated.", false);
    $("#acct-cpw").value = ""; $("#acct-npw").value = "";
  }

  async function logout() {
    await fetch("/auth/logout", {method:"POST"}).catch(()=>{});
    window.location.replace("/ui/login.html");
  }

  // -------------------------------------------------------- API keys
  async function loadKeys() {
    const r = await fetch("/auth/keys");
    if (!r.ok) return;
    const { keys } = await r.json();
    renderKeyList(keys);
  }

  function renderKeyList(keys) {
    const el = $("#acct-keys");
    if (!keys.length) { el.innerHTML = `<div class="hint">No keys yet.</div>`; return; }
    el.innerHTML = keys.map(k => `
      <div class="acct-key-row" data-id="${k.id}">
        <div class="acct-key-main">
          <div class="acct-key-name">${escapeHtml(k.name)}</div>
          <div class="acct-key-mask mono">${escapeHtml(k.masked)}</div>
          <div class="acct-key-meta">
            created ${fmtTs(k.created_at)} ·
            ${k.last_used_at ? `last used ${fmtTs(k.last_used_at)} from ${escapeHtml(k.last_used_ip||"?")}` : "never used"}
            ${k.ip_allowlist ? ` · allow ${escapeHtml(k.ip_allowlist)}` : ""}
            ${k.revoked ? ` · <span class="danger">revoked</span>` : ""}
          </div>
        </div>
        <div class="acct-key-actions">
          ${k.revoked ? "" : `<button class="ghost danger" data-act="revoke" data-id="${k.id}" type="button">Revoke</button>`}
        </div>
      </div>`).join("");
    el.querySelectorAll('button[data-act="revoke"]').forEach(b =>
      b.addEventListener("click", () => revokeKey(b.dataset.id)));
  }

  async function createKey() {
    const name = $("#acct-key-name").value.trim() || "key";
    const cidr = $("#acct-key-cidr").value.trim() || null;
    const r = await fetch("/auth/keys", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name, ip_allowlist: cidr})
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { alert(j.detail || "Failed"); return; }
    const box = $("#acct-key-newval");
    box.hidden = false;
    box.innerHTML = `
      <div class="acct-key-newhdr">New key — copy now, you won't see it again:</div>
      <pre class="mono">${escapeHtml(j.plaintext)}</pre>
      <button class="ghost" id="acct-key-copy" type="button">Copy</button>
    `;
    $("#acct-key-copy").addEventListener("click", () => navigator.clipboard.writeText(j.plaintext));
    $("#acct-key-name").value = ""; $("#acct-key-cidr").value = "";
    loadKeys();
  }

  async function revokeKey(id) {
    if (!confirm("Revoke this API key? Clients using it will start getting 401.")) return;
    const r = await fetch(`/auth/keys/${id}`, {method:"DELETE"});
    if (!r.ok) { alert("Failed"); return; }
    loadKeys();
  }

  // -------------------------------------------------------- TOTP
  async function beginTotp() {
    const r = await fetch("/auth/totp/begin", {method:"POST"});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { msg("acct-2fa-msg", j.detail || "Failed", true); return; }
    $("#acct-2fa-secret").textContent = j.secret_base32;
    $("#acct-2fa-link").href = j.otpauth_url;
    $("#acct-2fa-link").textContent = j.otpauth_url;
    $("#acct-2fa-enroll").hidden = false;
    msg("acct-2fa-msg", "Add this secret to your authenticator app, then enter a code below.", false);
  }
  async function finishTotp() {
    const code = $("#acct-2fa-code").value.trim();
    const r = await fetch("/auth/totp/finish", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({code})
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { msg("acct-2fa-msg", j.detail || "Bad code", true); return; }
    msg("acct-2fa-msg", "2FA activated.", false);
    await loadProfile();
  }
  async function disableTotp() {
    const pw = $("#acct-2fa-disable-pw").value;
    if (!pw) { msg("acct-2fa-msg", "Confirm with your password.", true); return; }
    const r = await fetch("/auth/totp/disable", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({current_password: pw, new_password: "x".repeat(8)})
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { msg("acct-2fa-msg", j.detail || "Failed", true); return; }
    msg("acct-2fa-msg", "2FA disabled.", false);
    $("#acct-2fa-disable-pw").value = "";
    await loadProfile();
  }
  async function regenRecovery() {
    if (!confirm("Generate new recovery codes? Old codes will stop working.")) return;
    const r = await fetch("/auth/recovery-codes", {method:"POST"});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { msg("acct-2fa-msg", j.detail || "Failed", true); return; }
    const out = $("#acct-recovery-out");
    out.hidden = false;
    out.textContent = j.codes.join("\n");
  }

  // -------------------------------------------------------- admin
  async function loadAdmin() {
    if (!_me || _me.role !== "admin") return;
    await Promise.all([loadAdminUsers(), loadAdminKeys(), loadAdminSessions(), loadAudit(), loadAuditSummary()]);
  }

  // -------------------------------------------------------- audit
  const _audit = { limit: 100, offset: 0 };

  function _auditFilters() {
    const win = $("#acct-audit-window").value;
    const path = $("#acct-audit-path").value.trim();
    const ip = $("#acct-audit-ip").value.trim();
    const status = $("#acct-audit-status").value.trim();
    const params = new URLSearchParams();
    if (win) params.set("since", String(Math.floor(Date.now()/1000) - parseInt(win, 10)));
    if (path) params.set("path_prefix", path);
    if (ip) params.set("ip", ip);
    if (status) {
      const n = parseInt(status, 10);
      if (!isNaN(n)) { params.set("status_min", String(n)); params.set("status_max", String(n)); }
    }
    return params;
  }

  async function loadAudit() {
    const params = _auditFilters();
    params.set("limit",  String(_audit.limit));
    params.set("offset", String(_audit.offset));
    const r = await fetch("/auth/admin/audit?" + params.toString());
    if (!r.ok) return;
    const j = await r.json();
    const tbl = $("#acct-audit");
    tbl.innerHTML = `
      <thead><tr><th>Time</th><th>User</th><th>Key</th><th>IP</th><th>Method</th><th>Path</th><th>Status</th><th>ms</th><th>In/Out</th></tr></thead>
      <tbody>${j.rows.map(r => `
        <tr class="audit-${r.status >= 500 ? "err" : r.status >= 400 ? "warn" : "ok"}">
          <td class="mono" style="white-space:nowrap">${fmtTs(r.ts)}</td>
          <td>${escapeHtml(r.username || "")}</td>
          <td>${r.key_prefix ? `<span class="mono" title="${escapeHtml(r.key_name||"")}">${escapeHtml(r.key_prefix)}…</span>` : ""}</td>
          <td class="mono">${escapeHtml(r.ip || "")}</td>
          <td>${escapeHtml(r.method)}</td>
          <td class="mono" style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(r.path)}">${escapeHtml(r.path)}</td>
          <td>${r.status}</td>
          <td>${r.duration_ms}</td>
          <td class="mono" style="font-size:10.5px">${(r.bytes_in||0)}/${(r.bytes_out||0)}</td>
        </tr>`).join("")}
      </tbody>`;
    const meta = $("#acct-audit-meta");
    if (meta) {
      const start = j.total ? j.offset + 1 : 0;
      const end = Math.min(j.total, j.offset + j.rows.length);
      meta.textContent = `${start}–${end} of ${j.total}`;
    }
  }

  async function loadAuditSummary() {
    const params = new URLSearchParams();
    const win = $("#acct-audit-window").value;
    if (win) params.set("since", String(Math.floor(Date.now()/1000) - parseInt(win, 10)));
    const r = await fetch("/auth/admin/audit/summary?" + params.toString());
    if (!r.ok) return;
    const j = await r.json();
    const el = $("#acct-audit-summary");
    if (!el) return;
    const totalReq = j.by_status.reduce((s, x) => s + x.n, 0);
    const errs = j.by_status.filter(x => x.status >= 400).reduce((s, x) => s + x.n, 0);
    const topUsers = (j.by_user || []).slice(0, 5).map(u =>
      `<span class="acct-pillbox">${escapeHtml(u.username || "anon")} · ${u.n}</span>`).join(" ");
    el.innerHTML = `
      <div class="acct-summary-row">
        <div><strong>${totalReq}</strong> requests</div>
        <div>${errs} errors</div>
        <div class="top-users">${topUsers}</div>
      </div>`;
  }

  async function loadAdminUsers() {
    const r = await fetch("/auth/users");
    if (!r.ok) return;
    const { users } = await r.json();
    const tbl = $("#acct-users");
    tbl.innerHTML = `
      <thead><tr><th>ID</th><th>Username</th><th>Email</th><th>Role</th><th>Active</th><th>2FA</th><th>OIDC</th><th></th></tr></thead>
      <tbody>${users.map(u => `
        <tr data-id="${u.id}">
          <td>${u.id}</td>
          <td>${escapeHtml(u.username)}</td>
          <td>${escapeHtml(u.email||"")}</td>
          <td>
            <select data-act="role" ${u.id===_me.id?"disabled":""}>
              <option value="user"${u.role==="user"?" selected":""}>user</option>
              <option value="admin"${u.role==="admin"?" selected":""}>admin</option>
            </select>
          </td>
          <td>
            <input type="checkbox" data-act="active" ${u.is_active?"checked":""} ${u.id===_me.id?"disabled":""}/>
          </td>
          <td>${u.totp_enabled?"✓":""}</td>
          <td>${u.oidc_subject?escapeHtml(u.oidc_subject):""}</td>
          <td><button class="ghost" data-act="resetpw" data-id="${u.id}">Reset password</button></td>
        </tr>`).join("")}
      </tbody>`;
    tbl.querySelectorAll('select[data-act="role"]').forEach(s =>
      s.addEventListener("change", e => patchUser(s.closest("tr").dataset.id, {role: e.target.value})));
    tbl.querySelectorAll('input[data-act="active"]').forEach(c =>
      c.addEventListener("change", e => patchUser(c.closest("tr").dataset.id, {is_active: e.target.checked})));
    tbl.querySelectorAll('button[data-act="resetpw"]').forEach(b =>
      b.addEventListener("click", () => {
        const np = prompt("New password (min 8 chars):");
        if (np && np.length >= 8) patchUser(b.dataset.id, {password: np});
      }));
  }

  async function patchUser(id, body) {
    const r = await fetch(`/auth/users/${id}`, {
      method:"PATCH", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)
    });
    if (!r.ok) { const j = await r.json().catch(()=>({})); alert(j.detail||"Failed"); }
    loadAdminUsers();
  }

  async function adminCreateUser() {
    const username = prompt("Username:"); if (!username) return;
    const password = prompt("Initial password (min 8 chars):"); if (!password || password.length < 8) return;
    const role = (prompt("Role (user|admin):", "user") || "user").trim();
    const r = await fetch("/auth/users", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({username, password, role})
    });
    if (!r.ok) { const j = await r.json().catch(()=>({})); alert(j.detail||"Failed"); return; }
    loadAdminUsers();
  }

  async function loadAdminKeys() {
    const r = await fetch("/auth/admin/keys");
    if (!r.ok) return;
    const { keys } = await r.json();
    const tbl = $("#acct-allkeys");
    tbl.innerHTML = `
      <thead><tr><th>ID</th><th>User</th><th>Name</th><th>Masked</th><th>IP allow</th><th>Last used</th><th>Status</th><th></th></tr></thead>
      <tbody>${keys.map(k => `
        <tr>
          <td>${k.id}</td><td>${k.user_id}</td>
          <td>${escapeHtml(k.name)}</td><td class="mono">${escapeHtml(k.masked)}</td>
          <td>${escapeHtml(k.ip_allowlist||"")}</td>
          <td>${k.last_used_at ? fmtTs(k.last_used_at)+" "+escapeHtml(k.last_used_ip||"") : "—"}</td>
          <td>${k.revoked?'<span class="danger">revoked</span>':"active"}</td>
          <td>${k.revoked?"":`<button class="ghost danger" data-revoke="${k.id}">Revoke</button>`}</td>
        </tr>`).join("")}
      </tbody>`;
    tbl.querySelectorAll("button[data-revoke]").forEach(b =>
      b.addEventListener("click", async () => {
        if (!confirm("Revoke this key?")) return;
        await fetch(`/auth/admin/keys/${b.dataset.revoke}`, {method:"DELETE"});
        loadAdminKeys();
      }));
  }

  async function loadAdminSessions() {
    const r = await fetch("/auth/sessions");
    if (!r.ok) return;
    const { sessions } = await r.json();
    const tbl = $("#acct-sessions");
    tbl.innerHTML = `
      <thead><tr><th>User</th><th>IP</th><th>Created</th><th>Expires</th><th>UA</th><th></th></tr></thead>
      <tbody>${sessions.map(s => `
        <tr>
          <td>${escapeHtml(s.username)}</td>
          <td>${escapeHtml(s.ip||"")}</td>
          <td>${fmtTs(s.created_at)}</td>
          <td>${fmtTs(s.expires_at)}</td>
          <td class="ua">${escapeHtml((s.user_agent||"").slice(0,60))}</td>
          <td><button class="ghost danger" data-kill="${escapeHtml(s.id_prefix.replace("…",""))}">Kill</button></td>
        </tr>`).join("")}
      </tbody>`;
    tbl.querySelectorAll("button[data-kill]").forEach(b =>
      b.addEventListener("click", async () => {
        if (!confirm("Kill this session?")) return;
        await fetch(`/auth/sessions/${b.dataset.kill}`, {method:"DELETE"});
        loadAdminSessions();
      }));
  }

  // -------------------------------------------------------- models admin
  async function loadModels() {
    if (!_me || _me.role !== "admin") return;
    const wrap = $("#acct-models-list");
    if (!wrap) return;
    wrap.innerHTML = `<p class="hint">Loading…</p>`;
    let models = [];
    try {
      const r = await fetch("/admin/models");
      if (!r.ok) { wrap.innerHTML = `<p class="hint err">Failed to load models (${r.status}).</p>`; return; }
      models = (await r.json()).models || [];
    } catch (e) {
      wrap.innerHTML = `<p class="hint err">${escapeHtml(String(e))}</p>`;
      return;
    }

    // Pull each model's full config in parallel.
    const cfgs = await Promise.all(models.map(async m => {
      try {
        const r = await fetch(`/admin/models/${encodeURIComponent(m.id)}/config`);
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    }));

    wrap.innerHTML = models.map((m, i) => {
      const c = cfgs[i] || {};
      const extra = (c.extra_args || []).join("\n");
      const sp = c.system_prompt || "";
      const ctx = c.ctx_size || "";
      const yamlArgs = (c.yaml_args || []).join(" ");
      return `
      <details class="acct-model-row" data-id="${escapeHtml(m.id)}">
        <summary>
          <span class="mono" style="font-weight:600">${escapeHtml(m.id)}</span>
          <span class="acct-pillbox">${escapeHtml(m.kind)}</span>
          <span class="acct-pillbox">${escapeHtml(m.backend || "llama_cpp")}</span>
          ${m.published ? `<span class="acct-pillbox" style="background:#16a34a33;color:#86efac">published</span>` : `<span class="acct-pillbox">draft</span>`}
          ${m.label ? `<span class="hint">"${escapeHtml(m.label)}"</span>` : ""}
        </summary>
        <div class="acct-model-body">
          <div class="hint mono" style="font-size:11px;word-break:break-all">${escapeHtml(m.path || "")}</div>
          ${yamlArgs ? `<div class="hint">YAML args: <code>${escapeHtml(yamlArgs)}</code></div>` : ""}
          <div class="acct-grid" style="grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
            <label>Label (display name)
              <input type="text" data-field="label" value="${escapeHtml(m.label || "")}" placeholder="e.g. Qwen3 35B (chat)">
            </label>
            <label>Context size (--max-model-len / -c)
              <input type="number" data-field="ctx_size" value="${escapeHtml(String(ctx))}" placeholder="(default)">
            </label>
          </div>
          <label style="margin-top:8px">Extra runtime args (one per token, e.g. <code>--tokenizer</code> on one line, value on next)
            <textarea data-field="extra_args" rows="5" placeholder="--tokenizer&#10;Qwen/Qwen3-30B-A3B&#10;--tensor-parallel-size&#10;1">${escapeHtml(extra)}</textarea>
          </label>
          <label style="margin-top:8px">System prompt (prepended for this model)
            <textarea data-field="system_prompt" rows="3" placeholder="(none)">${escapeHtml(sp)}</textarea>
          </label>
          <div class="acct-admin-actions" style="justify-content:flex-end;margin-top:8px">
            <button class="ghost" data-act="toggle-publish">${m.published ? "Unpublish" : "Publish"}</button>
            <button class="primary" data-act="save">Save</button>
          </div>
          <div class="acct-msg" data-msg></div>
        </div>
      </details>`;
    }).join("") || `<p class="hint">No models discovered.</p>`;

    wrap.querySelectorAll(".acct-model-row").forEach(row => {
      const id = row.dataset.id;
      const get = sel => row.querySelector(sel);
      get('button[data-act="save"]').addEventListener("click", () => saveModelConfig(id, row));
      get('button[data-act="toggle-publish"]').addEventListener("click", () => togglePublish(id, row));
    });
  }

  async function saveModelConfig(id, row) {
    const msgEl = row.querySelector("[data-msg]");
    const f = sel => row.querySelector(sel);
    const ctxRaw = f('input[data-field="ctx_size"]').value.trim();
    const labelRaw = f('input[data-field="label"]').value.trim();
    const extraRaw = f('textarea[data-field="extra_args"]').value;
    const spRaw = f('textarea[data-field="system_prompt"]').value;
    const extra_args = extraRaw.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    const body = {
      ctx_size: ctxRaw ? parseInt(ctxRaw, 10) : null,
      extra_args,
      system_prompt: spRaw || null,
    };
    msgEl.textContent = "Saving…"; msgEl.className = "acct-msg";
    try {
      const r = await fetch(`/admin/models/${encodeURIComponent(id)}/config`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const t = await r.text();
        msgEl.textContent = `Save failed: ${t}`;
        msgEl.classList.add("err");
        return;
      }
      // Update label separately if it changed (publish endpoint takes label).
      if (labelRaw !== undefined) {
        await fetch(`/admin/models/${encodeURIComponent(id)}/publish`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({label: labelRaw || null}),
        });
      }
      msgEl.textContent = "Saved. Will apply on the next model swap.";
      msgEl.classList.add("ok");
    } catch (e) {
      msgEl.textContent = `Error: ${e}`;
      msgEl.classList.add("err");
    }
  }

  async function togglePublish(id, row) {
    const msgEl = row.querySelector("[data-msg]");
    const summary = row.querySelector("summary");
    const isPublished = summary.innerHTML.includes("published");
    const url = isPublished
      ? `/admin/models/${encodeURIComponent(id)}/unpublish`
      : `/admin/models/${encodeURIComponent(id)}/publish`;
    const labelRaw = row.querySelector('input[data-field="label"]').value.trim();
    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: isPublished ? "{}" : JSON.stringify({label: labelRaw || null}),
    });
    if (!r.ok) {
      msgEl.textContent = `Failed: ${await r.text()}`;
      msgEl.classList.add("err");
      return;
    }
    loadModels();
  }

  // -------------------------------------------------------- helpers
  function msg(id, text, isErr) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("err", !!isErr);
    el.classList.toggle("ok", !isErr);
  }
  function escapeHtml(s) {
    return String(s||"").replace(/[&<>"']/g, c =>
      ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  }
  function fmtTs(t) {
    if (!t) return "—";
    try { return new Date(t*1000).toLocaleString(); } catch { return String(t); }
  }

  // -------------------------------------------------------- boot probe
  // On every page load, confirm we're authenticated; otherwise → login page.
  (async function probe() {
    try {
      const r = await origFetch("/auth/me-anonymous");
      if (!r.ok) return;
      const j = await r.json();
      if (!j.authenticated) {
        window.location.replace("/ui/login.html");
        return;
      }
      _me = j.user;
      // Decorate topbar with username + Account button (if not present yet).
      decorateTopbar(j.user);
    } catch {}
  })();

  function decorateTopbar(user) {
    const actions = document.querySelector(".topbar-actions");
    if (!actions || $("#acct-open-btn")) return;
    const wrap = document.createElement("div");
    wrap.className = "acct-pill";
    wrap.id = "acct-open-btn";
    wrap.title = "Open account settings";
    wrap.innerHTML = `
      <span class="mi">person</span>
      <span id="acct-pill-name">${escapeHtml(user.username)}</span>
      <span class="acct-pill-role" id="acct-pill-role">${escapeHtml(user.role)}</span>
    `;
    wrap.addEventListener("click", openModal);
    actions.insertBefore(wrap, actions.firstChild);
  }
})();
