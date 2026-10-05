/* ============================================================
   Trip Concierge - the chat UI of the multi-agent sample
   Vanilla JS frontend: no build step, no CDN, no framework.
   The backend serves it at GET / (same origin, so fetches are relative).
   ============================================================ */

"use strict";

/* ------------------- Constants and helpers ------------------- */

const $ = (id) => document.getElementById(id);

const API = {
  INVOCATIONS: "/invocations",
  INFO: "/api/info",
  STREAM: "/api/chat/stream",
};

// Headers required by the chat endpoints: the user id is passed on to every remote agent
// (they partition their memory with it), the session id names the conversation.
const HDR_USER = "X-GreenNode-AgentBase-User-Id";
const HDR_SESSION = "X-GreenNode-AgentBase-Session-Id";
const HDR_KEY = "X-API-Key";
const KEY_STORAGE = "tripConciergeApiKey";
const USER_STORAGE = "tripConciergeUser";
const ID_PATTERN = /^[A-Za-z0-9._:@+=~-]{1,128}$/; // what the backend accepts as a user or session id
const DEFAULT_USER = "demo-user";

// Sample request (Vietnamese): itinerary + table booking + stock data in one message.
const SAMPLE_REQUEST =
  "Lên lịch trình 2 ngày ở Đà Nẵng vào cuối tuần sau cho 2 người, đặt bàn tại Quán Ngon 123 " +
  "vào 19:00 thứ Bảy, và cho mình biết cổ phiếu ngành du lịch tuần này thế nào.";

const state = {
  user: "",       // user id sent to the backend
  session: "",    // conversation id
  sending: false, // waiting for the agent's reply
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function truncate(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// Creates an element with a class name and optional text (textContent: always safe)
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ------------------- Minimal markdown -------------------
   HTML is escaped FIRST, then: **bold**, `code`, ```code blocks```,
   links [text](url) / bare URLs (target=_blank), headings rendered bold. */

function renderInlineMd(s) {
  const stash = [];
  const keep = (html) => { stash.push(html); return `\u0001${stash.length - 1}\u0001`; };

  // Inline code is stashed behind a token so bold/link rules never touch its content
  s = s.replace(/`([^`\n]+)`/g, (m, code) => keep(`<code class="md-code">${code}</code>`));
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a class="md-link" href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)"]+)/g,
    (m, p, url) => `${p}<a class="md-link" href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
  s = s.replace(/\u0001(\d+)\u0001/g, (m, i) => stash[Number(i)]);
  return s;
}

function renderMarkdown(raw) {
  const esc = escapeHtml(raw ?? "");
  const stash = [];
  const keep = (html) => { stash.push(html); return `\u0000${stash.length - 1}\u0000`; };

  // 1) Code blocks ```...``` -> token (content already escaped, kept as is)
  let text = esc.replace(/```[^\n]*\n?([\s\S]*?)```/g, (m, code) =>
    keep(`<pre class="md-pre"><code>${code.replace(/\n$/, "")}</code></pre>`));

  // 2) Line by line: bold headings, bullet lists, paragraphs
  const out = [];
  let list = null;
  const flushList = () => {
    if (list) {
      out.push(`<ul class="md-ul">${list.map((li) => `<li>${li}</li>`).join("")}</ul>`);
      list = null;
    }
  };

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();

    // A line that is a code-block token is emitted as is, not wrapped in <p>
    if (/^\u0000\d+\u0000$/.test(line)) { flushList(); out.push(line); continue; }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushList();
      out.push(`<div class="md-h md-h${heading[1].length}"><strong>${renderInlineMd(heading[2])}</strong></div>`);
      continue;
    }

    const item = line.match(/^(?:[-*•]|\d+[.)])\s+(.+)$/);
    if (item) { (list ??= []).push(renderInlineMd(item[1])); continue; }

    if (!line) { flushList(); continue; }

    flushList();
    out.push(`<p class="md-p">${renderInlineMd(line)}</p>`);
  }
  flushList();

  // 3) Restore the code blocks from their tokens
  return out.join("\n").replace(/\u0000(\d+)\u0000/g, (m, i) => stash[Number(i)]);
}

/* ------------------- API layer (same origin) ------------------- */

// The API key (AGENT_API_KEY on the server) is kept in sessionStorage only: it is
// forgotten when the tab closes and never written to disk by this page.
function getApiKey() {
  try { return sessionStorage.getItem(KEY_STORAGE) || ""; } catch { return ""; }
}

function setApiKey(key) {
  try {
    if (key) sessionStorage.setItem(KEY_STORAGE, key);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch { /* storage unavailable: the key is asked again on the next 401 */ }
}

// Asks for the key once; returns false if the user cancels.
function askApiKey(message) {
  const key = (window.prompt(message) || "").trim();
  if (!key) return false;
  setApiKey(key);
  return true;
}

// fetch() with the API key attached. A 401 means the key is missing or wrong: ask once and retry.
async function apiFetch(url, options = {}) {
  const send = () => {
    const headers = { ...(options.headers || {}) };
    const key = getApiKey();
    if (key) headers[HDR_KEY] = key;
    return fetch(url, { ...options, headers });
  };
  let res = await send();
  if (res.status === 401) {
    setApiKey("");
    if (askApiKey("Máy chủ yêu cầu API key (header X-API-Key). Nhập API key:")) res = await send();
  }
  return res;
}

// Error text of a failed response; includes the request id the server logged, if any.
function errorText(data, res) {
  const text = (data && (data.error || data.message)) || `HTTP ${res.status}`;
  const requestId = data && ((data.details && data.details.request_id) || data.request_id);
  return requestId ? `${text} (${requestId})` : text;
}

async function requestJson(url, options = {}) {
  let res;
  try {
    res = await apiFetch(url, options);
  } catch (e) {
    throw new Error(`không kết nối được server (${e.message})`);
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty body or not JSON */ }
  if (!res.ok) throw new Error(errorText(data, res));
  if (data && data.status === "error") throw new Error(errorText(data, res));
  return data;
}

function chatHeaders() {
  return {
    "Content-Type": "application/json",
    [HDR_USER]: state.user,
    [HDR_SESSION]: state.session,
  };
}

function postInvocation(message) {
  return requestJson(API.INVOCATIONS, {
    method: "POST",
    headers: chatHeaders(),
    body: JSON.stringify({ message }),
  });
}

/* ------------------- Chat: message bubbles ------------------- */

function scrollToBottom() {
  const box = $("chatMessages");
  box.scrollTop = box.scrollHeight;
}

function appendUserBubble(text) {
  const wrap = el("div", "msg user");
  wrap.appendChild(el("div", "bubble", text)); // textContent is safe
  $("chatMessages").appendChild(wrap);
  scrollToBottom();
}

function renderEmptyChat() {
  const box = $("chatMessages");
  box.innerHTML = "";
  const empty = el("div", "chat-empty");
  empty.appendChild(el("div", "chat-empty-icon", "🎯"));
  empty.appendChild(el("div", "chat-empty-title", "Một yêu cầu, nhiều agent cùng làm"));
  empty.appendChild(el("div", "chat-empty-sub",
    "Trip Concierge chia yêu cầu của bạn thành từng phần, giao cho agent lịch trình, agent đặt bàn " +
    "và công cụ dữ liệu chứng khoán, rồi tổng hợp thành một câu trả lời. Bạn xem được từng bước ở ngay trong tin nhắn."));
  const sample = el("button", "btn ghost small", "Dùng câu hỏi mẫu");
  sample.type = "button";
  sample.addEventListener("click", () => {
    $("composerInput").value = SAMPLE_REQUEST;
    autosizeComposer();
    $("composerInput").focus();
  });
  empty.appendChild(sample);
  box.appendChild(empty);
}

/* ------------------- One turn: steps, tokens, final answer ------------------- */

// A bot message under construction: the orchestration steps (agents asked, tools called)
// above the answer bubble. Events of POST /api/chat/stream are applied with handleEvent().
function createTurn() {
  const wrap = el("div", "msg bot");
  wrap.appendChild(el("div", "avatar", "🎯"));
  const body = el("div", "msg-body");
  body.appendChild(el("div", "msg-name", "Trip Concierge"));
  const steps = el("div", "steps");
  const bubble = el("div", "bubble md");
  bubble.innerHTML =
    '<span class="typing-dots"><span></span><span></span><span></span></span>';
  const foot = el("div", "msg-foot");
  body.append(steps, bubble, foot);
  wrap.appendChild(body);
  $("chatMessages").appendChild(wrap);
  scrollToBottom();

  // The elapsed time is shown in the footer while the turn runs
  const startedAt = Date.now();
  const timer = setInterval(() => {
    foot.textContent = ((Date.now() - startedAt) / 1000).toFixed(1) + " giây";
  }, 100);

  return {
    wrap, steps, bubble, foot,
    text: "",       // tokens received so far
    pending: {},    // agent name -> its running delegation steps, oldest first
    stopTimer: () => clearInterval(timer),
  };
}

function setBubbleText(turn, text) {
  turn.bubble.classList.add("streaming");
  turn.bubble.textContent = text;
  turn.bubble.appendChild(el("span", "stream-cursor", "▍"));
}

function addStep(turn, className, icon, title) {
  const step = el("div", "step " + className);
  const head = el("div", "step-head");
  const iconEl = el("span", "step-icon", icon);
  const titleEl = el("span", "step-title", title);
  const timeEl = el("span", "step-time");
  head.append(iconEl, titleEl, timeEl);
  step.appendChild(head);
  turn.steps.appendChild(step);
  scrollToBottom();
  return { step, iconEl, titleEl, timeEl };
}

function addDelegateStep(turn, agent, request) {
  const row = addStep(turn, "running", "◌", `Đang hỏi ${agent}`);
  if (request) row.step.appendChild(el("div", "step-request", request));
  (turn.pending[agent] ??= []).push(row);
}

// Partial text of the agent (only when the server runs with A2A_STREAMING=1). The agent's calls
// in one turn run one after the other, so the oldest running step is the one that is talking.
function showAgentText(turn, agent, text, append) {
  const row = (turn.pending[agent] || [])[0];
  if (!row) return;
  if (!row.preview) {
    const details = el("details");
    details.open = true;
    details.appendChild(el("summary", "", "Trả lời của agent"));
    row.preview = el("pre", "step-preview");
    details.appendChild(row.preview);
    row.step.appendChild(details);
  }
  row.preview.textContent = append ? row.preview.textContent + text : text;
  row.preview.scrollTop = row.preview.scrollHeight;
}

function finishDelegateStep(turn, agent, ok, elapsedMs) {
  const row = (turn.pending[agent] || []).shift();
  if (!row) return;
  row.step.classList.remove("running");
  row.step.classList.toggle("failed", !ok);
  row.iconEl.textContent = ok ? "✓" : "✗";
  row.titleEl.textContent = ok ? `${agent} đã trả lời` : `${agent} không trả lời được`;
  row.timeEl.textContent = (elapsedMs / 1000).toFixed(1) + " giây";
}

// Shows the answer: markdown, the request id for support, and stops the timer.
function finishTurn(turn, response, requestId) {
  turn.stopTimer();
  turn.bubble.classList.remove("streaming");
  turn.bubble.innerHTML = renderMarkdown(response);
  turn.foot.textContent = requestId ? `Mã yêu cầu: ${requestId}` : "";
  scrollToBottom();
}

function failTurn(turn, message) {
  turn.stopTimer();
  turn.foot.textContent = "";
  // Steps that were still running will never finish now
  for (const row of Object.values(turn.pending).flat()) {
    row.step.classList.remove("running");
    row.step.classList.add("failed");
    row.iconEl.textContent = "✗";
    row.titleEl.textContent = row.titleEl.textContent.replace("Đang hỏi", "Bị gián đoạn khi hỏi");
  }
  turn.bubble.classList.remove("streaming");
  turn.bubble.innerHTML = renderMarkdown(`⚠️ **Không gọi được agent:** ${message}`);
  scrollToBottom();
}

// Applies one SSE event; returns true when it was the final "done" event.
function handleEvent(turn, ev, res) {
  switch (ev.type) {
    case "token":
      turn.text += ev.text || "";
      setBubbleText(turn, turn.text);
      scrollToBottom();
      return false;
    case "delegate":
      addDelegateStep(turn, ev.agent, truncate(ev.request || "", 300));
      return false;
    case "delegate_text":
      showAgentText(turn, ev.agent, ev.text || "", ev.append === true);
      return false;
    case "delegate_done":
      finishDelegateStep(turn, ev.agent, ev.ok === true, ev.elapsed_ms || 0);
      return false;
    case "tool":
      addStep(turn, "tool", "🔧", `Gọi công cụ ${ev.name}`);
      return false;
    case "done":
      finishTurn(turn, ev.response || turn.text, ev.request_id);
      return true;
    case "error":
      throw new Error(errorText(ev, res));
    default:
      return false; // an event type this UI does not know: ignore it
  }
}

/* ---------- SSE streaming: POST /api/chat/stream ---------- */

// The streaming endpoint cannot be used at all (missing route, or a proxy that does not
// pass SSE). Only then does the caller fall back to POST /invocations. Any other failure
// is shown as is: the server may already have processed the message, and a retry would run it twice.
class StreamUnavailable extends Error {}

async function postStream(message, turn) {
  let res;
  try {
    res = await apiFetch(API.STREAM, {
      method: "POST",
      headers: chatHeaders(),
      body: JSON.stringify({ message }),
    });
  } catch (e) {
    throw new StreamUnavailable(e.message);
  }
  const ct = res.headers.get("content-type") || "";
  if (!res.ok || !ct.includes("text/event-stream")) {
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    const text = errorText(data, res);
    if (res.ok || res.status === 404 || res.status === 405) throw new StreamUnavailable(text);
    throw new Error(text);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let done = false;
  for (;;) {
    const { value, done: closed } = await reader.read();
    if (closed) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let ev = null;
      try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
      if (handleEvent(turn, ev, res)) done = true;
    }
  }
  if (!done) throw new Error("Stream kết thúc mà không có câu trả lời.");
}

/* ------------------- Chat state and status dot ------------------- */

function setDot(kind) {
  const dot = $("statusDot");
  dot.classList.remove("ok", "err");
  if (kind) dot.classList.add(kind);
}

function updateChips() {
  $("chipUser").textContent = "@" + state.user;
  $("chipSession").textContent = state.session;
}

/* ------------------- GET /api/info ------------------- */

function renderAgents(agents) {
  const box = $("agentsList");
  box.innerHTML = "";
  if (!Array.isArray(agents)) {
    box.appendChild(el("div", "list-empty", "Nhập API key để xem danh sách agent."));
    return;
  }
  for (const agent of agents) {
    const item = el("div", "agent-item");
    const dot = el("span", "dot" + (agent.name ? " ok" : " err"));
    item.appendChild(dot);
    item.appendChild(el("span", "agent-name", agent.name || agent.label));
    item.appendChild(el("span", "agent-state", agent.name ? "sẵn sàng" : "chưa kết nối"));
    box.appendChild(item);
  }
}

async function loadInfo() {
  try {
    let data = await requestJson(API.INFO);
    // GET /api/info is public and only reports auth_required; with the key it also returns the details.
    if (data.auth_required && !getApiKey()
        && askApiKey("Máy chủ yêu cầu API key (header X-API-Key). Nhập API key:")) {
      data = await requestJson(API.INFO);
    }
    $("infoAgent").textContent = data.agent || "—";
    $("infoMemory").textContent = String(data.memory_id || "—").slice(0, 8); // memory id shortened to 8 characters
    $("infoModel").textContent = data.llm_model || "—";
    $("infoGateway").textContent = truncate(data.gateway || "—", 30);
    renderAgents(data.agents);
    setDot("ok");
  } catch (e) {
    setDot("err");
    showToast("Không tải được /api/info: " + e.message);
  }
}

/* ------------------- User and conversation ------------------- */

function newSession() {
  state.session = "s-" + Date.now();
  updateChips();
  renderEmptyChat();
}

function applyUser() {
  const input = $("userInput");
  const value = input.value.trim();
  if (value === state.user) return;
  if (state.sending) {
    input.value = state.user;
    showToast("Đang xử lý một yêu cầu, hãy đợi xong rồi đổi người dùng.");
    return;
  }
  if (!ID_PATTERN.test(value)) {
    input.value = state.user;
    showToast("Mã người dùng gồm 1-128 ký tự: chữ, số và . _ : @ + = ~ -");
    return;
  }
  state.user = value;
  try { localStorage.setItem(USER_STORAGE, value); } catch { /* storage unavailable: not remembered */ }
  newSession(); // another user, another conversation
}

/* ------------------- Sending a message ------------------- */

async function sendMessage() {
  const input = $("composerInput");
  const text = input.value.trim();
  if (!text || state.sending) return;

  state.sending = true;
  $("sendBtn").disabled = true;
  if ($("chatMessages").querySelector(".chat-empty")) $("chatMessages").innerHTML = "";

  appendUserBubble(text);
  input.value = "";
  autosizeComposer();

  const turn = createTurn();
  try {
    try {
      // 1) SSE streaming: steps and tokens render as they arrive
      await postStream(text, turn);
    } catch (streamErr) {
      if (!(streamErr instanceof StreamUnavailable)) throw streamErr;
      // 2) Streaming is not available: fall back to POST /invocations
      const data = await postInvocation(text);
      if (!data || data.status !== "success") {
        throw new Error((data && data.error) || streamErr.message || "Agent trả về lỗi.");
      }
      for (const d of data.delegations || []) {
        addDelegateStep(turn, d.agent, "");
        finishDelegateStep(turn, d.agent, d.ok === true, d.elapsed_ms || 0);
      }
      finishTurn(turn, data.response, data.request_id);
    }
    setDot("ok");
  } catch (e) {
    failTurn(turn, e.message);
    setDot("err");
    showToast("Gọi agent thất bại: " + e.message);
  } finally {
    turn.stopTimer();
    state.sending = false;
    $("sendBtn").disabled = false;
    $("composerInput").focus();
  }
}

/* ------------------- Error toast (auto-hides, dismissible) ------------------- */

let toastTimer = null;

function showToast(message) {
  const toast = $("toast");
  $("toastMsg").textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 6000);
}

/* ------------------- Composer (auto-growing textarea) ------------------- */

function autosizeComposer() {
  const box = $("composerInput");
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 160) + "px";
}

/* ------------------- Event binding and startup ------------------- */

function bindEvents() {
  $("sendBtn").addEventListener("click", sendMessage);

  const composer = $("composerInput");
  composer.addEventListener("input", autosizeComposer);
  composer.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { // Enter sends, Shift+Enter inserts a newline
      e.preventDefault();
      sendMessage();
    }
  });

  $("userInput").addEventListener("change", applyUser);
  $("newSessionBtn").addEventListener("click", () => {
    if (state.sending) {
      showToast("Đang xử lý một yêu cầu, hãy đợi xong rồi tạo hội thoại mới.");
      return;
    }
    newSession();
  });

  // The sidebar opens/closes on narrow screens
  $("sidebarToggle").addEventListener("click", () => $("layout").classList.toggle("show-sidebar"));

  $("toastClose").addEventListener("click", () => {
    clearTimeout(toastTimer);
    $("toast").classList.remove("show");
  });
}

async function init() {
  let user = DEFAULT_USER;
  try { user = localStorage.getItem(USER_STORAGE) || DEFAULT_USER; } catch { /* storage unavailable */ }
  state.user = ID_PATTERN.test(user) ? user : DEFAULT_USER;
  $("userInput").value = state.user;
  bindEvents();
  newSession();
  await loadInfo();
}

document.addEventListener("DOMContentLoaded", init);
