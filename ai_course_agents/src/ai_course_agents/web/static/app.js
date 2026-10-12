// Главный модуль интерфейса: состояние, вход, список чатов, отправка вопроса,
// обработка событий SSE. Отрисовка — в render.js, карта — в map.js, запросы — в api.js.

import { AuthError, request, streamAnswer } from "./api.js";
import { focusPlace, initMap, refreshMapSize, showPlaces } from "./map.js";
import { el, renderBot, renderUser, renderWelcome, shortTime } from "./render.js";

// ── состояние приложения ────────────────────────────────────────────────────
const state = {
  user: null,          // {id, username} — кто вошёл
  chats: [],           // [{id, title, updated_at}] — для боковой панели, свежие сверху
  chatId: null,        // открытый чат; null — «новый чат», он создастся при первом вопросе
  messages: [],        // [{role: "user", text, node} | msg ассистента (см. render.js) + node]
  abort: null,         // AbortController текущего ответа; не null — ответ идёт
  mapMsg: null,        // чьи места сейчас на карте
};

// Короткая запись для поиска элементов по id.
const $ = (id) => document.getElementById(id);
const isMobile = () => window.matchMedia("(max-width: 900px)").matches;

// ── вход и выход ────────────────────────────────────────────────────────────
let authMode = "login";        // "login" | "register" — какая вкладка формы открыта

function showAuth() {
  state.abort?.abort();
  $("app").hidden = true;
  $("auth").hidden = false;
  $("auth-form").username.focus();
}

async function showApp(user) {
  state.user = user;
  $("who").textContent = user.username;
  $("auth").hidden = true;
  $("app").hidden = false;
  initMap();
  await loadChats();
  // Открываем последний чат, если он есть, иначе — приветствие нового.
  if (state.chats.length) await openChat(state.chats[0].id); else newChat();
}

/** Ошибка «не вошли» в любом месте приложения → экран входа; остальное — пробрасываем. */
function handleAuth(err) {
  if (err instanceof AuthError) { showAuth(); return true; }
  return false;
}

function setupAuthForm() {
  const form = $("auth-form");
  for (const tab of form.querySelectorAll(".tabs button")) {
    tab.addEventListener("click", () => {
      authMode = tab.dataset.mode;
      form.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === tab));
      $("auth-submit").textContent = authMode === "login" ? "Войти" : "Создать аккаунт";
      form.password.autocomplete = authMode === "login" ? "current-password" : "new-password";
      $("auth-error").textContent = "";
    });
  }
  form.addEventListener("submit", async (e) => {
    e.preventDefault();                                   // не перезагружать страницу
    $("auth-error").textContent = "";
    $("auth-submit").disabled = true;
    try {
      const body = { username: form.username.value.trim(), password: form.password.value };
      const user = await request("POST", `/api/auth/${authMode}`, body);
      form.password.value = "";
      await showApp(user);
    } catch (err) {
      $("auth-error").textContent = err.message || "не получилось";
    } finally {
      $("auth-submit").disabled = false;
    }
  });
  $("logout").addEventListener("click", async () => {
    await request("POST", "/api/auth/logout").catch(() => {});
    state.user = null; state.chats = []; state.chatId = null;
    showAuth();
  });
}

// ── список чатов ────────────────────────────────────────────────────────────
async function loadChats() {
  state.chats = await request("GET", "/api/chats");
  renderChatList();
}

function renderChatList() {
  const list = $("chat-list");
  list.replaceChildren();
  if (!state.chats.length) {
    list.append(el("li", { class: "empty-list" }, "Здесь будут ваши разговоры"));
    return;
  }
  for (const chat of state.chats) {
    const title = el("span", { class: "title" }, chat.title);
    const item = el("li", { class: `chat-item${chat.id === state.chatId ? " active" : ""}`, title: chat.title,
                            onclick: () => openChat(chat.id) },
      title,
      el("span", { class: "time" }, shortTime(chat.updated_at)),
      el("span", { class: "actions" },
        el("button", { class: "icon-btn", type: "button", title: "Переименовать",
                       onclick: (e) => { e.stopPropagation(); startRename(chat, title); } }, "✏️"),
        el("button", { class: "icon-btn", type: "button", title: "Удалить",
                       onclick: (e) => { e.stopPropagation(); deleteChat(chat); } }, "🗑️")));
    list.append(item);
  }
}

/** Переименование «на месте»: название превращается в поле ввода. Enter/уход фокуса — сохранить, Esc — отмена. */
function startRename(chat, titleNode) {
  const input = el("input", { value: chat.title, maxlength: 80 });
  titleNode.replaceWith(input);
  input.focus(); input.select();
  input.addEventListener("click", (e) => e.stopPropagation());   // клик в поле не открывает чат
  let finished = false;
  const finish = async (save) => {
    if (finished) return;
    finished = true;
    const value = input.value.trim();
    if (save && value && value !== chat.title) {
      try {
        const updated = await request("PATCH", `/api/chats/${chat.id}`, { title: value });
        chat.title = updated.title;
        if (chat.id === state.chatId) $("chat-title").textContent = chat.title;
      } catch (err) { if (!handleAuth(err)) alert(err.message); }
    }
    renderChatList();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

async function deleteChat(chat) {
  if (!confirm(`Удалить чат «${chat.title}»? Историю нельзя будет вернуть.`)) return;
  try {
    await request("DELETE", `/api/chats/${chat.id}`);
    state.chats = state.chats.filter((c) => c.id !== chat.id);
    if (chat.id === state.chatId) newChat(); else renderChatList();
  } catch (err) { if (!handleAuth(err)) alert(err.message); }
}

function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("scrim").hidden = true;
}

// ── открытие чата и история ─────────────────────────────────────────────────
function newChat() {
  if (state.abort) return;                    // во время ответа не уходим из чата
  state.chatId = null;
  state.messages = [];
  $("chat-title").textContent = "Новый чат";
  renderChatList();
  renderMessages();
  showPlaces([], "Найденные места появятся на карте");
  closeSidebar();
  $("input").focus();
}

/** Сообщение ассистента из сохранённой истории → объект-состояние для render.js. */
function botFromHistory(h, question) {
  return {
    // status приходит из истории: "done", либо "stopped"/"error" — ответ был прерван
    role: "assistant", status: h.status || "done", question,
    error: h.status === "error" ? { message: "Ответ был прерван из-за ошибки.", retryable: true } : null,
    text: h.text, thoughts: h.thoughts || "",
    steps: (h.steps || []).map((s) => ({ ...s, done: true })),
    places: h.places || [],
    usage: null, ms: null, guardrail: false, traceOpen: false,
  };
}

async function openChat(chatId) {
  if (state.abort) return;
  try {
    const { chat, messages } = await request("GET", `/api/chats/${chatId}/messages`);
    state.chatId = chat.id;
    $("chat-title").textContent = chat.title;
    let lastQuestion = "";
    state.messages = messages.map((m) => {
      if (m.role === "user") { lastQuestion = m.text; return { role: "user", text: m.text }; }
      return botFromHistory(m, lastQuestion);
    });
    renderChatList();
    renderMessages();
    // На карту — места последнего ответа, где они были.
    const withPlaces = [...state.messages].reverse().find((m) => m.role === "assistant" && m.places.length);
    if (withPlaces) showMsgOnMap(withPlaces, false); else showPlaces([], "Найденные места появятся на карте");
    closeSidebar();
  } catch (err) {
    if (!handleAuth(err)) alert(err.message);
  }
}

// ── отрисовка сообщений ─────────────────────────────────────────────────────
const handlers = {
  onPlaceClick: (msg, name) => {
    if (state.mapMsg !== msg) showMsgOnMap(msg, false);
    if (focusPlace(name) && isMobile()) openMap();
  },
  onShowMap: (msg) => { showMsgOnMap(msg, true); },
  onRetry: (msg) => { send(msg.question); },
};

function showMsgOnMap(msg, openOnMobile) {
  state.mapMsg = msg;
  const n = msg.places.filter((p) => !p.is_origin).length;
  showPlaces(msg.places, msg.question ? `«${msg.question.slice(0, 60)}» — ${n} на карте` : null);
  if (openOnMobile && isMobile()) openMap();
}

function nodeFor(msg) {
  return msg.role === "user" ? renderUser(msg.text) : renderBot(msg, { ...handlers, rerender: () => update(msg) });
}

/** Полная перерисовка списка сообщений (при открытии чата). */
function renderMessages() {
  const box = $("messages-inner");
  box.replaceChildren();
  if (!state.messages.length) {
    box.append(renderWelcome((q) => send(q)));
    return;
  }
  for (const msg of state.messages) {
    msg.node = nodeFor(msg);
    box.append(msg.node);
  }
  scrollToBottom(true);
}

/** Перерисовать одно сообщение — не чаще раза за кадр, сколько бы событий ни пришло. */
function update(msg) {
  if (msg.pending) return;
  msg.pending = true;
  requestAnimationFrame(() => {
    msg.pending = false;
    const nearBottom = isNearBottom();
    const node = nodeFor(msg);
    msg.node.replaceWith(node);
    msg.node = node;
    if (nearBottom) scrollToBottom();      // автопрокрутка, только если пользователь внизу
  });
}

function isNearBottom() {
  const box = $("messages");
  return box.scrollHeight - box.scrollTop - box.clientHeight < 120;
}

function scrollToBottom(instant = false) {
  const box = $("messages");
  box.scrollTo({ top: box.scrollHeight, behavior: instant ? "auto" : "smooth" });
}

// ── отправка вопроса и поток ответа ─────────────────────────────────────────
/** Добавить место к местам сообщения без повторов (по id; точка адреса — по названию). */
function mergePlaces(msg, places) {
  const key = (p) => (p.id != null ? `id:${p.id}` : `o:${p.name}`);
  const seen = new Set(msg.places.map(key));
  for (const p of places) if (!seen.has(key(p))) { msg.places.push(p); seen.add(key(p)); }
}

/** Обработка одного события SSE: меняем состояние msg и просим перерисовку. */
function onEvent(msg, type, data) {
  switch (type) {
    case "title": {                                  // чату дали название по первому вопросу
      const chat = state.chats.find((c) => c.id === state.chatId);
      if (chat) chat.title = data.title;
      $("chat-title").textContent = data.title;
      renderChatList();
      break;
    }
    case "queued": msg.queued = true; break;
    case "thinking": msg.thoughts += data.text; break;
    case "text": msg.text += data.text; break;
    case "tool_call":
      // Текст перед вызовом инструмента — не ответ, а пояснение модели («сейчас найду…»):
      // переносим его в размышления, чтобы ответ начинался с чистого листа.
      if (msg.text.trim()) { msg.thoughts += `\n${msg.text}\n`; msg.text = ""; }
      msg.steps.push({ id: data.id, name: data.name, title: data.title, args: data.args, done: false });
      break;
    case "tool_result": {
      const step = msg.steps.find((s) => s.id === data.id) || msg.steps.find((s) => !s.done);
      if (step) Object.assign(step, { done: true, summary: data.summary, error: data.error, ms: data.ms });
      if (data.places?.length) {
        mergePlaces(msg, data.places);
        showMsgOnMap(msg, false);                    // карта обновляется прямо по ходу ответа
      }
      break;
    }
    case "done":
      Object.assign(msg, { status: "done", text: data.answer || msg.text, usage: data.usage,
                           ms: data.ms, guardrail: data.guardrail, traceOpen: false });
      break;
    case "error":
      Object.assign(msg, { status: "error", error: data, traceOpen: false });
      break;
  }
  update(msg);
}

async function send(text) {
  text = (text || "").trim();
  if (!text || state.abort) return;
  // Новый чат создаётся только при первом вопросе — пустых чатов в списке не бывает.
  if (!state.chatId) {
    try {
      const chat = await request("POST", "/api/chats");
      state.chatId = chat.id;
      state.chats.unshift(chat);
      renderChatList();
    } catch (err) { if (!handleAuth(err)) alert(err.message); return; }
  }
  if (!state.messages.length) $("messages-inner").replaceChildren();   // убрать приветствие

  const userMsg = { role: "user", text };
  const msg = { role: "assistant", status: "streaming", question: text, text: "", thoughts: "",
                steps: [], places: [], usage: null, ms: null, guardrail: false };
  for (const m of [userMsg, msg]) {
    state.messages.push(m);
    m.node = nodeFor(m);
    $("messages-inner").append(m.node);
  }
  scrollToBottom();
  setStreaming(new AbortController());
  try {
    await streamAnswer(state.chatId, text, (type, data) => onEvent(msg, type, data), state.abort.signal);
    if (msg.status === "streaming") msg.status = "done";          // поток закрылся без done — на всякий случай
  } catch (err) {
    if (err.name === "AbortError") Object.assign(msg, { status: "stopped", traceOpen: false });   // нажали «Стоп»
    else if (handleAuth(err)) return;
    else Object.assign(msg, { status: "error", error: { message: err.message, retryable: err.status !== 409 } });
  } finally {
    setStreaming(null);
    update(msg);
    // Чат с новым ответом — наверх списка.
    const chat = state.chats.find((c) => c.id === state.chatId);
    if (chat) { chat.updated_at = Date.now() / 1000; state.chats = [chat, ...state.chats.filter((c) => c !== chat)]; renderChatList(); }
  }
}

/** Режим «идёт ответ»: кнопка «Отправить» становится «Стоп». */
function setStreaming(controller) {
  state.abort = controller;
  const btn = $("send");
  btn.textContent = controller ? "■ Стоп" : "Отправить";
  btn.classList.toggle("btn-primary", !controller);
  $("new-chat").disabled = !!controller;
}

function setupComposer() {
  const input = $("input");
  const autosize = () => { input.style.height = "auto"; input.style.height = `${input.scrollHeight}px`; };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    // Enter — отправить; Shift+Enter — перенос строки; во время набора иероглифов/IME — не трогаем.
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.abort) { state.abort.abort(); return; }    // кнопка в режиме «Стоп»
    const text = input.value;
    if (!text.trim()) return;
    input.value = ""; autosize();
    send(text);
  });
}

// ── мобильная вёрстка: выдвижная панель чатов и карта во весь экран ─────────
function openMap() {
  $("map-panel").classList.add("open");
  refreshMapSize();
}

function setupMobile() {
  $("open-sidebar").addEventListener("click", () => { $("sidebar").classList.add("open"); $("scrim").hidden = false; });
  $("scrim").addEventListener("click", closeSidebar);
  $("open-map").addEventListener("click", openMap);
  $("close-map").addEventListener("click", () => $("map-panel").classList.remove("open"));
}

// ── запуск ──────────────────────────────────────────────────────────────────
async function boot() {
  setupAuthForm();
  setupComposer();
  setupMobile();
  $("new-chat").addEventListener("click", newChat);
  try {
    await showApp(await request("GET", "/api/auth/me"));   // уже вошли (cookie жив)?
  } catch (err) {
    if (!handleAuth(err)) console.error(err);
  }
}

boot();
