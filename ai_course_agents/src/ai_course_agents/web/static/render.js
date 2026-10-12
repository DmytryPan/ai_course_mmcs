// Отрисовка сообщений чата.
//
// Ответ ассистента описывается объектом-состоянием msg (его наполняет app.js из событий SSE
// или из сохранённой истории), а renderBot(msg) строит по нему DOM заново. Пересобирать
// карточку целиком проще и надёжнее, чем точечно править узлы, а по скорости это незаметно:
// app.js вызывает перерисовку не чаще одного раза за кадр (requestAnimationFrame).
//
// msg = {
//   status: "streaming" | "done" | "error" | "stopped",
//   text, thoughts,                     — ответ и размышления модели (накопленный текст)
//   steps: [{id, name, title, args, summary, error, ms, done}],
//   places: [...],                      — все места из результатов шагов (для карты)
//   usage, ms, guardrail, error: {message, retryable},
//   traceOpen, showThoughts             — что пользователь раскрыл (сохраняется между перерисовками)
// }

import { normalize, plural } from "./map.js";

const TOOL_ICONS = {
  semantic_search: "🔎", search_places: "📋", count_places: "🔢", top_brands: "🏷️",
  geocode_address: "📍", nearest_places: "🧭", rank_by_distance: "↕️", place_details: "🗂️",
};

/** Создать элемент: el("div", {class: "x"}, "текст", дочерний_узел). Текст — только через
 *  textContent (узлы-строки), поэтому данные из базы и от модели не превращаются в HTML. */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);   // onclick → click
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** Аргументы шага коротко и по-человечески: «пельмени» · радиус 1500 м. */
function formatArgs(name, a = {}) {
  const parts = [];
  if (a.query) parts.push(`«${a.query}»`);
  if (a.kind) parts.push(a.kind);
  if (a.street) parts.push(a.street + (a.housenumber ? `, ${a.housenumber}` : ""));
  if (a.brand) parts.push(a.brand);
  if (a.cuisine) parts.push(`кухня ${a.cuisine}`);
  if (a.only_24_7) parts.push("24/7");
  if (a.radius_m) parts.push(`радиус ${a.radius_m} м`);
  if (Array.isArray(a.place_ids)) parts.push(`${a.place_ids.length} ${plural(a.place_ids.length, "место", "места", "мест")}`);
  if (a.place_id != null) parts.push(`id ${a.place_id}`);
  if (!parts.length && a.lat != null) parts.push(`${(+a.lat).toFixed(4)}, ${(+a.lon).toFixed(4)}`);
  return parts.join(" · ");
}

/** Пузырь вопроса пользователя. */
export function renderUser(text) {
  return el("div", { class: "msg-user" }, text);
}

/** Блок «Ход мысли»: шаги агента и (по кнопке) размышления модели. */
function renderTrace(msg, rerender) {
  const live = msg.status === "streaming";
  const running = msg.steps.find((s) => !s.done);
  // Подпись свёрнутого блока: пока идёт работа — что делается сейчас; потом — итог.
  let label;
  if (live && msg.queued && !msg.thoughts && !msg.steps.length) label = "Все модели заняты, ждём очереди…";
  else if (live) label = running ? `Выполняю: ${running.title}…` : msg.text ? "Пишу ответ…" : "Думаю…";
  else label = `Ход мысли · ${msg.steps.length} ${plural(msg.steps.length, "шаг", "шага", "шагов")}`;

  const steps = el("ol", { class: "steps" }, msg.steps.map((s) => el("li", { class: `step${s.error ? " error" : ""}` },
    el("span", { class: "ico" }, s.done ? (s.error ? "⚠️" : TOOL_ICONS[s.name] || "🔧") : el("span", { class: "spinner" })),
    el("span", { class: "name" }, s.title, " ", el("span", { class: "args" }, formatArgs(s.name, s.args))),
    el("span", { class: "ms" }, s.done && s.ms != null ? `${s.ms} мс` : ""),
    el("span", { class: "result" }, s.done ? (s.error || s.summary || "готово") : "выполняется…"),
  )));

  // Размышления: пока агент думает — видны «вживую» (последние строки); после ответа —
  // спрятаны под ссылку, чтобы не мешать (DeepSeek размышляет по-английски).
  let thoughts = null;
  if (msg.thoughts) {
    if (live && !msg.text) thoughts = el("div", { class: "thoughts live" }, msg.thoughts.slice(-600));
    else thoughts = [
      el("button", { class: "thoughts-toggle", type: "button",
        onclick: () => { msg.showThoughts = !msg.showThoughts; rerender(); } },
        msg.showThoughts ? "скрыть размышления модели" : "размышления модели (англ.)"),
      msg.showThoughts ? el("div", { class: "thoughts" }, msg.thoughts) : null,
    ];
  }

  const details = el("details", { class: "trace", open: msg.traceOpen ?? live },
    el("summary", {}, live ? el("span", { class: "spinner" }) : "🧠", label, el("span", { class: "chev" }, "▾")),
    el("div", { class: "trace-body" }, msg.steps.length ? steps : null, thoughts));
  // Запоминаем, раскрыл ли пользователь блок, — чтобы перерисовка его не схлопнула.
  details.addEventListener("toggle", () => { msg.traceOpen = details.open; });
  return details;
}

/** Текст ответа: markdown → HTML → очистка DOMPurify; названия найденных мест — кликабельны. */
function renderAnswer(msg, onPlaceClick) {
  const box = el("div", { class: `answer${msg.status === "streaming" ? " cursor" : ""}` });
  box.innerHTML = DOMPurify.sanitize(marked.parse(msg.text || ""));
  // Жирные названия в ответе («**Пельмень** — …») связываем с метками на карте.
  const names = new Set(msg.places.map((p) => normalize(p.name)));
  for (const strong of box.querySelectorAll("strong")) {
    if (names.has(normalize(strong.textContent))) {
      strong.classList.add("place-link");
      strong.title = "показать на карте";
      strong.addEventListener("click", () => onPlaceClick(msg, strong.textContent));
    }
  }
  return box;
}

/** Нижняя строка: шаги к модели, токены, цена, время; кнопка «на карте». */
function renderMeta(msg, onShowMap) {
  const items = [];
  if (msg.guardrail) items.push(el("span", { class: "badge" }, "ответил фильтр, модель не вызывалась"));
  if (msg.usage && !msg.guardrail) {
    const u = msg.usage;
    const k = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)} тыс.` : String(n));
    items.push(el("span", {}, `${u.llm_calls} ${plural(u.llm_calls, "запрос", "запроса", "запросов")} к модели`));
    items.push(el("span", { title: `из кэша: ${u.cached_tokens}` }, `токены: ${k(u.input_tokens)} → ${k(u.output_tokens)}`));
    items.push(el("span", {}, `≈ ${u.cost_rub.toFixed(2)} ₽`));
  }
  if (msg.ms != null) items.push(el("span", {}, msg.ms < 1000 ? `${msg.ms} мс` : `${(msg.ms / 1000).toFixed(1)} с`));
  const mappable = msg.places.filter((p) => !p.is_origin).length;
  if (mappable) items.push(el("button", { type: "button", onclick: () => onShowMap(msg) }, `🗺 на карте (${mappable})`));
  return items.length ? el("div", { class: "meta" }, items) : null;
}

/**
 * Карточка ответа ассистента целиком.
 * handlers: {onPlaceClick(msg, name), onShowMap(msg), onRetry(msg), rerender()}
 */
export function renderBot(msg, handlers) {
  const parts = [];
  if (msg.steps.length || msg.thoughts || msg.status === "streaming") parts.push(renderTrace(msg, handlers.rerender));
  if (msg.text || msg.status === "streaming") parts.push(renderAnswer(msg, handlers.onPlaceClick));
  if (msg.status === "stopped") parts.push(el("div", { class: "notice info" }, "⏹ Ответ остановлен."));
  if (msg.status === "error") {
    parts.push(el("div", { class: "notice error" }, "⚠️ ", msg.error?.message || "Что-то пошло не так.",
      msg.error?.retryable ? el("button", { class: "btn", type: "button", onclick: () => handlers.onRetry(msg) }, "Повторить") : null));
  }
  if (msg.status !== "streaming") parts.push(renderMeta(msg, handlers.onShowMap));
  return el("div", { class: "msg-bot" }, parts);
}

/** Приветствие пустого чата с подсказками-примерами. onPick(text) — отправить пример. */
export function renderWelcome(onPick) {
  const examples = [
    ["Где поесть пельменей?", "поиск по смыслу — даже если не помните название"],
    ["Хочу суши рядом с Пушкинской улицей", "смысл + район на карте"],
    ["Я на Белорусской, 44. Где ближайшая аптека и до скольки работает?", "цепочка: адрес → ближайшее → карточка"],
    ["Сколько в Ростове круглосуточных аптек?", "точный подсчёт по базе"],
  ];
  return el("div", { class: "welcome" },
    el("div", { class: "logo" }, "🧭"),
    el("h2", {}, "Чем помочь в Ростове?"),
    el("p", {}, "Кафе, аптеки, магазины, пункты выдачи и ещё 8 тысяч мест из OpenStreetMap. Спрашивайте своими словами — уточнять можно по ходу разговора."),
    el("div", { class: "chips" }, examples.map(([q, hint]) =>
      el("button", { class: "chip", type: "button", onclick: () => onPick(q) }, el("b", {}, q), el("span", {}, hint)))));
}

/** «Сегодня 14:05», «вчера», «12 окт.» — время чата в боковой панели. */
export function shortTime(ts) {
  const d = new Date(ts * 1000), now = new Date();
  const sameDay = (a, b) => a.toDateString() === b.toDateString();
  if (sameDay(d, now)) return d.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (sameDay(d, y)) return "вчера";
  return d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
}

export { el };
