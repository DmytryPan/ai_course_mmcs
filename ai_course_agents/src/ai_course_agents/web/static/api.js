// Обращения к бэкенду: обычные JSON-запросы и чтение потока SSE.
// Токен входа живёт в httpOnly-cookie — JavaScript его не видит и не передаёт сам:
// браузер прикладывает cookie к запросам на тот же сайт автоматически.

/** Ошибка «нужно войти» (HTTP 401): приложение по ней показывает экран входа. */
export class AuthError extends Error {}

/** Ошибка API с понятным текстом из ответа сервера (поле detail у FastAPI). */
export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

/** Достаёт человеческое сообщение из ответа FastAPI: detail — строка или список ошибок проверки. */
async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      // 422: pydantic перечисляет, какие поля не прошли проверку
      return body.detail.map((d) => fieldHint(d)).join("; ");
    }
  } catch { /* тело не JSON — вернём общий текст ниже */ }
  return `ошибка сервера (${response.status})`;
}

/** Ошибки проверки полей логина/пароля — по-русски, а не текстом pydantic. */
function fieldHint(d) {
  const field = d.loc?.[d.loc.length - 1];
  if (field === "username") return "логин: 3–32 символа, буквы, цифры, . _ -";
  if (field === "password") return "пароль: от 6 до 72 символов";
  if (field === "text") return "вопрос: от 1 до 1000 символов";
  return d.msg;
}

/** JSON-запрос. 401 → AuthError, другие ошибки → ApiError с текстом от сервера. */
export async function request(method, url, body) {
  const response = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (response.status === 401 && !url.startsWith("/api/auth/login")) throw new AuthError();
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return response.status === 204 ? null : response.json();   // 204 No Content — тела нет
}

/**
 * Задать вопрос и читать ответ агента потоком SSE.
 *
 * Встроенный в браузер EventSource умеет только GET, а вопрос уходит в теле POST,
 * поэтому поток читаем вручную: fetch → ReadableStream → куски байт → строки → события.
 * Формат SSE: события разделены пустой строкой; в событии строки «event: тип» и «data: JSON».
 *
 * @param {string} chatId
 * @param {string} text           вопрос пользователя
 * @param {(type: string, data: object) => void} onEvent  вызывается на каждое событие
 * @param {AbortSignal} signal     abort() — кнопка «Стоп»: соединение рвётся, сервер отменяет прогон
 */
export async function streamAnswer(chatId, text, onEvent, signal) {
  const response = await fetch(`/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ text }),
    signal,
  });
  if (response.status === 401) throw new AuthError();
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");     // байты → текст; stream:true — не рвать символ посередине
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    // sse-starlette разделяет строки «\r\n» — приводим к «\n», чтобы разбор был один
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {   // в буфере есть целое событие
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let type = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        // строки, начинающиеся с «:», — служебный ping сервера, их пропускаем
      }
      if (data.length) onEvent(type, JSON.parse(data.join("\n")));
    }
  }
}
