// Карта Leaflet: метки мест из результатов инструментов агента.
// L — глобальный объект библиотеки Leaflet (подключена тегом <script> в index.html).

const ROSTOV = [47.235, 39.715];       // центр Ростова-на-Дону: [широта, долгота]

let map = null;                        // объект карты; создаётся при первом показе
let layer = null;                      // группа текущих меток: очищается целиком при смене мест
let markers = [];                      // метки по порядку — чтобы открыть нужную по клику в ответе
let lastBounds = [];                   // координаты текущих меток — для повторной подгонки масштаба

// Плитки — стандартные OpenStreetMap: работают без ключа. (CARTO из браузера без ключа API
// рисует поверх карты водяной знак «API KEY REQUIRED» — проверено.) Тёмная тема получается
// CSS-фильтром инверсии цветов (styles.css → .leaflet-tile-pane), отдельные тёмные плитки не нужны.
const TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

/** Создать карту в элементе #map (один раз). */
export function initMap() {
  if (map) return;
  // zoomControl: false + отдельный контрол справа внизу — чтобы кнопки «+/−» не закрывали подпись сверху
  map = L.map("map", { zoomControl: false }).setView(ROSTOV, 12);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.tileLayer(TILES, {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  layer = L.layerGroup().addTo(map);
}

/** Leaflet считает размер карты при создании; после показа скрытой панели (мобильная
 *  вёрстка) размер надо пересчитать, иначе карта нарисуется кусками. И заново подогнать
 *  масштаб под метки: пока панель была скрыта (размер 0×0), подгонка давала неверный вид. */
export function refreshMapSize() {
  if (!map) return;
  setTimeout(() => { map.invalidateSize(); fit(); }, 50);
}

/** Подогнать масштаб так, чтобы все метки были видны; одну точку — крупно. */
function fit() {
  if (lastBounds.length === 1) map.setView(lastBounds[0], 16);
  else if (lastBounds.length > 1) map.fitBounds(lastBounds, { padding: [40, 40], maxZoom: 16 });
}

/** Экранирование текста для HTML всплывающей подсказки: названия мест — внешние данные. */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Текст всплывающей подсказки метки: название, адрес, часы, расстояние. */
function popupHtml(p) {
  const lines = [`<b>${esc(p.name || "без названия")}</b>`];
  if (p.kind) lines.push(`<span class="muted">${esc(p.kind)}</span>`);
  lines.push(esc(p.address || "адрес: нет данных"));
  if (p.hours) lines.push(`🕒 ${esc(p.hours)}`);
  if (p.distance_m != null) lines.push(`📏 ${p.distance_m} м`);
  return lines.join("<br>");
}

/**
 * Показать места на карте (заменяет прежние метки).
 * places — [{id, name, lat, lon, address?, hours?, distance_m?, is_origin?}]
 * caption — подпись над картой («Поиск по смыслу: 8 мест»)
 */
export function showPlaces(places, caption) {
  initMap();
  layer.clearLayers();
  markers = [];
  const bounds = [];
  let n = 0;
  for (const p of places) {
    if (p.lat == null || p.lon == null) continue;
    // Точка адреса пользователя («откуда считали») — отдельный значок без номера.
    const html = p.is_origin ? '<div class="pin-origin">📍</div>' : `<div class="pin"><span>${++n}</span></div>`;
    const icon = L.divIcon({ html, className: "", iconSize: [26, 26], iconAnchor: [13, 26], popupAnchor: [0, -24] });
    const marker = L.marker([p.lat, p.lon], { icon, zIndexOffset: p.is_origin ? 1000 : 0 })
      .bindPopup(popupHtml(p)).addTo(layer);
    markers.push({ place: p, marker });
    bounds.push([p.lat, p.lon]);
  }
  document.getElementById("map-caption").textContent =
    caption || (n ? `На карте: ${n} ${plural(n, "место", "места", "мест")}` : "Найденные места появятся на карте");
  lastBounds = bounds;
  fit();
}

/** Открыть метку места по названию (клик по названию в тексте ответа). */
export function focusPlace(name) {
  const key = normalize(name);
  const hit = markers.find((m) => normalize(m.place.name) === key);
  if (!hit) return false;
  map.setView(hit.marker.getLatLng(), Math.max(map.getZoom(), 15));
  hit.marker.openPopup();
  return true;
}

/** Сравнение названий без регистра, кавычек и лишних пробелов: «Антикафе "Тень"» = «антикафе «тень»». */
export function normalize(s) {
  return String(s ?? "").toLowerCase().replace(/[«»"'„“”]/g, "").replace(/\s+/g, " ").trim();
}

/** Русское склонение после числа: 1 место, 2 места, 5 мест. */
export function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
