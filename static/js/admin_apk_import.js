import { refs } from "./core/dom.js?v=20260617-question-search-04";
import { api } from "./core/api.js?v=20260617-question-search-04";
import { tg } from "./core/telegram.js?v=20260523-cases-search-02";
import { escapeHtml } from "./core/ui.js?v=20260809-unified-ui-02";

const ENTRY_ID = "admin-apk-import-entry";
const OVERLAY_ID = "admin-apk-import-overlay";
const BASE = "/api/admin/apk-import/sessions";
let token = "";
let selectedSection = "";
let query = "";
let offset = 0;

function ensureEntry() {
  const title = refs.mainPanel?.querySelector(".page-title")?.textContent.trim();
  const list = refs.mainPanel?.querySelector(".group__list");
  if (title !== "Адмін" || !list || document.querySelector(`#${ENTRY_ID}`)) return;
  const button = document.createElement("button");
  button.id = ENTRY_ID;
  button.type = "button";
  button.className = "cell";
  button.innerHTML = `<span class="cell__icon cell__icon--green">APK</span><span class="cell__body"><span class="cell__title">Витягнути питання з APK</span><span class="cell__subtitle">Знайти банк, перевірити та завантажити JSON</span></span><span class="cell__chevron"></span>`;
  button.addEventListener("click", open);
  list.prepend(button);
}

function content() { return document.querySelector(`#${OVERLAY_ID} .apk-import-content`); }
function message(text, error = false) {
  const node = document.querySelector(`#${OVERLAY_ID} .apk-import-message`);
  if (node) { node.textContent = text; node.classList.toggle("is-error", error); }
}

function open() {
  document.querySelector(`#${OVERLAY_ID}`)?.remove();
  const overlay = document.createElement("div");
  overlay.id = OVERLAY_ID;
  overlay.className = "modal-overlay apk-import-overlay";
  overlay.innerHTML = `<div class="modal apk-import-modal" role="dialog" aria-modal="true" aria-labelledby="apk-import-title"><div class="modal__header"><div class="modal__heading"><span class="modal__title" id="apk-import-title">Питання з APK</span><span class="modal__subtitle">APK, XAPK або APKS · до 50 MiB</span></div><button class="modal__close" type="button" aria-label="Закрити">✕</button></div><div class="apk-import-content"></div><div class="apk-import-message" aria-live="polite"></div></div>`;
  document.body.append(overlay);
  overlay.querySelector(".modal__close").addEventListener("click", close);
  renderUpload();
}

async function close() {
  if (token) { try { await api(`${BASE}/${token}`, { method: "DELETE" }); } catch {} }
  token = "";
  document.querySelector(`#${OVERLAY_ID}`)?.remove();
}

function renderUpload() {
  content().innerHTML = `<form id="apk-upload-form"><label class="apk-drop"><strong>Завантажити APK</strong><span>Оберіть файл із пристрою</span><input id="apk-file" type="file" accept=".apk,.xapk,.apks" required aria-label="Завантажити APK"></label><button class="btn btn--primary btn--lg" type="submit">Перевірити файл</button></form>`;
  content().querySelector("form").addEventListener("submit", upload);
}

async function upload(event) {
  event.preventDefault();
  const file = content().querySelector("#apk-file").files[0];
  if (!file) return;
  const form = new FormData(); form.append("file", file);
  message("Завантаження та перевірка…");
  try {
    const session = await api(BASE, { method: "POST", body: form, timeoutMs: 120000 });
    token = session.token;
    renderBanks(session.banks);
    tg?.HapticFeedback?.notificationOccurred?.("success");
  } catch (error) { message(error.message, true); }
}

function renderBanks(banks) {
  content().innerHTML = `<h2 class="apk-import-heading">Оберіть банк</h2><div class="apk-bank-list"></div>`;
  const list = content().querySelector(".apk-bank-list");
  banks.forEach((bank) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "cell"; button.disabled = !bank.supported;
    button.innerHTML = `<span class="cell__body"><span class="cell__title">${escapeHtml(bank.filename)}</span><span class="cell__subtitle">${bank.supported ? "Підтримується" : "Поки не підтримується"}</span></span><span class="cell__chevron"></span>`;
    if (bank.supported) button.addEventListener("click", () => parse(bank.id));
    list.append(button);
  });
}

async function parse(bankId) {
  message("Розшифровуємо питання…");
  try {
    await api(`${BASE}/${token}/banks/${bankId}/parse`, { method: "POST", timeoutMs: 120000 });
    selectedSection = ""; query = ""; offset = 0;
    await preview();
  } catch (error) { message(error.message, true); }
}

async function preview() {
  try {
    const data = await api(`${BASE}/${token}/preview?section=${encodeURIComponent(selectedSection)}&q=${encodeURIComponent(query)}&offset=${offset}&limit=25`);
    content().innerHTML = `<div class="apk-summary"><strong>${data.count} питань</strong><a class="btn btn--sm" id="apk-download" href="#">Завантажити JSON</a></div><div class="apk-filters"><label>Розділ<select id="apk-section"><option value="">Усі розділи</option>${data.sections.map((item) => `<option value="${escapeHtml(item.title)}" ${item.title === selectedSection ? "selected" : ""}>${escapeHtml(item.title)} (${item.questions_count})</option>`).join("")}</select></label><label>Пошук питань<input id="apk-search" value="${escapeHtml(query)}" placeholder="Пошук питань"></label></div><div class="apk-question-list">${data.items.map(renderQuestion).join("")}</div><div class="apk-pagination"><button class="btn btn--sm" id="apk-prev" ${!data.has_prev ? "disabled" : ""}>Назад</button><span>${data.total} знайдено</span><button class="btn btn--sm" id="apk-next" ${!data.has_next ? "disabled" : ""}>Далі</button></div>`;
    content().querySelector("#apk-section").addEventListener("change", (e) => { selectedSection = e.target.value; offset = 0; preview(); });
    content().querySelector("#apk-search").addEventListener("change", (e) => { query = e.target.value.trim(); offset = 0; preview(); });
    content().querySelector("#apk-prev").addEventListener("click", () => { offset = Math.max(0, offset - 25); preview(); });
    content().querySelector("#apk-next").addEventListener("click", () => { offset += 25; preview(); });
    content().querySelector("#apk-download").addEventListener("click", download);
    message("");
  } catch (error) { message(error.message, true); }
}

function renderQuestion(item) {
  const correct = new Set(item.correct);
  return `<article class="apk-question"><div><strong>${item.qnum}. ${escapeHtml(item.question)}</strong>${item.shuffle_choices ? "" : '<span class="apk-badge">Без перемішування</span>'}</div><ol>${item.choices.map((choice, index) => `<li class="${correct.has(index + 1) ? "is-correct" : ""}">${escapeHtml(choice)}${correct.has(index + 1) ? " · Правильна відповідь" : ""}</li>`).join("")}</ol></article>`;
}

function download(event) {
  event.preventDefault();
  const url = new URL(`${BASE}/${token}/download`, window.location.origin);
  const params = new URLSearchParams(window.location.search);
  if (tg?.initData) url.searchParams.set("initData", tg.initData);
  else if (params.get("dev_user_id")) url.searchParams.set("dev_user_id", params.get("dev_user_id"));
  window.location.href = url.href;
}

new MutationObserver(ensureEntry).observe(refs.mainPanel, { childList: true, subtree: true });
ensureEntry();
