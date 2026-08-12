import { refs } from "./core/dom.js?v=20260617-question-search-04";
import { api } from "./core/api.js?v=20260617-question-search-04";
import { tg } from "./core/telegram.js?v=20260523-cases-search-02";
import { escapeHtml } from "./core/ui.js?v=20260809-unified-ui-02";

const BASE = "/api/admin/apk-import/sessions";
let token = "";
let selectedSection = "";
let query = "";
let offset = 0;
let suggestedTitle = "";
let publishing = false;

function content() { return refs.mainPanel?.querySelector(".apk-import-content"); }
function message(text, error = false) {
  const node = refs.mainPanel?.querySelector(".apk-import-message");
  if (node) { node.textContent = text; node.classList.toggle("is-error", error); }
}

export function renderAdminApkImport(ctx) {
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content apk-import-screen">
      <h1 class="page-title">Питання з APK</h1>
      <p class="page-subtitle">APK, XAPK або APKS · до 100 MiB</p>
      <div class="apk-import-content"></div>
      <div class="apk-import-message" role="status" aria-live="polite"></div>
    </section>
  `;
  renderUpload();
}

export async function cleanupAdminApkImport() {
  if (token) { try { await api(`${BASE}/${token}`, { method: "DELETE" }); } catch {} }
  token = "";
  selectedSection = "";
  query = "";
  offset = 0;
  suggestedTitle = "";
  publishing = false;
}

function renderUpload() {
  const root = content();
  if (!root) return;
  root.innerHTML = `
    <form id="apk-upload-form">
      <div class="group">
        <div class="group__label">Файл із питаннями</div>
        <div class="group__list">
          <label class="cell apk-file-picker" for="apk-file">
            <span class="cell__icon cell__icon--green">APK</span>
            <span class="cell__body">
              <span class="cell__title">Вибрати файл</span>
              <span class="cell__subtitle" id="apk-file-name">APK, XAPK або APKS</span>
            </span>
            <span class="cell__chevron" aria-hidden="true"></span>
          </label>
        </div>
        <div class="group__footer">Максимальний розмір файлу — 100 MiB.</div>
      </div>
      <input class="apk-file-input" id="apk-file" type="file" accept=".apk,.xapk,.apks" required aria-label="Вибрати APK">
      <button class="btn btn--primary btn--block" type="submit" disabled>Перевірити файл</button>
    </form>
  `;
  const input = root.querySelector("#apk-file");
  const submit = root.querySelector('button[type="submit"]');
  input.addEventListener("change", () => {
    const file = input.files[0];
    root.querySelector("#apk-file-name").textContent = file?.name || "APK, XAPK або APKS";
    submit.disabled = !file;
  });
  root.querySelector("form").addEventListener("submit", upload);
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
    message("");
  } catch (error) { message(error.message, true); }
}

function renderBanks(banks) {
  content().innerHTML = `<div class="group"><div class="group__label">Оберіть банк</div><div class="group__list apk-bank-list"></div><div class="group__footer">Натисніть на банк, щоб витягнути й перевірити питання.</div></div>`;
  const list = content().querySelector(".apk-bank-list");
  banks.forEach((bank) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "cell apk-bank"; button.disabled = !bank.supported;
    button.innerHTML = `<span class="cell__icon cell__icon--green">${escapeHtml(String(bank.title || bank.filename || "Б").trim().charAt(0).toUpperCase())}</span><span class="cell__body"><span class="cell__title">${escapeHtml(bank.title || bank.filename)}</span><span class="cell__subtitle">${bank.supported ? "Витягнути та переглянути питання" : `Поки не підтримується · ${escapeHtml(bank.filename)}`}</span></span><span class="cell__chevron"></span>`;
    if (bank.supported) button.addEventListener("click", (event) => parse(bank.id, event.currentTarget));
    list.append(button);
  });
}

async function parse(bankId, trigger) {
  if (trigger?.disabled) return;
  if (trigger) trigger.disabled = true;
  message("Розшифровуємо питання…");
  try {
    const parsed = await api(`${BASE}/${token}/banks/${bankId}/parse`, { method: "POST", timeoutMs: 120000 });
    suggestedTitle = parsed.suggested_title || "Новий тест";
    selectedSection = ""; query = ""; offset = 0;
    await preview();
  } catch (error) {
    if (trigger) trigger.disabled = false;
    message(error.message, true);
  }
}

async function preview() {
  try {
    const data = await api(`${BASE}/${token}/preview?section=${encodeURIComponent(selectedSection)}&q=${encodeURIComponent(query)}&offset=${offset}&limit=25`);
    content().innerHTML = `<div class="apk-summary"><strong>${data.count} питань</strong><a class="btn btn--sm" id="apk-download" href="#">Завантажити JSON</a></div><div class="apk-publish"><label>Назва нового розділу<input id="apk-publish-title" value="${escapeHtml(suggestedTitle)}" maxlength="160" autocomplete="off"></label><button class="btn btn--primary btn--lg" id="apk-publish" type="button">Створити розділ</button></div><div class="apk-filters"><label>Розділ<select id="apk-section"><option value="">Усі розділи</option>${data.sections.map((item) => `<option value="${escapeHtml(item.title)}" ${item.title === selectedSection ? "selected" : ""}>${escapeHtml(item.title)} (${item.questions_count})</option>`).join("")}</select></label><label>Пошук питань<input id="apk-search" value="${escapeHtml(query)}" placeholder="Пошук питань"></label></div><div class="apk-question-list">${data.items.map(renderQuestion).join("")}</div><div class="apk-pagination"><button class="btn btn--sm" id="apk-prev" ${!data.has_prev ? "disabled" : ""}>Назад</button><span>${data.total} знайдено</span><button class="btn btn--sm" id="apk-next" ${!data.has_next ? "disabled" : ""}>Далі</button></div>`;
    content().querySelector("#apk-section").addEventListener("change", (e) => { selectedSection = e.target.value; offset = 0; preview(); });
    content().querySelector("#apk-search").addEventListener("change", (e) => { query = e.target.value.trim(); offset = 0; preview(); });
    content().querySelector("#apk-prev").addEventListener("click", () => { offset = Math.max(0, offset - 25); preview(); });
    content().querySelector("#apk-next").addEventListener("click", () => { offset += 25; preview(); });
    content().querySelector("#apk-download").addEventListener("click", download);
    content().querySelector("#apk-publish").addEventListener("click", createSection);
    message("");
  } catch (error) { message(error.message, true); }
}

async function createSection() {
  if (publishing) return;
  const titleInput = content().querySelector("#apk-publish-title");
  const button = content().querySelector("#apk-publish");
  const title = titleInput?.value.trim() || "";
  if (!title) { message("Вкажіть назву нового розділу.", true); titleInput?.focus(); return; }
  publishing = true;
  if (button) { button.disabled = true; button.textContent = "Створюємо…"; }
  message("Створюємо повноцінний тестовий розділ…");
  try {
    const result = await api(`${BASE}/${token}/publish`, {
      method: "POST",
      body: { title },
      timeoutMs: 120000,
    });
    content().innerHTML = `<div class="apk-publish-success"><strong>${escapeHtml(result.title)}</strong><span>${result.count} питань успішно додано.</span><button class="btn btn--primary btn--lg" id="apk-open-created" type="button">Відкрити розділ</button></div>`;
    content().querySelector("#apk-open-created").addEventListener("click", () => {
      window.sessionStorage.setItem("openAttestationBank", result.slug);
      window.location.reload();
    });
    message("Розділ створено.");
  } catch (error) {
    publishing = false;
    if (button) { button.disabled = false; button.textContent = "Створити розділ"; }
    message(error.message, true);
  }
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
