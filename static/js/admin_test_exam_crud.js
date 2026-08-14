import { api } from "./core/api.js?v=20260617-question-search-04";

let overlay = null;
let savedOverflow = "";
let importFile = null;
let importPreview = null;
let importEdits = {};
let importResolutions = {};
let importConfirmed = false;
let importBusy = false;

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function ensureStyles() {
  if (document.querySelector("#test-question-manager-styles")) return;
  const style = document.createElement("style");
  style.id = "test-question-manager-styles";
  style.textContent = `
    .test-q-manager-actions {
      display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:12px 0 16px;
    }
    .test-q-manager-actions .btn { min-height:48px; }
    #test-q-list.test-q-manager-list { gap:10px; }
    #test-q-list.test-q-manager-list .case-answer {
      padding:14px 16px; border-radius:16px; cursor:pointer;
    }
    #test-q-list.test-q-manager-list .case-answer__head { margin-bottom:6px; }
    #test-q-list.test-q-manager-list .case-answer__question {
      font-size:16px !important; line-height:1.35 !important; margin:6px 0 10px !important;
    }
    #test-q-list.test-q-manager-list .case-answer__label { margin-top:0; font-size:11px; }
    #test-q-list.test-q-manager-list .case-answer__correct { padding:9px 10px; border-radius:12px; }
    #test-q-list.test-q-manager-list .case-answer__correct-text { font-size:14px; line-height:1.35; }
    .test-q-manager-edit-note { margin-top:8px; font-size:12px; opacity:.55; font-weight:650; }

    .test-q-manager-overlay {
      position:fixed; inset:0; z-index:30000; background:var(--bg,#f4f5f7);
      overflow-y:auto; overscroll-behavior:contain;
    }
    .test-q-manager-shell { width:min(760px,100%); min-height:100%; margin:0 auto; }
    .test-q-manager-bar {
      position:sticky; top:0; z-index:3; display:flex; align-items:center; gap:12px;
      padding:calc(10px + env(safe-area-inset-top)) 14px 10px;
      background:var(--bg,#f4f5f7); border-bottom:1px solid var(--separator,rgba(128,128,128,.18));
    }
    .test-q-manager-back {
      border:0; background:var(--bg-fill-soft,rgba(128,128,128,.12)); border-radius:999px;
      padding:10px 14px; font:inherit; font-weight:700; color:inherit;
    }
    .test-q-manager-title { font-size:20px; font-weight:800; line-height:1.15; }
    .test-q-manager-subtitle { margin-top:2px; font-size:12px; opacity:.62; }
    .test-q-manager-body { padding:16px 14px calc(28px + env(safe-area-inset-bottom)); display:grid; gap:14px; }
    .test-q-manager-card {
      background:var(--bg-elevated,#fff); border:1px solid var(--separator,rgba(128,128,128,.18));
      border-radius:18px; padding:14px; display:grid; gap:12px;
    }
    .test-q-manager-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .test-q-manager-field { display:grid; gap:6px; }
    .test-q-manager-field span { font-size:12px; font-weight:700; opacity:.68; }
    .test-q-manager-input, .test-q-manager-textarea {
      width:100%; box-sizing:border-box; border:1px solid var(--separator,rgba(128,128,128,.25));
      border-radius:12px; background:var(--bg,#f5f6f8); color:inherit; font:inherit; padding:12px; outline:none;
    }
    .test-q-manager-textarea { min-height:116px; resize:vertical; line-height:1.4; }
    .test-q-manager-answer { min-height:90px; }
    .test-q-manager-status { min-height:18px; font-size:13px; line-height:1.4; }
    .test-q-manager-status--error { color:var(--danger,#c33); }
    .test-q-manager-status--success { color:var(--success,#16843d); }
    .test-q-manager-danger { border:1px solid var(--danger,#c33) !important; color:var(--danger,#c33) !important; background:transparent !important; }

    .test-q-import-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
    .test-q-import-stat { text-align:center; padding:10px 6px; background:var(--bg-fill-soft); border-radius:12px; }
    .test-q-import-stat strong { display:block; font-size:20px; }
    .test-q-import-stat span { display:block; margin-top:2px; font-size:11px; opacity:.65; }
    .test-q-import-section { display:grid; gap:10px; }
    .test-q-import-item { border:1px solid var(--separator,rgba(128,128,128,.18)); border-radius:14px; padding:12px; display:grid; gap:9px; }
    .test-q-import-meta { font-size:12px; opacity:.62; }
    .test-q-import-question { font-weight:750; line-height:1.38; }
    .test-q-import-answer { padding:9px 10px; background:var(--bg-fill-soft); border-radius:10px; font-size:13px; line-height:1.4; }
    .test-q-import-answer b { display:block; margin-bottom:3px; font-size:10px; text-transform:uppercase; opacity:.62; }
    .test-q-import-conflict { padding:10px 12px; border-radius:12px; background:var(--bg-fill-soft); font-size:13px; line-height:1.4; }
    .test-q-import-radio { display:flex; gap:8px; align-items:flex-start; font-size:13px; line-height:1.35; }
    .test-q-import-confirm { display:flex; gap:10px; align-items:flex-start; font-size:14px; line-height:1.4; }
    .test-q-manager-file { font-size:13px; opacity:.7; overflow-wrap:anywhere; }

    @media (max-width:560px) {
      .test-q-manager-grid { grid-template-columns:1fr; }
      .test-q-manager-actions { grid-template-columns:1fr 1fr; }
      .test-q-manager-actions .btn { padding-left:8px; padding-right:8px; font-size:14px; }
    }
  `;
  document.head.append(style);
}

function managerStatus(message, tone = "") {
  const node = document.querySelector("#test-q-manager-inline-status");
  if (!node) return;
  node.className = `test-q-manager-status${tone ? ` test-q-manager-status--${tone}` : ""}`;
  node.textContent = message || "";
}

function overlayStatus(message, tone = "") {
  const node = document.querySelector("#test-q-overlay-status");
  if (!node) return;
  node.className = `test-q-manager-status${tone ? ` test-q-manager-status--${tone}` : ""}`;
  node.textContent = message || "";
}

function openOverlay(title, subtitle = "") {
  closeOverlay();
  ensureStyles();
  savedOverflow = document.body.style.overflow;
  document.body.style.overflow = "hidden";
  overlay = document.createElement("section");
  overlay.className = "test-q-manager-overlay";
  overlay.innerHTML = `
    <div class="test-q-manager-shell">
      <div class="test-q-manager-bar">
        <button class="test-q-manager-back" id="test-q-manager-back" type="button">← Назад</button>
        <div>
          <div class="test-q-manager-title">${esc(title)}</div>
          ${subtitle ? `<div class="test-q-manager-subtitle">${esc(subtitle)}</div>` : ""}
        </div>
      </div>
      <div class="test-q-manager-body" id="test-q-manager-body"></div>
    </div>
  `;
  document.body.append(overlay);
  const backButton = overlay.querySelector("#test-q-manager-back");
  if (backButton) backButton.onclick = closeOverlay;
  return overlay.querySelector("#test-q-manager-body");
}

function closeOverlay() {
  overlay?.remove();
  overlay = null;
  document.body.style.overflow = savedOverflow;
}

function refreshList() {
  const input = document.querySelector("#test-q-input");
  if (input) input.dispatchEvent(new Event("input", { bubbles: true }));
}

function questionPayload() {
  const val = (id) => document.querySelector(id)?.value ?? "";
  return {
    num: val("#tqm-num"),
    module: val("#tqm-module"),
    question: val("#tqm-question"),
    correct_answer: val("#tqm-answer"),
    source: val("#tqm-source"),
    justification: val("#tqm-justification"),
  };
}

function renderQuestionEditor(item = null) {
  const isEdit = Boolean(item?.id);
  const body = openOverlay(isEdit ? "Редагувати питання" : "Додати питання", isEdit ? `ID ${item.id}` : "Тестові питання");
  body.innerHTML = `
    <section class="test-q-manager-card">
      <div class="test-q-manager-grid">
        <label class="test-q-manager-field"><span>Номер</span><input class="test-q-manager-input" id="tqm-num" value="${esc(item?.num || "")}" placeholder="Напр. 61" /></label>
        <label class="test-q-manager-field"><span>Модуль</span><input class="test-q-manager-input" id="tqm-module" value="${esc(item?.module || "")}" placeholder="Необов’язково" /></label>
      </div>
      <label class="test-q-manager-field"><span>Питання *</span><textarea class="test-q-manager-textarea" id="tqm-question">${esc(item?.question || "")}</textarea></label>
      <label class="test-q-manager-field"><span>Правильна відповідь *</span><textarea class="test-q-manager-textarea test-q-manager-answer" id="tqm-answer">${esc(item?.correct_answer || "")}</textarea></label>
      <label class="test-q-manager-field"><span>Джерело</span><input class="test-q-manager-input" id="tqm-source" value="${esc(item?.source || "Вручну")}" /></label>
      <label class="test-q-manager-field"><span>Обґрунтування / примітка</span><textarea class="test-q-manager-textarea" id="tqm-justification">${esc(item?.justification || "")}</textarea></label>
      <div class="test-q-manager-status" id="test-q-overlay-status"></div>
      <button class="btn btn--primary btn--block" id="tqm-save" type="button">${isEdit ? "Зберегти зміни" : "Додати питання"}</button>
      ${isEdit ? '<button class="btn btn--block test-q-manager-danger" id="tqm-delete" type="button">Видалити питання</button>' : ""}
    </section>
  `;

  body.querySelector("#tqm-save")?.addEventListener("click", async (event) => {
    const payload = questionPayload();
    if (!String(payload.question).trim()) return overlayStatus("Введіть текст питання.", "error");
    if (!String(payload.correct_answer).trim()) return overlayStatus("Введіть правильну відповідь.", "error");
    const button = event.currentTarget;
    button.disabled = true;
    overlayStatus(isEdit ? "Зберігаю…" : "Додаю…");
    try {
      const result = await api(isEdit ? `/api/admin/test-exam-questions/crud/${item.id}` : "/api/admin/test-exam-questions/crud", {
        method: isEdit ? "PATCH" : "POST",
        body: payload,
        timeoutMs: 20000,
      });
      closeOverlay();
      managerStatus(isEdit ? "Питання оновлено." : `Питання додано (ID ${result.item.id}).`, "success");
      refreshList();
    } catch (error) {
      overlayStatus(error.message || "Не вдалося зберегти питання.", "error");
      button.disabled = false;
    }
  });

  body.querySelector("#tqm-delete")?.addEventListener("click", async (event) => {
    if (!window.confirm("Видалити це питання?")) return;
    const button = event.currentTarget;
    button.disabled = true;
    overlayStatus("Видаляю…");
    try {
      await api(`/api/admin/test-exam-questions/crud/${item.id}`, { method: "DELETE", timeoutMs: 20000 });
      closeOverlay();
      managerStatus("Питання видалено.", "success");
      refreshList();
    } catch (error) {
      overlayStatus(error.message || "Не вдалося видалити питання.", "error");
      button.disabled = false;
    }
  });
}

async function editCard(card) {
  if (overlay) return;
  const question = card.querySelector(".case-answer__question")?.textContent?.trim() || "";
  const num = card.querySelector(".case-answer__number")?.textContent?.trim() || "";
  const answer = card.querySelector(".case-answer__correct-text")?.textContent?.trim() || "";
  const module = card.querySelector(".case-answer__count")?.textContent?.trim() || "";
  if (!question) return;
  managerStatus("Відкриваю питання…");
  try {
    const params = new URLSearchParams({ question, num, answer, module });
    const result = await api(`/api/admin/test-exam-questions/crud/lookup?${params.toString()}`, { timeoutMs: 15000 });
    managerStatus("");
    renderQuestionEditor(result.item);
  } catch (error) {
    managerStatus(error.message || "Не вдалося відкрити питання.", "error");
  }
}

function bindCards() {
  const list = document.querySelector("#test-q-list");
  if (!list) return;
  list.classList.add("test-q-manager-list");
  list.querySelectorAll(".case-answer").forEach((card) => {
    if (card.dataset.tqmBound === "1") return;
    card.dataset.tqmBound = "1";
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    const note = document.createElement("div");
    note.className = "test-q-manager-edit-note";
    note.textContent = "Натисніть, щоб редагувати";
    card.append(note);
    card.addEventListener("click", () => void editCard(card));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        void editCard(card);
      }
    });
  });
}

function fmtBytes(size) {
  const value = Number(size || 0);
  if (value < 1024) return `${value} Б`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} КБ`;
  return `${(value / 1024 / 1024).toFixed(1)} МБ`;
}

async function previewImport() {
  if (!importFile || importBusy) return;
  importBusy = true;
  overlayStatus("Перевіряю файл…");
  try {
    const form = new FormData();
    form.append("file", importFile, importFile.name);
    form.append("edits", JSON.stringify(importEdits));
    importPreview = await api("/api/admin/test-exam-questions/import/preview-edited", {
      method: "POST",
      body: form,
      timeoutMs: 30000,
    });
    importConfirmed = false;
    renderImportHome();
  } catch (error) {
    overlayStatus(error.message || "Не вдалося перевірити JSON.", "error");
  } finally {
    importBusy = false;
    updateImportApply();
  }
}

function importedItemByIndex(index) {
  const target = Number(index);
  const all = [
    ...(importPreview?.new || []),
    ...(importPreview?.conflicts || []).map((entry) => entry.imported),
    ...(importPreview?.duplicates || []).map((entry) => entry.imported),
  ];
  return all.find((item) => Number(item?.import_index) === target) || null;
}

function importMeta(item) {
  return [item?.num, item?.module, item?.source].filter(Boolean).join(" · ");
}

function renderImportEditor(index) {
  const item = importedItemByIndex(index);
  if (!item) return;
  const body = document.querySelector("#test-q-manager-body");
  if (!body) return;
  const back = document.querySelector("#test-q-manager-back");
  const title = document.querySelector(".test-q-manager-title");
  const subtitle = document.querySelector(".test-q-manager-subtitle");
  if (title) title.textContent = "Редагувати перед імпортом";
  if (subtitle) subtitle.textContent = item.num || "Нове питання";
  body.innerHTML = `
    <section class="test-q-manager-card">
      <div class="test-q-manager-grid">
        <label class="test-q-manager-field"><span>Номер</span><input class="test-q-manager-input" id="tqi-num" value="${esc(item.num || "")}" /></label>
        <label class="test-q-manager-field"><span>Модуль</span><input class="test-q-manager-input" id="tqi-module" value="${esc(item.module || "")}" /></label>
      </div>
      <label class="test-q-manager-field"><span>Питання *</span><textarea class="test-q-manager-textarea" id="tqi-question">${esc(item.question || "")}</textarea></label>
      <label class="test-q-manager-field"><span>Правильна відповідь *</span><textarea class="test-q-manager-textarea test-q-manager-answer" id="tqi-answer">${esc(item.correct_answer || "")}</textarea></label>
      <label class="test-q-manager-field"><span>Джерело</span><input class="test-q-manager-input" id="tqi-source" value="${esc(item.source || "")}" /></label>
      <label class="test-q-manager-field"><span>Обґрунтування</span><textarea class="test-q-manager-textarea" id="tqi-justification">${esc(item.justification || "")}</textarea></label>
      <div class="test-q-manager-status" id="test-q-overlay-status"></div>
      <button class="btn btn--primary btn--block" id="tqi-save" type="button">Зберегти і перевірити знову</button>
      ${Object.prototype.hasOwnProperty.call(importEdits, String(index)) ? '<button class="btn btn--block" id="tqi-reset" type="button">Скасувати мої зміни</button>' : ""}
    </section>
  `;

  const returnHome = () => {
    if (title) title.textContent = "Імпорт JSON";
    if (subtitle) subtitle.textContent = "Перевірка перед додаванням";
    renderImportHome();
  };
  if (back) back.onclick = returnHome;

  body.querySelector("#tqi-save")?.addEventListener("click", async (event) => {
    const question = document.querySelector("#tqi-question")?.value.trim() || "";
    const answer = document.querySelector("#tqi-answer")?.value.trim() || "";
    if (!question) return overlayStatus("Введіть текст питання.", "error");
    if (!answer) return overlayStatus("Введіть правильну відповідь.", "error");
    importEdits[String(index)] = {
      num: document.querySelector("#tqi-num")?.value.trim() || "",
      module: document.querySelector("#tqi-module")?.value.trim() || "",
      question,
      correct_answer: answer,
      source: document.querySelector("#tqi-source")?.value.trim() || "",
      justification: document.querySelector("#tqi-justification")?.value.trim() || "",
    };
    delete importResolutions[String(index)];
    importConfirmed = false;
    event.currentTarget.disabled = true;
    await previewImport();
  });

  body.querySelector("#tqi-reset")?.addEventListener("click", async (event) => {
    delete importEdits[String(index)];
    delete importResolutions[String(index)];
    importConfirmed = false;
    event.currentTarget.disabled = true;
    await previewImport();
  });
}

function conflictHtml(conflict) {
  const imported = conflict.imported || {};
  const existing = conflict.existing || {};
  const index = String(imported.import_index);
  if (conflict.kind === "file_answer_conflict") {
    return `
      <div class="test-q-import-item">
        <div class="test-q-import-question">${esc(imported.question)}</div>
        <div class="test-q-import-conflict">У самому JSON це питання повторюється з іншою відповіддю. Відредагуйте один із варіантів, щоб прибрати конфлікт.</div>
        <button class="btn btn--block" type="button" data-import-edit="${index}">Редагувати цей варіант</button>
      </div>
    `;
  }
  const isSimilar = conflict.match_type === "similar";
  const name = `tqi-conflict-${index}`;
  return `
    <div class="test-q-import-item">
      <div class="test-q-import-question">${esc(imported.question)}</div>
      ${isSimilar && existing.question && existing.question !== imported.question ? `<div class="test-q-import-answer"><b>Схоже питання в базі</b>${esc(existing.question)}</div>` : ""}
      <div class="test-q-import-answer"><b>У базі</b>${esc(existing.correct_answer || "—")}</div>
      <div class="test-q-import-answer"><b>У файлі</b>${esc(imported.correct_answer || "—")}</div>
      <label class="test-q-import-radio"><input type="radio" name="${name}" data-conflict-index="${index}" value="keep_existing" ${importResolutions[index] === "keep_existing" ? "checked" : ""} /> <span>Залишити відповідь із бази</span></label>
      <label class="test-q-import-radio"><input type="radio" name="${name}" data-conflict-index="${index}" value="use_imported" ${importResolutions[index] === "use_imported" ? "checked" : ""} /> <span>Взяти відповідь із файлу</span></label>
      ${isSimilar ? `<label class="test-q-import-radio"><input type="radio" name="${name}" data-conflict-index="${index}" value="add_new" ${importResolutions[index] === "add_new" ? "checked" : ""} /> <span>Це інше питання — додати окремо</span></label>` : ""}
      <button class="btn btn--block" type="button" data-import-edit="${index}">Редагувати варіант із файлу</button>
    </div>
  `;
}

function updateImportApply() {
  const button = document.querySelector("#tqi-apply");
  if (!button || !importPreview) return;
  const conflicts = importPreview.conflicts || [];
  const fileConflicts = conflicts.some((entry) => entry.kind === "file_answer_conflict");
  const unresolved = conflicts.some((entry) => entry.kind !== "file_answer_conflict" && !importResolutions[String(entry.imported.import_index)]);
  const newCount = Number(importPreview.new_count || 0);
  const actionableConflict = conflicts.some((entry) => {
    const value = importResolutions[String(entry.imported.import_index)];
    return value === "use_imported" || value === "add_new";
  });
  const hasWork = newCount > 0 || actionableConflict;
  button.disabled = importBusy || fileConflicts || unresolved || (newCount > 0 && !importConfirmed) || !hasWork;
}

function renderImportHome() {
  const body = document.querySelector("#test-q-manager-body");
  if (!body) return;
  const back = document.querySelector("#test-q-manager-back");
  const title = document.querySelector(".test-q-manager-title");
  const subtitle = document.querySelector(".test-q-manager-subtitle");
  if (back) back.onclick = closeOverlay;
  if (title) title.textContent = "Імпорт JSON";
  if (subtitle) subtitle.textContent = "Перевірка перед додаванням";

  if (!importFile || !importPreview) {
    body.innerHTML = `
      <section class="test-q-manager-card">
        <div style="font-weight:750;">Виберіть JSON-файл</div>
        <div class="test-q-manager-file">Дублікати не додаються. Нові питання ви побачите тут перед імпортом і зможете відредагувати.</div>
        <input id="tqi-file" type="file" accept="application/json,.json" hidden />
        <button class="btn btn--primary btn--block" id="tqi-pick" type="button">Вибрати JSON-файл</button>
        <div class="test-q-manager-status" id="test-q-overlay-status"></div>
      </section>
    `;
    const input = body.querySelector("#tqi-file");
    body.querySelector("#tqi-pick")?.addEventListener("click", () => input?.click());
    input?.addEventListener("change", () => {
      const file = input.files?.[0];
      if (!file) return;
      importFile = file;
      importPreview = null;
      importEdits = {};
      importResolutions = {};
      importConfirmed = false;
      void previewImport();
    });
    return;
  }

  const newItems = importPreview.new || [];
  const conflicts = importPreview.conflicts || [];
  body.innerHTML = `
    <section class="test-q-manager-card">
      <div class="test-q-manager-file"><b>${esc(importPreview.file_name)}</b> · ${fmtBytes(importPreview.file_size)} · ${importPreview.valid_count} питань</div>
      <div class="test-q-import-summary">
        <div class="test-q-import-stat"><strong>${importPreview.new_count}</strong><span>нових</span></div>
        <div class="test-q-import-stat"><strong>${importPreview.duplicate_count}</strong><span>дублікатів</span></div>
        <div class="test-q-import-stat"><strong>${importPreview.conflict_count}</strong><span>конфліктів</span></div>
      </div>
      <button class="btn btn--block" id="tqi-change-file" type="button">Вибрати інший файл</button>
    </section>

    ${newItems.length ? `
      <section class="test-q-manager-card test-q-import-section">
        <div style="font-weight:800;">Нові питання (${newItems.length})</div>
        <div class="test-q-manager-file">Перегляньте їх. Кожне можна відкрити та змінити перед імпортом.</div>
        ${newItems.map((item) => `
          <div class="test-q-import-item">
            <div class="test-q-import-meta">${esc(importMeta(item) || "Нове питання")}${Object.prototype.hasOwnProperty.call(importEdits, String(item.import_index)) ? " · ✎ змінено" : ""}</div>
            <div class="test-q-import-question">${esc(item.question)}</div>
            <div class="test-q-import-answer"><b>Правильна відповідь</b>${esc(item.correct_answer || "—")}</div>
            <button class="btn btn--block" type="button" data-import-edit="${item.import_index}">Редагувати</button>
          </div>
        `).join("")}
        <label class="test-q-import-confirm"><input type="checkbox" id="tqi-confirm" ${importConfirmed ? "checked" : ""} /> <span>Я переглянув нові питання та відповіді. Імпортувати їх.</span></label>
      </section>
    ` : ""}

    ${conflicts.length ? `
      <section class="test-q-manager-card test-q-import-section">
        <div style="font-weight:800;">Конфлікти (${conflicts.length})</div>
        ${conflicts.map(conflictHtml).join("")}
      </section>
    ` : ""}

    <section class="test-q-manager-card">
      <div class="test-q-manager-status" id="test-q-overlay-status"></div>
      <button class="btn btn--primary btn--block" id="tqi-apply" type="button">Імпортувати перевірені</button>
    </section>
  `;

  body.querySelector("#tqi-change-file")?.addEventListener("click", () => {
    importFile = null;
    importPreview = null;
    importEdits = {};
    importResolutions = {};
    importConfirmed = false;
    renderImportHome();
  });
  body.querySelectorAll("[data-import-edit]").forEach((button) => {
    button.addEventListener("click", () => renderImportEditor(Number(button.dataset.importEdit)));
  });
  body.querySelector("#tqi-confirm")?.addEventListener("change", (event) => {
    importConfirmed = event.currentTarget.checked;
    updateImportApply();
  });
  body.querySelectorAll("[data-conflict-index]").forEach((input) => {
    input.addEventListener("change", () => {
      importResolutions[String(input.dataset.conflictIndex)] = input.value;
      updateImportApply();
    });
  });
  body.querySelector("#tqi-apply")?.addEventListener("click", applyImport);
  updateImportApply();
}

async function applyImport(event) {
  if (!importFile || !importPreview || importBusy) return;
  const button = event?.currentTarget;
  importBusy = true;
  if (button) button.disabled = true;
  overlayStatus("Імпортую…");
  try {
    const form = new FormData();
    form.append("file", importFile, importFile.name);
    form.append("edits", JSON.stringify(importEdits));
    form.append("resolutions", JSON.stringify(importResolutions));
    const result = await api("/api/admin/test-exam-questions/import/apply-edited", {
      method: "POST",
      body: form,
      timeoutMs: 30000,
    });
    const body = document.querySelector("#test-q-manager-body");
    if (body) {
      body.innerHTML = `
        <section class="test-q-manager-card">
          <div style="font-size:22px; font-weight:800;">Готово</div>
          <div>Додано: <b>${result.inserted}</b><br>Оновлено відповідей: <b>${result.updated}</b><br>Пропущено/залишено: <b>${result.kept}</b></div>
          <button class="btn btn--primary btn--block" id="tqi-done" type="button">Повернутися до питань</button>
        </section>
      `;
      body.querySelector("#tqi-done")?.addEventListener("click", () => {
        closeOverlay();
        refreshList();
      });
    }
  } catch (error) {
    overlayStatus(error.message || "Не вдалося виконати імпорт.", "error");
    importBusy = false;
    updateImportApply();
  }
}

function openImport() {
  importFile = null;
  importPreview = null;
  importEdits = {};
  importResolutions = {};
  importConfirmed = false;
  importBusy = false;
  openOverlay("Імпорт JSON", "Перевірка перед додаванням");
  renderImportHome();
}

function injectToolbar() {
  const list = document.querySelector("#test-q-list");
  if (!list) return;
  ensureStyles();
  bindCards();
  const content = list.closest(".screen-content");
  const search = content?.querySelector(".case-search");
  if (!content || !search) return;

  document.querySelector("#admin-test-json-import")?.remove();
  document.querySelector("#test-q-crud-actions")?.remove();

  if (!document.querySelector("#test-q-manager-actions")) {
    const box = document.createElement("div");
    box.id = "test-q-manager-actions";
    box.className = "test-q-manager-actions";
    box.innerHTML = `
      <button class="btn btn--primary" id="test-q-manager-add" type="button">＋ Додати</button>
      <button class="btn" id="test-q-manager-import" type="button">Імпорт JSON</button>
      <div class="test-q-manager-status" id="test-q-manager-inline-status" style="grid-column:1/-1"></div>
    `;
    search.insertAdjacentElement("afterend", box);
    box.querySelector("#test-q-manager-add")?.addEventListener("click", () => renderQuestionEditor(null));
    box.querySelector("#test-q-manager-import")?.addEventListener("click", openImport);
  }
}

const observer = new MutationObserver(() => {
  injectToolbar();
  bindCards();
});
observer.observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("DOMContentLoaded", injectToolbar);
injectToolbar();
