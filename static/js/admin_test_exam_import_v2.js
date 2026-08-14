import { api } from "./core/api.js?v=20260617-question-search-04";

let activeFile = null;
let activePreview = null;
let busy = false;
let edits = {};
let reviewed = new Set();
let resolutions = {};
let editorIndex = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  const size = Number(bytes || 0);
  if (size < 1024) return `${size} Б`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} КБ`;
  return `${(size / 1024 / 1024).toFixed(1)} МБ`;
}

function ensureStyles() {
  if (document.querySelector("#admin-test-import-v2-styles")) return;
  const style = document.createElement("style");
  style.id = "admin-test-import-v2-styles";
  style.textContent = `
    .test-import-panel { margin-bottom: 16px; }
    .test-import-box { padding: 14px; display: grid; gap: 12px; }
    .test-import-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .test-import-stat { padding: 10px 8px; border-radius: 12px; background: var(--bg-fill-soft); text-align: center; }
    .test-import-stat strong { display: block; font-size: 20px; line-height: 1.15; }
    .test-import-stat span { display: block; margin-top: 3px; font-size: 12px; opacity: .72; }
    .test-import-file { font-size: 13px; opacity: .76; overflow-wrap: anywhere; }
    .test-import-list { display: grid; gap: 10px; }
    .test-import-card { border: 1px solid var(--separator, rgba(128,128,128,.22)); border-radius: 14px; padding: 12px; display: grid; gap: 10px; background: var(--bg-elevated, #fff); }
    .test-import-card__meta { font-size: 12px; opacity: .68; }
    .test-import-card__question { font-weight: 700; line-height: 1.42; }
    .test-import-answer { padding: 9px 10px; border-radius: 10px; background: var(--bg-fill-soft); font-size: 13px; line-height: 1.4; }
    .test-import-answer b { display: block; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; opacity: .65; }
    .test-import-support { font-size: 12px; line-height: 1.4; opacity: .78; overflow-wrap: anywhere; }
    .test-import-options { display: grid; gap: 7px; }
    .test-import-option { display: flex; gap: 8px; align-items: flex-start; font-size: 13px; line-height: 1.35; cursor: pointer; }
    .test-import-check { display: flex; gap: 9px; align-items: flex-start; font-size: 13px; line-height: 1.35; cursor: pointer; padding: 4px 0 0; }
    .test-import-warning { padding: 10px 12px; border-radius: 10px; background: var(--bg-fill-soft); font-size: 13px; line-height: 1.4; }
    .test-import-status { font-size: 13px; line-height: 1.4; min-height: 18px; }
    .test-import-status--error { color: var(--danger, #d33); }
    .test-import-status--success { color: var(--success, #16843d); }
    .test-import-section-head { display: flex; justify-content: space-between; gap: 10px; align-items: end; margin-top: 4px; }
    .test-import-section-head strong { font-size: 14px; }
    .test-import-section-head span { font-size: 12px; opacity: .7; }
    .test-import-edit-btn { width: 100%; }

    .test-import-editor { position: fixed; inset: 0; z-index: 20000; background: var(--bg, #f4f5f7); overflow-y: auto; overscroll-behavior: contain; }
    .test-import-editor[hidden] { display: none !important; }
    .test-import-editor__bar { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: calc(10px + env(safe-area-inset-top)) 16px 10px; background: var(--bg, #f4f5f7); border-bottom: 1px solid var(--separator, rgba(128,128,128,.2)); }
    .test-import-editor__bar strong { font-size: 18px; }
    .test-import-editor__back { border: 0; background: transparent; font: inherit; font-weight: 650; padding: 8px 4px; cursor: pointer; }
    .test-import-editor__content { max-width: 760px; margin: 0 auto; padding: 18px 16px calc(28px + env(safe-area-inset-bottom)); display: grid; gap: 14px; }
    .test-import-editor__hint { font-size: 13px; line-height: 1.45; opacity: .76; }
    .test-import-editor__field { display: grid; gap: 6px; }
    .test-import-editor__field span { font-size: 12px; font-weight: 700; opacity: .72; }
    .test-import-editor__input, .test-import-editor__textarea { width: 100%; box-sizing: border-box; border: 1px solid var(--separator, rgba(128,128,128,.28)); border-radius: 12px; background: var(--bg-elevated, #fff); color: inherit; font: inherit; padding: 12px; outline: none; }
    .test-import-editor__textarea { min-height: 112px; resize: vertical; line-height: 1.45; }
    .test-import-editor__textarea--answer { min-height: 90px; }
    .test-import-editor__actions { display: grid; gap: 8px; margin-top: 4px; }
    .test-import-editor__danger { color: var(--danger, #c33); }
    body.test-import-editor-open { overflow: hidden; }

    @media (max-width: 420px) {
      .test-import-summary { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .test-import-stat strong { font-size: 18px; }
      .test-import-stat { padding: 9px 5px; }
    }
  `;
  document.head.append(style);
}

function setStatus(message, tone = "") {
  const status = document.querySelector("#test-import-status");
  if (!status) return;
  status.className = `test-import-status${tone ? ` test-import-status--${tone}` : ""}`;
  status.textContent = message || "";
}

function effectiveItem(importIndex) {
  const index = Number(importIndex);
  const groups = [
    ...(activePreview?.new || []),
    ...(activePreview?.conflicts || []).map((item) => item.imported),
    ...(activePreview?.duplicates || []).map((item) => item.imported),
  ];
  return groups.find((item) => Number(item?.import_index) === index) || null;
}

function currentNewItems() {
  return Array.isArray(activePreview?.new) ? activePreview.new : [];
}

function blockingFileConflictCount() {
  return (activePreview?.conflicts || []).filter((item) => item.kind === "file_answer_conflict").length;
}

function allNewReviewed() {
  return currentNewItems().every((item) => reviewed.has(String(item.import_index)));
}

function unresolvedConflictCount() {
  return (activePreview?.conflicts || []).filter((item) => {
    if (item.kind === "file_answer_conflict") return true;
    return !resolutions[String(item.imported.import_index)];
  }).length;
}

function updateApplyButton() {
  const button = document.querySelector("#test-import-apply");
  if (!button || !activePreview) return;
  const newCount = Number(activePreview.new_count || 0);
  const conflicts = activePreview.conflicts || [];
  const blocking = blockingFileConflictCount();
  const unresolved = unresolvedConflictCount();
  const hasAction = newCount > 0 || conflicts.some((item) => {
    const decision = resolutions[String(item.imported.import_index)];
    return decision === "use_imported" || decision === "add_new";
  });
  button.disabled = busy || blocking > 0 || unresolved > 0 || !allNewReviewed() || !hasAction;

  const editsCount = Object.keys(edits).length;
  const parts = [];
  if (newCount) parts.push(`${newCount} нових`);
  if (editsCount) parts.push(`${editsCount} виправл.`);
  button.textContent = parts.length ? `Застосувати імпорт · ${parts.join(" · ")}` : "Застосувати імпорт";
}

function itemMeta(item) {
  return [item?.num, item?.module, item?.source].filter(Boolean).join(" · ");
}

function openEditor(importIndex) {
  const item = effectiveItem(importIndex);
  if (!item) {
    setStatus("Не вдалося відкрити це питання для редагування.", "error");
    return;
  }
  editorIndex = Number(importIndex);
  let editor = document.querySelector("#test-import-editor");
  if (!editor) {
    editor = document.createElement("section");
    editor.id = "test-import-editor";
    editor.className = "test-import-editor";
    document.body.append(editor);
  }
  const hasEdit = Object.prototype.hasOwnProperty.call(edits, String(editorIndex));
  editor.innerHTML = `
    <div class="test-import-editor__bar">
      <button class="test-import-editor__back" id="test-import-editor-back" type="button">← Назад</button>
      <strong>Редагування питання</strong>
      <span style="width:58px"></span>
    </div>
    <div class="test-import-editor__content">
      <div class="test-import-editor__hint">Зміни тут будуть використані під час імпорту. Початковий JSON-файл на телефоні не змінюється.</div>
      <label class="test-import-editor__field"><span>Номер</span><input class="test-import-editor__input" id="test-editor-num" value="${escapeHtml(item.num || "")}" placeholder="Наприклад: № 61" /></label>
      <label class="test-import-editor__field"><span>Модуль</span><input class="test-import-editor__input" id="test-editor-module" value="${escapeHtml(item.module || "")}" placeholder="Необов’язково" /></label>
      <label class="test-import-editor__field"><span>Питання</span><textarea class="test-import-editor__textarea" id="test-editor-question">${escapeHtml(item.question || "")}</textarea></label>
      <label class="test-import-editor__field"><span>Правильна відповідь</span><textarea class="test-import-editor__textarea test-import-editor__textarea--answer" id="test-editor-answer">${escapeHtml(item.correct_answer || "")}</textarea></label>
      <label class="test-import-editor__field"><span>Джерело</span><input class="test-import-editor__input" id="test-editor-source" value="${escapeHtml(item.source || "")}" placeholder="Наприклад: МКУ, ст. 123" /></label>
      <label class="test-import-editor__field"><span>Обґрунтування</span><textarea class="test-import-editor__textarea" id="test-editor-justification" placeholder="Необов’язково">${escapeHtml(item.justification || "")}</textarea></label>
      <div class="test-import-editor__actions">
        <button class="btn btn--primary btn--block" id="test-import-editor-save" type="button">Зберегти зміни</button>
        ${hasEdit ? '<button class="btn btn--block test-import-editor__danger" id="test-import-editor-reset" type="button">Скасувати мої зміни цього питання</button>' : ""}
      </div>
    </div>
  `;
  editor.hidden = false;
  document.body.classList.add("test-import-editor-open");
  editor.querySelector("#test-import-editor-back")?.addEventListener("click", closeEditor);
  editor.querySelector("#test-import-editor-save")?.addEventListener("click", saveEditor);
  editor.querySelector("#test-import-editor-reset")?.addEventListener("click", resetEditorChanges);
}

function closeEditor() {
  const editor = document.querySelector("#test-import-editor");
  if (editor) editor.hidden = true;
  document.body.classList.remove("test-import-editor-open");
  editorIndex = null;
}

async function saveEditor() {
  if (editorIndex == null || busy) return;
  const question = document.querySelector("#test-editor-question")?.value.trim() || "";
  const correctAnswer = document.querySelector("#test-editor-answer")?.value.trim() || "";
  if (!question) { window.alert("Введи текст питання."); return; }
  if (!correctAnswer) { window.alert("Введи правильну відповідь."); return; }

  const index = String(editorIndex);
  edits[index] = {
    num: document.querySelector("#test-editor-num")?.value.trim() || "",
    module: document.querySelector("#test-editor-module")?.value.trim() || "",
    question,
    correct_answer: correctAnswer,
    source: document.querySelector("#test-editor-source")?.value.trim() || "",
    justification: document.querySelector("#test-editor-justification")?.value.trim() || "",
  };
  reviewed.delete(index);
  delete resolutions[index];
  closeEditor();
  setStatus("Зміни збережено. Повторно перевіряю дублікати…");
  await refreshPreview();
}

async function resetEditorChanges() {
  if (editorIndex == null || busy) return;
  const index = String(editorIndex);
  delete edits[index];
  reviewed.delete(index);
  delete resolutions[index];
  closeEditor();
  setStatus("Виправлення скасовано. Повторно перевіряю файл…");
  await refreshPreview();
}

function newItemHtml(item) {
  const index = String(item.import_index);
  const meta = itemMeta(item);
  const edited = Object.prototype.hasOwnProperty.call(edits, index);
  return `
    <article class="test-import-card">
      <div class="test-import-card__meta">${escapeHtml(meta || "Нове питання")}${edited ? " · ✎ змінено вручну" : ""}</div>
      <div class="test-import-card__question">${escapeHtml(item.question)}</div>
      <div class="test-import-answer"><b>Правильна відповідь</b>${escapeHtml(item.correct_answer || "—")}</div>
      ${item.justification ? `<div class="test-import-support"><b>Обґрунтування:</b> ${escapeHtml(item.justification)}</div>` : ""}
      <button class="btn btn--block test-import-edit-btn" type="button" data-edit-index="${index}">Відкрити / редагувати</button>
      <label class="test-import-check"><input type="checkbox" data-review-index="${index}" ${reviewed.has(index) ? "checked" : ""} /><span>Перевірено: питання і правильна відповідь вірні</span></label>
    </article>
  `;
}

function conflictHtml(conflict) {
  const imported = conflict.imported || {};
  const existing = conflict.existing || {};
  const index = String(imported.import_index);
  const isFileConflict = conflict.kind === "file_answer_conflict";
  const isSimilar = conflict.match_type === "similar";
  const similarity = isSimilar ? ` · схожість ${Math.round(Number(conflict.similarity || 0) * 100)}%` : "";

  if (isFileConflict) {
    return `
      <article class="test-import-card">
        <div class="test-import-card__question">${escapeHtml(imported.question)}</div>
        <div class="test-import-warning">У самому JSON це питання повторюється з різними відповідями. Відкрий цей запис, виправ питання або відповідь і збережи — після цього перевірка запуститься повторно.</div>
        <button class="btn btn--block" type="button" data-edit-index="${index}">Відкрити / виправити</button>
      </article>
    `;
  }

  const selected = resolutions[index] || "";
  return `
    <article class="test-import-card">
      <div class="test-import-card__question">${escapeHtml(imported.question)}</div>
      <div class="test-import-file">${isSimilar ? "Знайдено дуже схоже питання" : "Знайдено те саме питання"}${similarity}</div>
      ${isSimilar && existing.question !== imported.question ? `<div class="test-import-answer"><b>Питання в базі</b>${escapeHtml(existing.question || "—")}</div>` : ""}
      <div class="test-import-answer"><b>Зараз у базі</b>${escapeHtml(existing.correct_answer || "—")}</div>
      <div class="test-import-answer"><b>У файлі</b>${escapeHtml(imported.correct_answer || "—")}</div>
      <button class="btn btn--block" type="button" data-edit-index="${index}">Редагувати варіант із файлу</button>
      <div class="test-import-options">
        <label class="test-import-option"><input type="radio" name="conflict-${index}" data-resolution-index="${index}" value="keep_existing" ${selected === "keep_existing" ? "checked" : ""} /> <span>Залишити відповідь, яка зараз у базі</span></label>
        <label class="test-import-option"><input type="radio" name="conflict-${index}" data-resolution-index="${index}" value="use_imported" ${selected === "use_imported" ? "checked" : ""} /> <span>Використати відредагований варіант із файлу</span></label>
        ${isSimilar ? `<label class="test-import-option"><input type="radio" name="conflict-${index}" data-resolution-index="${index}" value="add_new" ${selected === "add_new" ? "checked" : ""} /> <span>Це інше питання — додати окремо</span></label>` : ""}
      </div>
    </article>
  `;
}

function bindPreviewEvents() {
  document.querySelectorAll("[data-edit-index]").forEach((button) => button.addEventListener("click", () => openEditor(button.dataset.editIndex)));
  document.querySelectorAll("[data-review-index]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const index = String(checkbox.dataset.reviewIndex);
      if (checkbox.checked) reviewed.add(index); else reviewed.delete(index);
      updateApplyButton();
      const counter = document.querySelector("#test-import-reviewed-count");
      if (counter) counter.textContent = `${currentNewItems().filter((item) => reviewed.has(String(item.import_index))).length}/${currentNewItems().length}`;
    });
  });
  document.querySelectorAll("[data-resolution-index]").forEach((radio) => {
    radio.addEventListener("change", () => {
      resolutions[String(radio.dataset.resolutionIndex)] = radio.value;
      updateApplyButton();
    });
  });
  document.querySelector("#test-import-apply")?.addEventListener("click", applyImport);
}

function renderPreview(preview) {
  activePreview = preview;
  const result = document.querySelector("#test-import-result");
  if (!result) return;

  const newItems = currentNewItems();
  const conflicts = preview.conflicts || [];
  const invalidCount = Number(preview.invalid_count || 0);
  const reviewedCount = newItems.filter((item) => reviewed.has(String(item.import_index))).length;

  result.innerHTML = `
    <div class="test-import-file"><b>${escapeHtml(preview.file_name)}</b> · ${formatBytes(preview.file_size)} · ${preview.valid_count} придатних питань</div>
    <div class="test-import-summary">
      <div class="test-import-stat"><strong>${preview.new_count}</strong><span>нових</span></div>
      <div class="test-import-stat"><strong>${preview.duplicate_count}</strong><span>дублікатів</span></div>
      <div class="test-import-stat"><strong>${preview.conflict_count}</strong><span>потребують перевірки</span></div>
    </div>
    ${invalidCount ? `<div class="test-import-warning">Пропущено некоректних елементів: ${invalidCount}. Вони не будуть імпортовані.</div>` : ""}
    ${newItems.length ? `
      <div class="test-import-section-head"><strong>Нові питання — відкрий і перевір</strong><span id="test-import-reviewed-count">${reviewedCount}/${newItems.length}</span></div>
      <div class="test-import-warning">Натисни «Відкрити / редагувати», щоб побачити питання в окремому повноекранному редакторі та за потреби виправити його. Після перевірки постав галочку.</div>
      <div class="test-import-list">${newItems.map(newItemHtml).join("")}</div>
    ` : ""}
    ${conflicts.length ? `
      <div class="test-import-section-head"><strong>Конфлікти</strong><span>${conflicts.length}</span></div>
      <div class="test-import-list">${conflicts.map(conflictHtml).join("")}</div>
    ` : '<div class="test-import-warning">Конфліктів відповідей немає. Дублікати будуть пропущені автоматично.</div>'}
    <button class="btn btn--primary btn--block" id="test-import-apply" type="button">Застосувати імпорт</button>
  `;
  bindPreviewEvents();
  updateApplyButton();
}

async function requestPreview() {
  if (!activeFile) return null;
  const form = new FormData();
  form.append("file", activeFile, activeFile.name);
  form.append("edits", JSON.stringify(edits));
  return api("/api/admin/test-exam-questions/import/preview-edited", { method: "POST", body: form, timeoutMs: 30000 });
}

async function refreshPreview() {
  if (!activeFile || busy) return;
  busy = true;
  updateApplyButton();
  try {
    const preview = await requestPreview();
    if (!preview) return;
    const validIndexes = new Set([
      ...(preview.new || []).map((item) => String(item.import_index)),
      ...(preview.conflicts || []).map((item) => String(item.imported?.import_index)),
      ...(preview.duplicates || []).map((item) => String(item.imported?.import_index)),
    ]);
    reviewed = new Set([...reviewed].filter((index) => validIndexes.has(index)));
    Object.keys(resolutions).forEach((index) => {
      const stillConflict = (preview.conflicts || []).some((item) => String(item.imported?.import_index) === index && item.kind !== "file_answer_conflict");
      if (!stillConflict) delete resolutions[index];
    });
    renderPreview(preview);
    setStatus("Перевірку завершено.", "success");
  } catch (error) {
    setStatus(error.message || "Не вдалося перевірити файл.", "error");
  } finally {
    busy = false;
    updateApplyButton();
  }
}

async function previewFile(file) {
  if (!file || busy) return;
  activeFile = file;
  activePreview = null;
  edits = {};
  reviewed = new Set();
  resolutions = {};
  setStatus("Перевіряю файл, дублікати та відповіді…");
  const result = document.querySelector("#test-import-result");
  if (result) result.innerHTML = "";
  await refreshPreview();
}

async function applyImport() {
  if (!activeFile || !activePreview || busy) return;
  if (!allNewReviewed()) { setStatus("Спочатку відкрий, перевір і підтвердь усі нові питання.", "error"); return; }
  if (blockingFileConflictCount()) { setStatus("Спочатку виправ конфлікти всередині самого JSON.", "error"); return; }
  if (unresolvedConflictCount()) { setStatus("Потрібно вибрати рішення для всіх конфліктів.", "error"); return; }

  const newCount = Number(activePreview.new_count || 0);
  const editedCount = Object.keys(edits).length;
  if (!window.confirm(`Імпортувати перевірені питання? Нових: ${newCount}. Виправлено вручну: ${editedCount}.`)) return;

  busy = true;
  updateApplyButton();
  setStatus("Застосовую зміни…");
  try {
    const form = new FormData();
    form.append("file", activeFile, activeFile.name);
    form.append("edits", JSON.stringify(edits));
    form.append("resolutions", JSON.stringify(resolutions));
    const payload = await api("/api/admin/test-exam-questions/import/apply-edited", { method: "POST", body: form, timeoutMs: 30000 });
    setStatus(`Готово: додано ${payload.inserted}, оновлено ${payload.updated}, пропущено/залишено ${payload.kept}, ручних виправлень ${payload.edited}.`, "success");
    activeFile = null;
    activePreview = null;
    edits = {};
    reviewed = new Set();
    resolutions = {};
    const result = document.querySelector("#test-import-result");
    if (result) result.innerHTML = "";
    const fileInput = document.querySelector("#test-import-file");
    if (fileInput) fileInput.value = "";
    const searchInput = document.querySelector("#test-q-input");
    searchInput?.dispatchEvent(new Event("input", { bubbles: true }));
  } catch (error) {
    setStatus(error.message || "Не вдалося застосувати імпорт.", "error");
  } finally {
    busy = false;
    updateApplyButton();
  }
}

function injectPanel() {
  const list = document.querySelector("#test-q-list");
  if (!list || document.querySelector("#admin-test-json-import")) return;
  const content = list.closest(".screen-content");
  const questionsSection = list.closest(".case-questions");
  if (!content || !questionsSection) return;

  ensureStyles();
  const panel = document.createElement("section");
  panel.id = "admin-test-json-import";
  panel.className = "group test-import-panel";
  panel.innerHTML = `
    <div class="group__label">Імпорт JSON</div>
    <div class="group__list"><div class="test-import-box">
      <div><div style="font-weight:700;">Додати питання з файлу</div><div class="test-import-file" style="margin-top:4px;">Дублікати пропускаються. Нові питання відкриваються окремо для перегляду та редагування перед імпортом.</div></div>
      <input id="test-import-file" type="file" accept="application/json,.json" hidden />
      <button class="btn btn--primary btn--block" id="test-import-pick" type="button">Вибрати JSON-файл</button>
      <div class="test-import-status" id="test-import-status" aria-live="polite"></div>
      <div id="test-import-result"></div>
    </div></div>
  `;
  content.insertBefore(panel, questionsSection);
  const fileInput = panel.querySelector("#test-import-file");
  panel.querySelector("#test-import-pick")?.addEventListener("click", () => fileInput?.click());
  fileInput?.addEventListener("change", () => { const file = fileInput.files?.[0]; if (file) void previewFile(file); });
}

const observer = new MutationObserver(injectPanel);
observer.observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("DOMContentLoaded", injectPanel);
injectPanel();
