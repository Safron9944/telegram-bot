import { api } from "./core/api.js?v=20260617-question-search-04";

let activeFile = null;
let activePreview = null;
let busy = false;

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
  if (document.querySelector("#admin-test-import-styles")) return;
  const style = document.createElement("style");
  style.id = "admin-test-import-styles";
  style.textContent = `
    .test-import-panel { margin-bottom: 16px; }
    .test-import-box { padding: 14px; display: grid; gap: 12px; }
    .test-import-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .test-import-stat { padding: 10px 8px; border-radius: 12px; background: var(--bg-fill-soft); text-align: center; }
    .test-import-stat strong { display: block; font-size: 20px; line-height: 1.15; }
    .test-import-stat span { display: block; margin-top: 3px; font-size: 12px; opacity: .72; }
    .test-import-file { font-size: 13px; opacity: .76; overflow-wrap: anywhere; }
    .test-import-conflicts { display: grid; gap: 10px; }
    .test-import-conflict { border: 1px solid var(--separator, rgba(128,128,128,.22)); border-radius: 14px; padding: 12px; display: grid; gap: 10px; }
    .test-import-conflict__question { font-weight: 650; line-height: 1.35; }
    .test-import-answer { padding: 9px 10px; border-radius: 10px; background: var(--bg-fill-soft); font-size: 13px; }
    .test-import-answer b { display: block; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; opacity: .65; }
    .test-import-options { display: grid; gap: 7px; }
    .test-import-option { display: flex; gap: 8px; align-items: flex-start; font-size: 13px; line-height: 1.35; cursor: pointer; }
    .test-import-warning { padding: 10px 12px; border-radius: 10px; background: var(--bg-fill-soft); font-size: 13px; line-height: 1.4; }
    .test-import-status { font-size: 13px; line-height: 1.4; min-height: 18px; }
    .test-import-status--error { color: var(--danger, #d33); }
    .test-import-status--success { color: var(--success, #16843d); }
    @media (max-width: 420px) { .test-import-summary { grid-template-columns: 1fr; } }
  `;
  document.head.append(style);
}

function getPanel() {
  return document.querySelector("#admin-test-json-import");
}

function setStatus(message, tone = "") {
  const status = document.querySelector("#test-import-status");
  if (!status) return;
  status.className = `test-import-status${tone ? ` test-import-status--${tone}` : ""}`;
  status.textContent = message || "";
}

function selectedResolutions() {
  const result = {};
  document.querySelectorAll("#test-import-conflicts input[type=radio]:checked").forEach((input) => {
    result[input.dataset.importIndex] = input.value;
  });
  return result;
}

function blockingFileConflictCount() {
  return (activePreview?.conflicts || []).filter((item) => item.kind === "file_answer_conflict").length;
}

function updateApplyButton() {
  const button = document.querySelector("#test-import-apply");
  if (!button || !activePreview) return;
  const conflicts = activePreview.conflicts || [];
  const blocking = blockingFileConflictCount();
  const decisions = selectedResolutions();
  const resolvable = conflicts.filter((item) => item.kind !== "file_answer_conflict");
  const allResolved = resolvable.every((item) => decisions[String(item.imported.import_index)]);
  button.disabled = busy || blocking > 0 || !allResolved || (!activePreview.new_count && !conflicts.length);

  const changes = Number(activePreview.new_count || 0) + resolvable.filter((item) => {
    const decision = decisions[String(item.imported.import_index)];
    return decision === "use_imported" || decision === "add_new";
  }).length;
  button.textContent = changes ? `Застосувати імпорт (${changes} змін)` : "Застосувати імпорт";
}

function conflictHtml(conflict) {
  const imported = conflict.imported || {};
  const existing = conflict.existing || {};
  const index = String(imported.import_index);
  const name = `test-import-conflict-${index}`;

  if (conflict.kind === "file_answer_conflict") {
    return `
      <article class="test-import-conflict">
        <div class="test-import-conflict__question">${escapeHtml(imported.question)}</div>
        <div class="test-import-warning">У самому JSON це питання повторюється з різними правильними відповідями. Виправ файл і завантаж його ще раз — такий конфлікт автоматично не застосовується.</div>
      </article>
    `;
  }

  const isSimilar = conflict.match_type === "similar";
  const similarity = isSimilar ? ` · схожість ${Math.round(Number(conflict.similarity || 0) * 100)}%` : "";
  const addNewOption = isSimilar
    ? `<label class="test-import-option"><input type="radio" name="${name}" data-import-index="${index}" value="add_new" /> <span>Це інше питання — додати його як нове</span></label>`
    : "";

  return `
    <article class="test-import-conflict">
      <div class="test-import-conflict__question">${escapeHtml(imported.question)}</div>
      <div class="test-import-file">${isSimilar ? "Знайдено дуже схоже питання" : "Знайдено те саме питання"}${similarity}</div>
      ${isSimilar && existing.question !== imported.question ? `<div class="test-import-answer"><b>Питання в базі</b>${escapeHtml(existing.question)}</div>` : ""}
      <div class="test-import-answer"><b>Зараз у базі</b>${escapeHtml(existing.correct_answer || "—")}</div>
      <div class="test-import-answer"><b>У файлі</b>${escapeHtml(imported.correct_answer || "—")}</div>
      <div class="test-import-options">
        <label class="test-import-option"><input type="radio" name="${name}" data-import-index="${index}" value="keep_existing" /> <span>Залишити відповідь, яка зараз у базі</span></label>
        <label class="test-import-option"><input type="radio" name="${name}" data-import-index="${index}" value="use_imported" /> <span>Замінити правильною відповіддю з файлу</span></label>
        ${addNewOption}
      </div>
    </article>
  `;
}

function renderPreview(preview) {
  activePreview = preview;
  const result = document.querySelector("#test-import-result");
  if (!result) return;

  const conflicts = preview.conflicts || [];
  const invalidCount = Number(preview.invalid_count || 0);
  result.innerHTML = `
    <div class="test-import-file"><b>${escapeHtml(preview.file_name)}</b> · ${formatBytes(preview.file_size)} · ${preview.valid_count} придатних питань</div>
    <div class="test-import-summary">
      <div class="test-import-stat"><strong>${preview.new_count}</strong><span>нових</span></div>
      <div class="test-import-stat"><strong>${preview.duplicate_count}</strong><span>дублікатів</span></div>
      <div class="test-import-stat"><strong>${preview.conflict_count}</strong><span>потребують перевірки</span></div>
    </div>
    ${invalidCount ? `<div class="test-import-warning">Пропущено некоректних елементів: ${invalidCount}. Вони не будуть імпортовані.</div>` : ""}
    ${conflicts.length ? `
      <div class="group__label">Перевір конфлікти вручну</div>
      <div class="test-import-conflicts" id="test-import-conflicts">${conflicts.map(conflictHtml).join("")}</div>
    ` : '<div class="test-import-warning">Конфліктів відповідей немає. Дублікати будуть пропущені автоматично.</div>'}
    <button class="btn btn--primary btn--block" id="test-import-apply" type="button">Застосувати імпорт</button>
  `;

  document.querySelectorAll("#test-import-conflicts input[type=radio]").forEach((input) => {
    input.addEventListener("change", updateApplyButton);
  });
  document.querySelector("#test-import-apply")?.addEventListener("click", applyImport);
  updateApplyButton();
}

async function previewFile(file) {
  if (!file || busy) return;
  activeFile = file;
  activePreview = null;
  busy = true;
  setStatus("Перевіряю файл, дублікати та відповіді…");
  const result = document.querySelector("#test-import-result");
  if (result) result.innerHTML = "";
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const preview = await api("/api/admin/test-exam-questions/import/preview", {
      method: "POST",
      body: form,
      timeoutMs: 30000,
    });
    renderPreview(preview);
    setStatus("Перевірку завершено.", "success");
  } catch (error) {
    activeFile = null;
    setStatus(error.message || "Не вдалося перевірити файл.", "error");
  } finally {
    busy = false;
    updateApplyButton();
  }
}

async function applyImport() {
  if (!activeFile || !activePreview || busy) return;
  const blocking = blockingFileConflictCount();
  if (blocking) {
    setStatus("Спочатку виправ конфлікти відповідей усередині самого JSON-файлу.", "error");
    return;
  }

  const decisions = selectedResolutions();
  const unresolved = (activePreview.conflicts || []).filter(
    (item) => !decisions[String(item.imported.import_index)],
  );
  if (unresolved.length) {
    setStatus(`Потрібно вибрати рішення для всіх конфліктів (${unresolved.length}).`, "error");
    return;
  }

  if (!window.confirm("Застосувати перевірений імпорт до розділу «Тестові питання»?")) return;

  busy = true;
  updateApplyButton();
  setStatus("Застосовую зміни…");
  try {
    const form = new FormData();
    form.append("file", activeFile, activeFile.name);
    form.append("resolutions", JSON.stringify(decisions));
    const payload = await api("/api/admin/test-exam-questions/import/apply", {
      method: "POST",
      body: form,
      timeoutMs: 30000,
    });
    setStatus(
      `Готово: додано ${payload.inserted}, оновлено відповідей ${payload.updated}, залишено/пропущено ${payload.kept}.`,
      "success",
    );
    const input = document.querySelector("#test-q-input");
    input?.dispatchEvent(new Event("input", { bubbles: true }));
    activePreview = null;
    activeFile = null;
    const result = document.querySelector("#test-import-result");
    if (result) result.innerHTML = "";
    const fileInput = document.querySelector("#test-import-file");
    if (fileInput) fileInput.value = "";
  } catch (error) {
    setStatus(error.message || "Не вдалося застосувати імпорт.", "error");
  } finally {
    busy = false;
    updateApplyButton();
  }
}

function injectPanel() {
  const list = document.querySelector("#test-q-list");
  if (!list || getPanel()) return;
  const content = list.closest(".screen-content");
  const questionsSection = list.closest(".case-questions");
  if (!content || !questionsSection) return;

  ensureStyles();
  const panel = document.createElement("section");
  panel.id = "admin-test-json-import";
  panel.className = "group test-import-panel";
  panel.innerHTML = `
    <div class="group__label">Імпорт JSON</div>
    <div class="group__list">
      <div class="test-import-box">
        <div>
          <div style="font-weight:650;">Додати питання з файлу</div>
          <div class="test-import-file" style="margin-top:4px;">Дублікати пропускаються. Якщо правильні відповіді відрізняються, імпорт зупиниться на конфлікті й дасть вибрати правильний варіант вручну.</div>
        </div>
        <input id="test-import-file" type="file" accept="application/json,.json" hidden />
        <button class="btn btn--primary btn--block" id="test-import-pick" type="button">Вибрати JSON-файл</button>
        <div class="test-import-status" id="test-import-status" aria-live="polite"></div>
        <div id="test-import-result"></div>
      </div>
    </div>
  `;
  content.insertBefore(panel, questionsSection);

  const fileInput = panel.querySelector("#test-import-file");
  panel.querySelector("#test-import-pick")?.addEventListener("click", () => fileInput?.click());
  fileInput?.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) void previewFile(file);
  });
}

const observer = new MutationObserver(() => injectPanel());
observer.observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("DOMContentLoaded", injectPanel);
injectPanel();
