import { api } from "./core/api.js?v=20260617-question-search-04";

let editorOpen = false;
let savedBodyOverflow = "";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function ensureStyles() {
  if (document.querySelector("#test-question-crud-styles")) return;
  const style = document.createElement("style");
  style.id = "test-question-crud-styles";
  style.textContent = `
    .test-q-crud-actions { display: grid; gap: 8px; margin: 12px 0 16px; }
    .test-q-crud-card { cursor: pointer; position: relative; }
    .test-q-crud-card:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
    .test-q-crud-hint { margin-top: 10px; font-size: 12px; font-weight: 650; opacity: .58; }
    .test-q-crud-status { min-height: 18px; font-size: 13px; line-height: 1.35; }
    .test-q-crud-status--success { color: var(--success, #16843d); }
    .test-q-crud-status--error { color: var(--danger, #d33); }

    .test-q-editor-overlay {
      position: fixed; inset: 0; z-index: 10050;
      background: var(--bg, #f5f6f8);
      overflow-y: auto; overscroll-behavior: contain;
    }
    .test-q-editor {
      width: min(760px, 100%); min-height: 100%;
      margin: 0 auto; padding: max(16px, env(safe-area-inset-top)) 16px max(28px, env(safe-area-inset-bottom));
      box-sizing: border-box;
    }
    .test-q-editor__top {
      position: sticky; top: 0; z-index: 2;
      display: flex; align-items: center; gap: 12px;
      padding: 8px 0 14px;
      background: var(--bg, #f5f6f8);
    }
    .test-q-editor__back {
      border: 0; background: var(--bg-fill-soft, rgba(128,128,128,.12));
      border-radius: 999px; padding: 10px 14px; font: inherit; font-weight: 650; cursor: pointer;
    }
    .test-q-editor__heading { min-width: 0; }
    .test-q-editor__title { font-size: 24px; font-weight: 800; line-height: 1.15; }
    .test-q-editor__subtitle { margin-top: 3px; font-size: 12px; opacity: .62; }
    .test-q-editor__form {
      display: grid; gap: 14px;
      background: var(--bg-card, #fff);
      border: 1px solid var(--separator, rgba(128,128,128,.18));
      border-radius: 20px; padding: 16px;
    }
    .test-q-editor__row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .test-q-editor__field { display: grid; gap: 6px; }
    .test-q-editor__label { font-size: 12px; font-weight: 700; opacity: .68; }
    .test-q-editor__input,
    .test-q-editor__textarea {
      width: 100%; box-sizing: border-box;
      border: 1px solid var(--separator, rgba(128,128,128,.25));
      border-radius: 12px; background: var(--bg, #f5f6f8); color: inherit;
      padding: 12px 13px; font: inherit; outline: none;
    }
    .test-q-editor__textarea { min-height: 116px; resize: vertical; line-height: 1.4; }
    .test-q-editor__input:focus,
    .test-q-editor__textarea:focus { border-color: currentColor; }
    .test-q-editor__buttons { display: grid; gap: 9px; margin-top: 4px; }
    .test-q-editor__status { min-height: 20px; font-size: 13px; line-height: 1.4; }
    .test-q-editor__danger { margin-top: 16px; }
    .test-q-editor__danger button {
      width: 100%; border: 1px solid var(--danger, #d33); color: var(--danger, #d33);
      background: transparent; border-radius: 14px; padding: 13px; font: inherit; font-weight: 700;
    }
    @media (max-width: 560px) {
      .test-q-editor__row { grid-template-columns: 1fr; }
      .test-q-editor { padding-left: 12px; padding-right: 12px; }
    }
  `;
  document.head.append(style);
}

function setManagerStatus(message, tone = "") {
  const el = document.querySelector("#test-q-crud-status");
  if (!el) return;
  el.className = `test-q-crud-status${tone ? ` test-q-crud-status--${tone}` : ""}`;
  el.textContent = message || "";
}

function setEditorStatus(message, tone = "") {
  const el = document.querySelector("#test-q-editor-status");
  if (!el) return;
  el.className = `test-q-editor__status${tone ? ` test-q-crud-status--${tone}` : ""}`;
  el.textContent = message || "";
}

function refreshQuestionList() {
  const input = document.querySelector("#test-q-input");
  if (input) input.dispatchEvent(new Event("input", { bubbles: true }));
}

function closeEditor() {
  document.querySelector("#test-q-editor-overlay")?.remove();
  document.body.style.overflow = savedBodyOverflow;
  editorOpen = false;
}

function fieldValue(id) {
  return document.querySelector(id)?.value ?? "";
}

function editorPayload() {
  return {
    num: fieldValue("#test-q-editor-num"),
    module: fieldValue("#test-q-editor-module"),
    question: fieldValue("#test-q-editor-question"),
    correct_answer: fieldValue("#test-q-editor-answer"),
    source: fieldValue("#test-q-editor-source"),
    justification: fieldValue("#test-q-editor-justification"),
  };
}

function openEditor(item = null) {
  if (editorOpen) return;
  ensureStyles();
  editorOpen = true;
  savedBodyOverflow = document.body.style.overflow;
  document.body.style.overflow = "hidden";

  const isEdit = Boolean(item?.id);
  const overlay = document.createElement("div");
  overlay.id = "test-q-editor-overlay";
  overlay.className = "test-q-editor-overlay";
  overlay.innerHTML = `
    <main class="test-q-editor">
      <div class="test-q-editor__top">
        <button class="test-q-editor__back" id="test-q-editor-back" type="button">← Назад</button>
        <div class="test-q-editor__heading">
          <div class="test-q-editor__title">${isEdit ? "Редагувати питання" : "Нове питання"}</div>
          <div class="test-q-editor__subtitle">${isEdit ? `ID ${item.id}` : "Додавання в «Тестові питання»"}</div>
        </div>
      </div>

      <section class="test-q-editor__form">
        <div class="test-q-editor__row">
          <label class="test-q-editor__field">
            <span class="test-q-editor__label">Номер</span>
            <input class="test-q-editor__input" id="test-q-editor-num" value="${escapeHtml(item?.num || "")}" placeholder="Напр. 61" />
          </label>
          <label class="test-q-editor__field">
            <span class="test-q-editor__label">Модуль</span>
            <input class="test-q-editor__input" id="test-q-editor-module" value="${escapeHtml(item?.module || "")}" placeholder="Необов’язково" />
          </label>
        </div>

        <label class="test-q-editor__field">
          <span class="test-q-editor__label">Питання *</span>
          <textarea class="test-q-editor__textarea" id="test-q-editor-question" placeholder="Введіть текст питання">${escapeHtml(item?.question || "")}</textarea>
        </label>

        <label class="test-q-editor__field">
          <span class="test-q-editor__label">Правильна відповідь *</span>
          <textarea class="test-q-editor__textarea" id="test-q-editor-answer" placeholder="Введіть правильну відповідь">${escapeHtml(item?.correct_answer || "")}</textarea>
        </label>

        <label class="test-q-editor__field">
          <span class="test-q-editor__label">Джерело</span>
          <input class="test-q-editor__input" id="test-q-editor-source" value="${escapeHtml(item?.source || "Вручну")}" placeholder="Напр. МКУ, ст. 123" />
        </label>

        <label class="test-q-editor__field">
          <span class="test-q-editor__label">Обґрунтування / примітка</span>
          <textarea class="test-q-editor__textarea" id="test-q-editor-justification" placeholder="Необов’язково">${escapeHtml(item?.justification || "")}</textarea>
        </label>

        <div class="test-q-editor__status" id="test-q-editor-status" aria-live="polite"></div>

        <div class="test-q-editor__buttons">
          <button class="btn btn--primary btn--block" id="test-q-editor-save" type="button">
            ${isEdit ? "Зберегти зміни" : "Додати питання"}
          </button>
        </div>
      </section>

      ${isEdit ? `
        <div class="test-q-editor__danger">
          <button id="test-q-editor-delete" type="button">Видалити питання</button>
        </div>
      ` : ""}
    </main>
  `;

  document.body.append(overlay);
  overlay.querySelector("#test-q-editor-back")?.addEventListener("click", closeEditor);

  const saveButton = overlay.querySelector("#test-q-editor-save");
  saveButton?.addEventListener("click", async () => {
    const payload = editorPayload();
    if (!String(payload.question).trim()) {
      setEditorStatus("Введіть текст питання.", "error");
      overlay.querySelector("#test-q-editor-question")?.focus();
      return;
    }
    if (!String(payload.correct_answer).trim()) {
      setEditorStatus("Введіть правильну відповідь.", "error");
      overlay.querySelector("#test-q-editor-answer")?.focus();
      return;
    }

    saveButton.disabled = true;
    setEditorStatus(isEdit ? "Зберігаю зміни…" : "Додаю питання…");
    try {
      const response = await api(
        isEdit ? `/api/admin/test-exam-questions/crud/${item.id}` : "/api/admin/test-exam-questions/crud",
        {
          method: isEdit ? "PATCH" : "POST",
          body: payload,
          timeoutMs: 20000,
        },
      );
      closeEditor();
      setManagerStatus(
        isEdit ? "Питання успішно оновлено." : `Питання додано (ID ${response.item.id}).`,
        "success",
      );
      refreshQuestionList();
    } catch (error) {
      setEditorStatus(error.message || "Не вдалося зберегти питання.", "error");
      saveButton.disabled = false;
    }
  });

  overlay.querySelector("#test-q-editor-delete")?.addEventListener("click", async (event) => {
    if (!window.confirm("Видалити це питання? Цю дію неможливо скасувати.")) return;
    const button = event.currentTarget;
    button.disabled = true;
    setEditorStatus("Видаляю питання…");
    try {
      await api(`/api/admin/test-exam-questions/crud/${item.id}`, {
        method: "DELETE",
        timeoutMs: 20000,
      });
      closeEditor();
      setManagerStatus("Питання видалено.", "success");
      refreshQuestionList();
    } catch (error) {
      setEditorStatus(error.message || "Не вдалося видалити питання.", "error");
      button.disabled = false;
    }
  });

  window.setTimeout(() => overlay.querySelector("#test-q-editor-question")?.focus(), 50);
}

async function openCardEditor(card) {
  if (editorOpen) return;
  const question = card.querySelector(".case-answer__question")?.textContent?.trim() || "";
  const num = card.querySelector(".case-answer__number")?.textContent?.trim() || "";
  const answer = card.querySelector(".case-answer__correct-text")?.textContent?.trim() || "";
  const module = card.querySelector(".case-answer__count")?.textContent?.trim() || "";
  if (!question) return;

  card.setAttribute("aria-busy", "true");
  setManagerStatus("Відкриваю питання…");
  try {
    const params = new URLSearchParams({ question, num, answer, module });
    const response = await api(`/api/admin/test-exam-questions/crud/lookup?${params.toString()}`, {
      timeoutMs: 15000,
    });
    setManagerStatus("");
    openEditor(response.item);
  } catch (error) {
    setManagerStatus(error.message || "Не вдалося відкрити питання.", "error");
  } finally {
    card.removeAttribute("aria-busy");
  }
}

function bindCards() {
  document.querySelectorAll("#test-q-list .case-answer").forEach((card) => {
    if (card.dataset.testCrudBound === "1") return;
    card.dataset.testCrudBound = "1";
    card.classList.add("test-q-crud-card");
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");

    const hint = document.createElement("div");
    hint.className = "test-q-crud-hint";
    hint.textContent = "Натисніть, щоб редагувати";
    card.append(hint);

    card.addEventListener("click", () => void openCardEditor(card));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        void openCardEditor(card);
      }
    });
  });
}

function injectManager() {
  const list = document.querySelector("#test-q-list");
  if (!list) return;
  ensureStyles();

  const content = list.closest(".screen-content");
  const search = content?.querySelector(".case-search");
  if (!content || !search) return;

  if (!document.querySelector("#test-q-crud-actions")) {
    const actions = document.createElement("div");
    actions.id = "test-q-crud-actions";
    actions.className = "test-q-crud-actions";
    actions.innerHTML = `
      <button class="btn btn--primary btn--block" id="test-q-crud-add" type="button">＋ Додати питання</button>
      <div class="test-q-crud-status" id="test-q-crud-status" aria-live="polite"></div>
    `;
    search.insertAdjacentElement("afterend", actions);
    actions.querySelector("#test-q-crud-add")?.addEventListener("click", () => openEditor());
  }

  bindCards();
}

const observer = new MutationObserver(() => injectManager());
observer.observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("DOMContentLoaded", injectManager);
injectManager();
