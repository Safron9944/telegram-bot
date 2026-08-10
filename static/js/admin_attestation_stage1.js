import { refs } from "./core/dom.js?v=20260617-question-search-04";
import { api } from "./core/api.js?v=20260617-question-search-04";
import { tg } from "./core/telegram.js?v=20260523-cases-search-02";
import { escapeHtml, setMessage } from "./core/ui.js?v=20260809-unified-ui-02";

const ENTRY_ID = "admin-attestation-stage1-entry";
const OVERLAY_ID = "admin-attestation-stage1-overlay";
const PAGE_SIZE = 50;

let currentView = "sections";
let selectedSection = "";
let selectedOffset = 0;
let overlay = null;
let modalTitle = null;
let modalSubtitle = null;
let modalBack = null;
let modalContent = null;
let modalFooter = null;
let previousBodyOverflow = "";

function ensureAdminEntry() {
  if (!refs.mainPanel) return;

  const heading = refs.mainPanel.querySelector(".page-title");
  if (!heading || heading.textContent.trim() !== "Адмін") return;
  if (refs.mainPanel.querySelector(`#${ENTRY_ID}`)) return;

  const list = refs.mainPanel.querySelector(".group__list");
  if (!list) return;

  const entry = document.createElement("button");
  entry.id = ENTRY_ID;
  entry.type = "button";
  entry.className = "cell";
  entry.innerHTML = `
    <span class="cell__icon cell__icon--orange">А1</span>
    <span class="cell__body">
      <span class="cell__title">Атестація посадових осіб — 1 етап</span>
      <span class="cell__subtitle">Розділи, питання та редагування</span>
    </span>
    <span class="cell__chevron" aria-hidden="true"></span>
  `;
  entry.addEventListener("click", openOverlay);

  const bankQuestions = Array.from(list.querySelectorAll(".cell")).find(
    (item) => item.querySelector(".cell__title")?.textContent.trim() === "Банк питань",
  );
  if (bankQuestions) {
    bankQuestions.after(entry);
  } else {
    list.prepend(entry);
  }
}

function createOverlay() {
  document.querySelector(`#${OVERLAY_ID}`)?.remove();

  overlay = document.createElement("div");
  overlay.id = OVERLAY_ID;
  overlay.className = "modal-overlay";
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeOverlay();
  });

  const modal = document.createElement("div");
  modal.className = "modal attestation-admin-modal";

  const header = document.createElement("div");
  header.className = "modal__header attestation-admin-header";
  header.innerHTML = `
    <button class="btn btn--sm attestation-admin-back" type="button" id="attestation-admin-back">‹ Назад</button>
    <span class="modal__heading">
      <span class="modal__title" id="attestation-admin-title"></span>
      <span class="modal__subtitle" id="attestation-admin-subtitle"></span>
    </span>
    <button class="modal__close" type="button" id="attestation-admin-close" aria-label="Закрити">✕</button>
  `;

  modalContent = document.createElement("div");
  modalContent.className = "attestation-admin-content";

  modalFooter = document.createElement("div");
  modalFooter.className = "attestation-admin-footer";
  modalFooter.hidden = true;

  modal.append(header, modalContent, modalFooter);
  overlay.append(modal);
  previousBodyOverflow = document.body.style.overflow;
  document.body.style.overflow = "hidden";
  document.body.append(overlay);

  modalTitle = header.querySelector("#attestation-admin-title");
  modalSubtitle = header.querySelector("#attestation-admin-subtitle");
  modalBack = header.querySelector("#attestation-admin-back");

  modalBack.addEventListener("click", goBack);
  header.querySelector("#attestation-admin-close").addEventListener("click", closeOverlay);
}

function openOverlay() {
  createOverlay();
  void renderSections();
}

function closeOverlay() {
  overlay?.remove();
  overlay = null;
  modalTitle = null;
  modalSubtitle = null;
  modalBack = null;
  modalContent = null;
  modalFooter = null;
  document.body.style.overflow = previousBodyOverflow;
  previousBodyOverflow = "";
  currentView = "sections";
  selectedSection = "";
  selectedOffset = 0;
}

function goBack() {
  if (currentView === "editor") {
    void renderQuestions(selectedSection, selectedOffset);
    return;
  }
  if (currentView === "questions") {
    void renderSections();
    return;
  }
  closeOverlay();
}

function setHeader(title, subtitle, showBack = true) {
  if (modalTitle) modalTitle.textContent = title;
  if (modalSubtitle) modalSubtitle.textContent = subtitle || "";
  if (modalBack) modalBack.hidden = !showBack;
}

function setLoading(text = "Завантажуємо…") {
  if (!modalContent) return;
  if (modalFooter) {
    modalFooter.hidden = true;
    modalFooter.innerHTML = "";
  }
  modalContent.scrollTop = 0;
  modalContent.innerHTML = `
    <div class="empty empty--inline">
      <h2>${escapeHtml(text)}</h2>
    </div>
  `;
}

function renderError(error) {
  if (!modalContent) return;
  if (modalFooter) modalFooter.hidden = true;
  modalContent.innerHTML = `
    <div class="empty empty--inline">
      <h2>Не вдалося завантажити</h2>
      <p>${escapeHtml(error.message || String(error))}</p>
    </div>
  `;
}

function showEditorNotice(type, message) {
  const notice = modalContent?.querySelector("#attestation-editor-notice");
  if (!notice) return;
  notice.className = `attestation-editor-notice attestation-editor-notice--${type}`;
  notice.textContent = message;
  notice.hidden = false;
  notice.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function makeCell({ icon, tint = "purple", title, subtitle, onClick }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cell attestation-admin-cell";
  button.innerHTML = `
    <span class="cell__icon cell__icon--${tint}">${escapeHtml(String(icon))}</span>
    <span class="cell__body">
      <span class="cell__title">${escapeHtml(title)}</span>
      <span class="cell__subtitle">${escapeHtml(subtitle || "")}</span>
    </span>
    <span class="cell__chevron" aria-hidden="true"></span>
  `;
  button.addEventListener("click", onClick);
  return button;
}

async function downloadCurrentQuestionsJson() {
  try {
    const ticket = await api("/api/admin/attestation-stage-1/export-ticket", {
      method: "POST",
    });

    if (ticket.count !== 800) {
      const confirmed = window.confirm(
        `У базі знайдено ${ticket.count} питань замість 800. Все одно завантажити файл?`,
      );
      if (!confirmed) return;
    }

    const downloadUrl = new URL(ticket.download_path, window.location.origin).href;
    const canUseTelegramDownload =
      downloadUrl.startsWith("https://") &&
      tg?.isVersionAtLeast?.("8.0") &&
      typeof tg.downloadFile === "function";

    if (canUseTelegramDownload) {
      try {
        tg.downloadFile(
          {
            url: downloadUrl,
            file_name: ticket.file_name,
          },
          (accepted) => {
            if (accepted) {
              setMessage("success", `Завантаження ${ticket.count} питань розпочато.`);
            }
          },
        );
        return;
      } catch {
        // Some older Telegram clients expose the method before it is usable.
      }
    }

    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = ticket.file_name;
    link.target = "_blank";
    link.rel = "noopener";
    document.body.append(link);
    link.click();
    link.remove();

    setMessage("success", `Завантаження ${ticket.count} питань розпочато.`);
  } catch (error) {
    setMessage("error", error.message || "Не вдалося створити JSON-файл.");
  }
}

async function renderSections() {
  if (!modalContent) return;
  currentView = "sections";
  selectedSection = "";
  selectedOffset = 0;
  setHeader("Атестація посадових осіб — 1 етап", "Оберіть розділ", false);
  setLoading();

  try {
    const payload = await api("/api/admin/attestation-stage-1/sections");
    if (!modalContent || currentView !== "sections") return;

    modalContent.innerHTML = "";

    const exportRow = document.createElement("div");
    exportRow.className = "row";
    exportRow.style.marginBottom = "14px";

    const exportButton = document.createElement("button");
    exportButton.type = "button";
    exportButton.className = "btn btn--primary";
    exportButton.style.width = "100%";
    exportButton.textContent = "⬇ Експорт поточних питань у JSON";
    exportButton.addEventListener("click", downloadCurrentQuestionsJson);

    exportRow.append(exportButton);
    modalContent.append(exportRow);

    const group = document.createElement("div");
    group.className = "group";
    const list = document.createElement("div");
    list.className = "group__list";

    if (!payload.items?.length) {
      list.innerHTML = `
        <div class="empty empty--inline">
          <h2>Розділів немає</h2>
          <p>Питання першого етапу ще не завантажені.</p>
        </div>
      `;
    } else {
      payload.items.forEach((item, index) => {
        list.append(
          makeCell({
            icon: index + 1,
            tint: "orange",
            title: item.title,
            subtitle: `${item.count} питань`,
            onClick: () => void renderQuestions(item.key, 0),
          }),
        );
      });
    }

    group.append(list);
    modalContent.append(group);
  } catch (error) {
    renderError(error);
  }
}

async function renderQuestions(section, offset = 0) {
  if (!modalContent) return;
  currentView = "questions";
  selectedSection = section;
  selectedOffset = Math.max(0, Number(offset) || 0);
  setHeader(section, "Натисніть на потрібне питання", true);
  setLoading();

  try {
    const payload = await api(
      `/api/admin/attestation-stage-1/questions?section=${encodeURIComponent(section)}&offset=${selectedOffset}&limit=${PAGE_SIZE}`,
    );
    if (!modalContent || currentView !== "questions" || selectedSection !== section) return;

    modalContent.innerHTML = "";

    const group = document.createElement("div");
    group.className = "group";
    const list = document.createElement("div");
    list.className = "group__list";

    payload.items.forEach((item, index) => {
      const number = item.qnum ?? payload.offset + index + 1;
      list.append(
        makeCell({
          icon: `#${number}`,
          title: item.question,
          subtitle: `Питання №${number}`,
          onClick: () => void renderEditor(item.id),
        }),
      );
    });

    group.append(list);
    modalContent.append(group);

    const pagination = document.createElement("div");
    pagination.className = "row";
    pagination.style.justifyContent = "center";
    pagination.style.gap = "8px";
    pagination.style.marginTop = "14px";

    if (payload.has_prev) {
      const previous = document.createElement("button");
      previous.type = "button";
      previous.className = "btn btn--sm";
      previous.textContent = "← Попередні";
      previous.addEventListener("click", () =>
        void renderQuestions(section, Math.max(0, payload.offset - payload.limit)),
      );
      pagination.append(previous);
    }

    if (payload.has_next) {
      const next = document.createElement("button");
      next.type = "button";
      next.className = "btn btn--sm";
      next.textContent = "Наступні →";
      next.addEventListener("click", () =>
        void renderQuestions(section, payload.offset + payload.limit),
      );
      pagination.append(next);
    }

    if (pagination.children.length) modalContent.append(pagination);
  } catch (error) {
    renderError(error);
  }
}

async function renderEditor(questionId) {
  if (!modalContent) return;
  currentView = "editor";
  setHeader(selectedSection, `Редагування питання #${questionId}`, true);
  setLoading();

  try {
    const payload = await api(`/api/admin/questions/${questionId}`);
    if (!modalContent || currentView !== "editor") return;

    const question = payload.question;
    modalContent.innerHTML = `
      <div id="attestation-editor-notice" class="attestation-editor-notice" role="status" aria-live="polite" hidden></div>
      <form id="attestation-question-edit-form" class="attestation-editor-form">
        <div class="field">
          <label class="field__label" for="attestation-question-text">Текст питання</label>
          <textarea id="attestation-question-text" class="textarea attestation-question-text">${escapeHtml(question.question)}</textarea>
        </div>
        <div id="attestation-choices-editor" class="attestation-choices-editor"></div>
      </form>
    `;

    const choicesEditor = modalContent.querySelector("#attestation-choices-editor");
    question.choices.forEach((choice) => {
      const block = document.createElement("div");
      block.className = "attestation-choice-card";
      block.innerHTML = `
        <div class="attestation-choice-card__header">
          <label class="field__label" for="attestation-choice-${choice.index}">Варіант ${choice.index}</label>
          <label class="attestation-correct-toggle" for="attestation-correct-${choice.index}">
            <span>Правильна</span>
            <span class="switch">
              <input id="attestation-correct-${choice.index}" type="checkbox" ${choice.is_correct ? "checked" : ""} />
              <span class="switch__track"></span>
            </span>
          </label>
        </div>
        <textarea id="attestation-choice-${choice.index}" class="textarea attestation-choice-text">${escapeHtml(choice.text)}</textarea>
      `;
      choicesEditor.append(block);
    });

    modalFooter.innerHTML = `
      <button class="btn btn--primary btn--lg" type="submit" form="attestation-question-edit-form" id="attestation-question-save">Зберегти зміни</button>
      <button class="btn btn--lg" type="button" id="attestation-question-reset">Скинути</button>
    `;
    modalFooter.hidden = false;

    modalFooter
      .querySelector("#attestation-question-reset")
      .addEventListener("click", () => void renderEditor(questionId));

    modalContent
      .querySelector("#attestation-question-edit-form")
      .addEventListener("submit", async (event) => {
        event.preventDefault();

        const questionText = modalContent
          .querySelector("#attestation-question-text")
          .value.trim();
        const choices = [];
        const correct = [];

        question.choices.forEach((choice) => {
          choices.push(
            modalContent.querySelector(`#attestation-choice-${choice.index}`).value.trim(),
          );
          if (modalContent.querySelector(`#attestation-correct-${choice.index}`).checked) {
            correct.push(choice.index);
          }
        });

        if (!questionText) {
          showEditorNotice("error", "Текст питання не може бути порожнім.");
          return;
        }
        if (choices.some((item) => !item)) {
          showEditorNotice("error", "Заповніть усі варіанти відповіді.");
          return;
        }
        if (!correct.length) {
          showEditorNotice("error", "Позначте хоча б одну правильну відповідь.");
          return;
        }

        const saveButton = modalFooter?.querySelector("#attestation-question-save");
        const resetButton = modalFooter?.querySelector("#attestation-question-reset");
        try {
          if (saveButton) {
            saveButton.disabled = true;
            saveButton.textContent = "Зберігаємо…";
          }
          if (resetButton) resetButton.disabled = true;
          await api(`/api/admin/questions/${questionId}`, {
            method: "PATCH",
            body: {
              question: questionText,
              choices,
              correct,
            },
          });
          await renderEditor(questionId);
          showEditorNotice("success", "Питання збережено.");
        } catch (error) {
          showEditorNotice("error", error.message || "Не вдалося зберегти питання.");
          if (saveButton) {
            saveButton.disabled = false;
            saveButton.textContent = "Зберегти зміни";
          }
          if (resetButton) resetButton.disabled = false;
        }
      });
  } catch (error) {
    renderError(error);
  }
}

refs.backButton?.addEventListener(
  "click",
  (event) => {
    if (!document.querySelector(`#${OVERLAY_ID}`)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    goBack();
  },
  true,
);

const observer = new MutationObserver(ensureAdminEntry);
if (refs.mainPanel) {
  observer.observe(refs.mainPanel, { childList: true, subtree: true });
}
ensureAdminEntry();
