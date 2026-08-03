import { refs } from "./core/dom.js?v=20260617-question-search-04";
import { api } from "./core/api.js?v=20260617-question-search-04";
import { escapeHtml, setMessage } from "./core/ui.js?v=20260617-question-search-04";

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
  overlay.style.zIndex = "9999";
  overlay.style.padding = "12px";
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeOverlay();
  });

  const modal = document.createElement("div");
  modal.className = "modal";
  modal.style.width = "min(760px, 100%)";
  modal.style.maxHeight = "calc(100dvh - 24px)";
  modal.style.display = "flex";
  modal.style.flexDirection = "column";
  modal.style.overflow = "hidden";

  const header = document.createElement("div");
  header.className = "modal__header";
  header.innerHTML = `
    <button class="btn btn--sm" type="button" id="attestation-admin-back">‹ Назад</button>
    <span class="modal__heading" style="flex: 1; min-width: 0;">
      <span class="modal__title" id="attestation-admin-title"></span>
      <span class="modal__subtitle" id="attestation-admin-subtitle"></span>
    </span>
    <button class="modal__close" type="button" id="attestation-admin-close" aria-label="Закрити">✕</button>
  `;

  modalContent = document.createElement("div");
  modalContent.style.overflowY = "auto";
  modalContent.style.padding = "14px";
  modalContent.style.flex = "1";

  modal.append(header, modalContent);
  overlay.append(modal);
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
  modalContent.innerHTML = `
    <div class="empty empty--inline">
      <h2>${escapeHtml(text)}</h2>
    </div>
  `;
}

function renderError(error) {
  if (!modalContent) return;
  modalContent.innerHTML = `
    <div class="empty empty--inline">
      <h2>Не вдалося завантажити</h2>
      <p>${escapeHtml(error.message || String(error))}</p>
    </div>
  `;
}

function makeCell({ icon, tint = "purple", title, subtitle, onClick }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cell";
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
    const payload = await api("/api/admin/attestation-stage-1/export");

    if (payload.count !== 800) {
      const confirmed = window.confirm(
        `У базі знайдено ${payload.count} питань замість 800. Все одно завантажити файл?`,
      );
      if (!confirmed) return;
    }

    const jsonText = JSON.stringify(payload, null, 2);
    const blob = new Blob([jsonText], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const date = new Date().toISOString().slice(0, 10);

    link.href = url;
    link.download = `attestation_stage_1_current_${date}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);

    setMessage("success", `Експортовано ${payload.count} актуальних питань.`);
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
      <form id="attestation-question-edit-form" class="stack" style="gap: 12px;">
        <div class="field">
          <label class="field__label" for="attestation-question-text">Текст питання</label>
          <textarea id="attestation-question-text" class="textarea">${escapeHtml(question.question)}</textarea>
        </div>
        <div id="attestation-choices-editor" class="stack" style="gap: 10px;"></div>
        <div class="row" style="gap: 8px; margin-top: 4px;">
          <button class="btn btn--primary btn--lg" type="submit" style="flex: 1;">Зберегти</button>
          <button class="btn btn--lg" type="button" id="attestation-question-reset">Скинути</button>
        </div>
      </form>
    `;

    const choicesEditor = modalContent.querySelector("#attestation-choices-editor");
    question.choices.forEach((choice) => {
      const block = document.createElement("div");
      block.className = "stack";
      block.style.gap = "6px";
      block.style.padding = "10px";
      block.style.borderRadius = "10px";
      block.style.background = "var(--bg-fill-soft)";
      block.innerHTML = `
        <div class="field">
          <label class="field__label" for="attestation-choice-${choice.index}">Варіант ${choice.index}</label>
          <textarea id="attestation-choice-${choice.index}" class="textarea" style="min-height: 60px;">${escapeHtml(choice.text)}</textarea>
        </div>
        <label class="row" style="gap: 10px; cursor: pointer;">
          <span class="switch">
            <input id="attestation-correct-${choice.index}" type="checkbox" ${choice.is_correct ? "checked" : ""} />
            <span class="switch__track"></span>
          </span>
          <span style="font-size: 14px; font-weight: 500;">Правильна відповідь</span>
        </label>
      `;
      choicesEditor.append(block);
    });

    modalContent
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
          setMessage("error", "Текст питання не може бути порожнім.");
          return;
        }
        if (choices.some((item) => !item)) {
          setMessage("error", "Заповніть усі варіанти відповіді.");
          return;
        }
        if (!correct.length) {
          setMessage("error", "Позначте хоча б одну правильну відповідь.");
          return;
        }

        try {
          await api(`/api/admin/questions/${questionId}`, {
            method: "PATCH",
            body: {
              question: questionText,
              choices,
              correct,
            },
          });
          setMessage("success", "Питання збережено.");
          await renderEditor(questionId);
        } catch (error) {
          setMessage("error", error.message);
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
