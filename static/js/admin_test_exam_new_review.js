import { api } from "./core/api.js?v=20260617-question-search-04";

let latestPreview = null;
let reviewed = new Set();
let previewRequestId = 0;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function ensureStyles() {
  if (document.querySelector("#admin-test-new-review-styles")) return;
  const style = document.createElement("style");
  style.id = "admin-test-new-review-styles";
  style.textContent = `
    .test-import-new-review { display: grid; gap: 10px; margin-top: 12px; }
    .test-import-new-review__header { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
    .test-import-new-review__title { font-weight: 700; line-height: 1.3; }
    .test-import-new-review__counter { flex: 0 0 auto; font-size: 12px; opacity: .72; }
    .test-import-new-card { border: 1px solid var(--separator, rgba(128,128,128,.22)); border-radius: 14px; padding: 12px; display: grid; gap: 9px; }
    .test-import-new-card__meta { font-size: 12px; opacity: .7; }
    .test-import-new-card__question { font-weight: 650; line-height: 1.4; }
    .test-import-new-card__answer { padding: 9px 10px; border-radius: 10px; background: var(--bg-fill-soft); font-size: 13px; line-height: 1.4; }
    .test-import-new-card__answer b { display: block; margin-bottom: 4px; font-size: 11px; text-transform: uppercase; opacity: .65; }
    .test-import-new-card__support { font-size: 12px; line-height: 1.4; opacity: .78; overflow-wrap: anywhere; }
    .test-import-new-card__check { display: flex; gap: 9px; align-items: flex-start; font-size: 13px; line-height: 1.35; cursor: pointer; padding-top: 2px; }
    .test-import-review-warning { padding: 10px 12px; border-radius: 10px; background: var(--bg-fill-soft); font-size: 13px; line-height: 1.4; }
  `;
  document.head.append(style);
}

function newItems() {
  return Array.isArray(latestPreview?.new) ? latestPreview.new : [];
}

function allNewReviewed() {
  const items = newItems();
  return items.length === 0 || items.every((item) => reviewed.has(String(item.import_index)));
}

function setImportStatus(message) {
  const status = document.querySelector("#test-import-status");
  if (status) status.textContent = message;
}

function enforceApplyButton() {
  const button = document.querySelector("#test-import-apply");
  if (!button || !latestPreview) return;
  if (!allNewReviewed()) button.disabled = true;
}

function renderNewReview() {
  const result = document.querySelector("#test-import-result");
  if (!result || !latestPreview || !result.querySelector(".test-import-summary")) return;

  let block = result.querySelector("#test-import-new-review");
  const items = newItems();
  if (!items.length) {
    block?.remove();
    enforceApplyButton();
    return;
  }

  ensureStyles();
  if (!block) {
    block = document.createElement("section");
    block.id = "test-import-new-review";
    block.className = "test-import-new-review";
    const conflictsLabel = Array.from(result.querySelectorAll(".group__label"))
      .find((node) => node.textContent?.includes("конфлікт"));
    if (conflictsLabel) result.insertBefore(block, conflictsLabel);
    else {
      const applyButton = result.querySelector("#test-import-apply");
      if (applyButton) result.insertBefore(block, applyButton);
      else result.append(block);
    }
  }

  const done = items.filter((item) => reviewed.has(String(item.import_index))).length;
  block.innerHTML = `
    <div class="test-import-new-review__header">
      <div>
        <div class="test-import-new-review__title">Нові питання — перевір перед додаванням</div>
        <div class="test-import-file" style="margin-top:3px;">Цих питань ще немає в базі. Переглянь питання та правильну відповідь і підтвердь кожне.</div>
      </div>
      <div class="test-import-new-review__counter">${done}/${items.length}</div>
    </div>
    ${!allNewReviewed() ? `<div class="test-import-review-warning">Імпорт нових питань заблокований, доки не підтверджено кожне нове питання.</div>` : ""}
    ${items.map((item) => {
      const index = String(item.import_index);
      const meta = [item.num, item.module, item.source].filter(Boolean).join(" · ");
      return `
        <article class="test-import-new-card">
          ${meta ? `<div class="test-import-new-card__meta">${escapeHtml(meta)}</div>` : ""}
          <div class="test-import-new-card__question">${escapeHtml(item.question)}</div>
          <div class="test-import-new-card__answer"><b>Правильна відповідь із файлу</b>${escapeHtml(item.correct_answer || "—")}</div>
          ${item.justification ? `<div class="test-import-new-card__support"><b>Обґрунтування:</b> ${escapeHtml(item.justification)}</div>` : ""}
          <label class="test-import-new-card__check">
            <input type="checkbox" data-review-new-index="${index}" ${reviewed.has(index) ? "checked" : ""} />
            <span>Перевірено: питання і правильна відповідь вірні</span>
          </label>
        </article>
      `;
    }).join("")}
  `;

  block.querySelectorAll("[data-review-new-index]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const index = String(checkbox.dataset.reviewNewIndex);
      if (checkbox.checked) reviewed.add(index);
      else reviewed.delete(index);
      renderNewReview();
      queueMicrotask(enforceApplyButton);
    });
  });
  enforceApplyButton();
}

async function inspectFile(file) {
  if (!file) return;
  const requestId = ++previewRequestId;
  reviewed = new Set();
  latestPreview = null;
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const preview = await api("/api/admin/test-exam-questions/import/preview", {
      method: "POST",
      body: form,
      timeoutMs: 30000,
    });
    if (requestId !== previewRequestId) return;
    latestPreview = preview;
    renderNewReview();
  } catch (_) {
    if (requestId === previewRequestId) latestPreview = null;
  }
}

function attachFileListener() {
  const input = document.querySelector("#test-import-file");
  if (!input || input.dataset.newReviewBound === "1") return;
  input.dataset.newReviewBound = "1";
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) void inspectFile(file);
  });
}

function sync() {
  attachFileListener();
  const result = document.querySelector("#test-import-result");
  const input = document.querySelector("#test-import-file");
  if (result && !result.textContent?.trim() && input && !input.value) {
    latestPreview = null;
    reviewed = new Set();
  }
  renderNewReview();
  enforceApplyButton();
}

document.addEventListener("change", (event) => {
  if (event.target?.closest?.("#test-import-conflicts")) queueMicrotask(enforceApplyButton);
}, true);

document.addEventListener("click", (event) => {
  const button = event.target?.closest?.("#test-import-apply");
  if (!button || !latestPreview || allNewReviewed()) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  setImportStatus("Спочатку перевір і підтвердь усі нові питання.");
  document.querySelector("#test-import-new-review")?.scrollIntoView({ behavior: "smooth", block: "start" });
}, true);

const observer = new MutationObserver(sync);
observer.observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("DOMContentLoaded", sync);
sync();