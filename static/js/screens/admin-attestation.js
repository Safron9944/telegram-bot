const ISSUE_LABELS = {
  low_ocr_confidence: "Низька впевненість OCR",
  page_break_not_closed: "Незавершене перенесення між сторінками",
  suspicious_word_break: "Ймовірно обірване слово",
  bold_choice_ambiguous: "Неоднозначне жирне виділення",
  missing_correct_answer: "Не визначено правильну відповідь",
  correct_answer_conflict: "Відповідь конфліктує з базою",
  duplicate_number: "Повторений номер питання",
  missing_from_extraction: "Питання не витягнулося з PDF",
  missing_question_number: "Не розпізнано номер",
  too_few_choices: "Не всі варіанти відповіді",
  empty_text: "Порожній або обірваний текст",
  unknown_section: "Не визначено розділ",
};


export function reviewReasonLabels(issues) {
  return (issues || []).map((issue) => ISSUE_LABELS[issue] || String(issue));
}


export function changedFields(before, after) {
  const fields = [];
  if (String(before?.question || "") !== String(after?.question || "")) {
    fields.push("question");
  }
  if (JSON.stringify(before?.choices || []) !== JSON.stringify(after?.choices || [])) {
    fields.push("choices");
  }
  if (JSON.stringify(before?.correct || []) !== JSON.stringify(after?.correct || [])) {
    fields.push("correct");
  }
  return fields;
}


function issueChips(ctx, issues) {
  return reviewReasonLabels(issues)
    .map((label) => `<span class="issue-chip">${ctx.escapeHtml(label)}</span>`)
    .join("");
}


function bestMatch(review) {
  return (review.matches || [])[0] || null;
}


function renderSummary(ctx, summary) {
  return `
    <div class="stat-strip">
      ${ctx.statPill("Перевірено", String(summary.verified || 0))}
      ${ctx.statPill("Проблемні", String(summary.needs_review || 0))}
      ${ctx.statPill("Збіг із базою", String(summary.matched_database || 0))}
    </div>
  `;
}


export function renderAdminAttestationReviews(ctx) {
  const summary = ctx.state.adminAttestationSummary || {};
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <div class="muted-sm" style="font-weight:700; letter-spacing:.08em; text-transform:uppercase;">Контроль якості PDF</div>
      <h1 class="page-title">Проблемні питання</h1>
      <p class="page-subtitle">Сумнівні записи не потрапляють у тести, доки адміністратор не перевірить повний текст і правильну відповідь.</p>
      <div id="attestation-review-summary">${renderSummary(ctx, summary)}</div>
      <div class="group">
        <div class="group__label">Потребують перевірки</div>
        <div class="group__list" id="attestation-review-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
      </div>
      <div class="row" id="attestation-review-pagination" style="justify-content:center; gap:8px;"></div>
    </section>
  `;
}


function fillReviewList(ctx, payload) {
  const list = ctx.refs.mainPanel.querySelector("#attestation-review-list");
  const pagination = ctx.refs.mainPanel.querySelector("#attestation-review-pagination");
  if (!list || !pagination) return;
  list.innerHTML = "";
  if (!payload.items?.length) {
    list.innerHTML = `<div class="empty empty--inline"><h2>Черга порожня</h2><p>Усі знайдені питання опрацьовані.</p></div>`;
    return;
  }

  payload.items.forEach((review) => {
    const match = bestMatch(review);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cell attestation-review-row";
    button.innerHTML = `
      <span class="cell__icon cell__icon--orange">${ctx.escapeHtml(review.qnum || "!")}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(review.extracted_question || "Текст не розпізнано")}</span>
        <span class="cell__subtitle">${ctx.escapeHtml(review.section_title || review.section)} · PDF ${ctx.escapeHtml(review.source_page)}</span>
        <span class="issue-chip-row">${issueChips(ctx, review.issues)}</span>
        ${match ? `<span class="cell__subtitle">Найкращий збіг: ${Math.round((match.score || 0) * 100)}%</span>` : ""}
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    button.addEventListener("click", () => void openAdminAttestationReview(ctx, review.id));
    list.append(button);
  });

  pagination.innerHTML = "";
  if (payload.offset > 0) {
    pagination.append(
      ctx.actionButton(
        "← Назад",
        () => loadAdminAttestationReviews(ctx, Math.max(0, payload.offset - payload.limit)),
        "sm",
      ),
    );
  }
  if (payload.offset + payload.limit < payload.total) {
    pagination.append(
      ctx.actionButton(
        "Далі →",
        () => loadAdminAttestationReviews(ctx, payload.offset + payload.limit),
        "sm",
      ),
    );
  }
}


export async function loadAdminAttestationReviews(ctx, offset = 0) {
  if (ctx.state.currentScreen !== "admin-attestation-reviews") return;
  try {
    const [summary, reviews] = await Promise.all([
      ctx.api("/api/admin/attestation/summary"),
      ctx.api(`/api/admin/attestation/reviews?status=needs_review&offset=${offset}&limit=20`),
    ]);
    if (ctx.state.currentScreen !== "admin-attestation-reviews") return;
    ctx.state.adminAttestationSummary = summary;
    ctx.state.adminAttestationReviews = reviews;
    const summaryRoot = ctx.refs.mainPanel.querySelector("#attestation-review-summary");
    if (summaryRoot) summaryRoot.innerHTML = renderSummary(ctx, summary);
    fillReviewList(ctx, reviews);
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}


async function openAdminAttestationReview(ctx, reviewId) {
  try {
    const payload = await ctx.api(`/api/admin/attestation/reviews/${reviewId}`);
    ctx.state.selectedAttestationReview = payload.review;
    renderAdminAttestationReviewEditor(ctx, payload.review);
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}


function choiceEditor(ctx, text, index, selected) {
  const row = document.createElement("div");
  row.className = "attestation-review__choice";
  row.innerHTML = `
    <label class="attestation-review__correct" title="Позначити правильною">
      <input type="radio" name="attestation-correct" value="${index + 1}" ${selected === index + 1 ? "checked" : ""} />
      <span>${index + 1}</span>
    </label>
    <textarea class="attestation-review__choice-text" rows="2" aria-label="Варіант ${index + 1}">${ctx.escapeHtml(text)}</textarea>
    <button class="attestation-review__delete" type="button" aria-label="Видалити варіант ${index + 1}">×</button>
  `;
  row.querySelector(".attestation-review__delete")?.addEventListener("click", () => {
    const parent = row.parentElement;
    row.remove();
    renumberChoices(parent);
  });
  return row;
}


function renumberChoices(root) {
  [...(root?.children || [])].forEach((row, index) => {
    const number = index + 1;
    const radio = row.querySelector('input[type="radio"]');
    const badge = row.querySelector(".attestation-review__correct span");
    const textarea = row.querySelector("textarea");
    const remove = row.querySelector(".attestation-review__delete");
    if (radio) radio.value = String(number);
    if (badge) badge.textContent = String(number);
    if (textarea) textarea.setAttribute("aria-label", `Варіант ${number}`);
    if (remove) remove.setAttribute("aria-label", `Видалити варіант ${number}`);
  });
}


function renderSourceComparison(ctx, review) {
  const match = bestMatch(review);
  return `
    <div class="attestation-review__compare">
      <article class="attestation-review__source">
        <div class="muted-sm">PDF · сторінка ${ctx.escapeHtml(review.source_page)}</div>
        <h2>${ctx.escapeHtml(review.extracted_question || "Текст питання не розпізнано")}</h2>
        <ol>${(review.extracted_choices || []).map((choice) => `<li>${ctx.escapeHtml(choice)}</li>`).join("")}</ol>
      </article>
      <article class="attestation-review__match">
        <div class="muted-sm">${match ? `Збіг із ${ctx.escapeHtml(match.source)} · ${Math.round((match.score || 0) * 100)}%` : "Збігу в базі не знайдено"}</div>
        ${match
          ? `<h2>${ctx.escapeHtml(match.question || "")}</h2><ol>${(match.choices || []).map((choice) => `<li>${ctx.escapeHtml(choice)}</li>`).join("")}</ol>`
          : `<p class="muted">Перевірте запис без підказки з наявних банків.</p>`}
      </article>
    </div>
  `;
}


export function renderAdminAttestationReviewEditor(ctx, review) {
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content attestation-review">
      <button class="back-nav" id="review-queue-back" type="button"><span aria-hidden="true">‹</span> До черги</button>
      <div class="row-between">
        <div>
          <div class="muted-sm" style="font-weight:700; letter-spacing:.08em; text-transform:uppercase;">${ctx.escapeHtml(review.section_title || review.section)}</div>
          <h1 class="page-title">Питання №${ctx.escapeHtml(review.qnum || "—")}</h1>
        </div>
        <span class="chip chip--danger">PDF ${ctx.escapeHtml(review.source_page)}</span>
      </div>
      <div class="issue-chip-row">${issueChips(ctx, review.issues)}</div>
      ${renderSourceComparison(ctx, review)}

      <section class="group attestation-review__editor">
        <div class="group__label">Перевірена редакція</div>
        <div class="group__list" style="padding:16px;">
          <label class="field-label" for="attestation-question-text">Повний текст питання</label>
          <textarea id="attestation-question-text" class="attestation-review__question" rows="5">${ctx.escapeHtml(review.extracted_question || "")}</textarea>
          <div class="field-label" style="margin-top:16px;">Варіанти · оберіть одну правильну відповідь</div>
          <div class="attestation-review__choices" id="attestation-choice-editor"></div>
          <button class="btn btn--ghost btn--block" id="attestation-add-choice" type="button">+ Додати відповідь</button>
        </div>
      </section>
      <div class="sticky-cta" id="attestation-review-actions"></div>
    </section>
  `;

  const choicesRoot = ctx.refs.mainPanel.querySelector("#attestation-choice-editor");
  const proposed = Array.isArray(review.proposed_correct)
    ? review.proposed_correct[0]
    : review.proposed_correct;
  (review.extracted_choices || []).forEach((choice, index) => {
    choicesRoot.append(choiceEditor(ctx, choice, index, Number(proposed)));
  });
  ctx.refs.mainPanel.querySelector("#attestation-add-choice")?.addEventListener("click", () => {
    const index = choicesRoot.children.length;
    choicesRoot.append(choiceEditor(ctx, "", index, null));
    renumberChoices(choicesRoot);
  });
  ctx.refs.mainPanel.querySelector("#review-queue-back")?.addEventListener("click", () => {
    ctx.state.selectedAttestationReview = null;
    renderAdminAttestationReviews(ctx);
    void loadAdminAttestationReviews(ctx, ctx.state.adminAttestationReviews?.offset || 0);
  });

  const actions = ctx.refs.mainPanel.querySelector("#attestation-review-actions");
  actions.append(
    ctx.actionButton(
      "Підтвердити й додати до тесту",
      () => approveReview(ctx, review),
      "block",
    ),
    ctx.actionButton(
      "Відхилити запис",
      () => rejectReview(ctx, review),
      "block-danger",
    ),
  );
}


async function approveReview(ctx, review) {
  const question = ctx.refs.mainPanel.querySelector("#attestation-question-text")?.value || "";
  const rows = [...ctx.refs.mainPanel.querySelectorAll(".attestation-review__choice")];
  const choices = rows.map((row) => row.querySelector("textarea")?.value || "");
  const selected = ctx.refs.mainPanel.querySelector('input[name="attestation-correct"]:checked');
  if (!selected) {
    ctx.setMessage("error", "Оберіть одну правильну відповідь.");
    return;
  }
  try {
    await ctx.api(`/api/admin/attestation/reviews/${review.id}/approve`, {
      method: "POST",
      body: { question, choices, correct: [Number(selected.value)] },
    });
    ctx.setMessage("success", "Питання перевірено й додано до тесту.");
    ctx.state.selectedAttestationReview = null;
    renderAdminAttestationReviews(ctx);
    await loadAdminAttestationReviews(ctx, 0);
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}


async function rejectReview(ctx, review) {
  try {
    await ctx.api(`/api/admin/attestation/reviews/${review.id}/reject`, {
      method: "POST",
    });
    ctx.setMessage("success", "Проблемний запис відхилено.");
    ctx.state.selectedAttestationReview = null;
    renderAdminAttestationReviews(ctx);
    await loadAdminAttestationReviews(ctx, 0);
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}
