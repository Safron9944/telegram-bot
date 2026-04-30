function statusToClass(status) {
  if (status === "correct") return "choice--correct";
  if (status === "chosen") return "choice--chosen";
  return "";
}

function statusToLabel(status) {
  if (status === "correct") return "Правильна відповідь";
  if (status === "chosen") return "Ваш вибір";
  return "";
}

function progressBar(current, total) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  return `
    <div class="progress" aria-label="Прогрес">
      <div class="progress__bar" style="width: ${pct}%"></div>
    </div>
  `;
}

/* ===================== PRETEST ===================== */
function renderPretest(ctx, view) {
  const total = view.total || 0;
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Перед стартом</h1>
      <p class="page-subtitle">${ctx.escapeHtml(view.header || "Швидкий перегляд перед стартом.")}</p>

      <div class="group">
        <div class="group__label">Питання (${total})</div>
        <div class="group__list" style="padding: 12px;">
          <div class="numgrid" id="pretest-grid"></div>
        </div>
      </div>

      <div class="question-card" id="pretest-preview"></div>

      <div class="sticky-cta" id="pretest-actions"></div>
    </section>
  `;

  const grid = ctx.refs.mainPanel.querySelector("#pretest-grid");
  for (let index = 0; index < total; index += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === view.selected_index ? "is-active" : "";
    button.textContent = String(index + 1);
    button.addEventListener("click", async () => {
      try {
        ctx.impact("light");
        ctx.state.currentView = await ctx.api("/api/pretest/select", { method: "POST", body: { index } });
        ctx.render();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    });
    grid.append(button);
  }

  ctx.refs.mainPanel.querySelector("#pretest-preview").innerHTML = previewQuestion(
    ctx,
    view.question,
    view.selected_index + 1,
    total,
  );

  ctx.refs.mainPanel.querySelector("#pretest-actions").append(
    ctx.actionButton(
      "Почати навчання",
      async () => {
        try {
          ctx.state.currentView = await ctx.api("/api/pretest/start", { method: "POST" });
          ctx.impact("medium");
          ctx.render();
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      },
      "block",
    ),
    ctx.actionButton("Закрити", ctx.leaveCurrentView, "block-ghost"),
  );
}

function previewQuestion(ctx, question, index, total) {
  const options = question.choices
    .map(
      (choice) => `
        <div class="choice ${choice.is_correct ? "choice--correct" : ""}">
          <span class="choice__index">${choice.index}</span>
          <div class="choice__text">
            <div>${ctx.escapeHtml(choice.text)}</div>
            ${choice.is_correct ? `<div class="choice__hint">Правильна</div>` : ""}
          </div>
        </div>
      `,
    )
    .join("");

  return `
    <div class="question-card__meta">Питання ${index} / ${total}</div>
    <h3 class="question-card__text">${ctx.escapeHtml(question.question)}</h3>
    ${question.ok_label || question.topic || question.section
      ? `<div class="question-card__topic">${ctx.escapeHtml(question.ok_label || question.topic || question.section)}</div>`
      : ""}
    <div class="stack" style="gap: 8px;">${options}</div>
  `;
}

/* ===================== QUESTION (active) ===================== */
function renderQuestionView(ctx, view) {
  const question = view.question;
  const choices = question.choices
    .map(
      (choice, index) => `
        <button class="choice" type="button" data-choice="${index}">
          <span class="choice__index">${choice.index}</span>
          <span class="choice__text">${ctx.escapeHtml(choice.text)}</span>
        </button>
      `,
    )
    .join("");

  ctx.setChrome({ showBack: true });

  const phase = view.progress.phase === "skipped" ? " · повтор пропущених" : "";

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <div class="row-between">
        <div class="muted-sm" style="font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;">
          ${ctx.escapeHtml(view.header || "Сесія")}
        </div>
        <div class="muted-sm">${view.progress.current} / ${view.progress.total}${phase}</div>
      </div>

      ${progressBar(view.progress.current, view.progress.total)}

      <div class="question-card">
        <h3 class="question-card__text">${ctx.escapeHtml(question.question)}</h3>
        ${question.ok_label || question.topic || question.section
          ? `<div class="question-card__topic">${ctx.escapeHtml(question.ok_label || question.topic || question.section)}</div>`
          : ""}
      </div>

      <div class="stack" style="gap: 8px;">${choices}</div>

      <div class="sticky-cta" id="question-actions"></div>
    </section>
  `;

  ctx.refs.mainPanel.querySelectorAll("[data-choice]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        ctx.impact("medium");
        ctx.state.currentView = await ctx.api("/api/session/answer", {
          method: "POST",
          body: { choice: Number(button.dataset.choice) },
        });
        ctx.render();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    });
  });

  const actions = ctx.refs.mainPanel.querySelector("#question-actions");
  if (view.actions.allow_skip) {
    actions.append(
      ctx.actionButton(
        "Пропустити питання",
        async () => {
          try {
            ctx.state.currentView = await ctx.api("/api/session/skip", { method: "POST" });
            ctx.render();
          } catch (error) {
            ctx.setMessage("error", error.message);
          }
        },
        "block-ghost",
      ),
    );
  }
  actions.append(ctx.actionButton("Завершити", ctx.leaveCurrentView, "block-ghost"));
}

/* ===================== FEEDBACK ===================== */
function renderFeedbackView(ctx, view) {
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice ${statusToClass(option.status)}">
          <span class="choice__index">${option.index}</span>
          <div class="choice__text">
            <div>${ctx.escapeHtml(option.text)}</div>
            ${statusToLabel(option.status) ? `<div class="choice__hint">${statusToLabel(option.status)}</div>` : ""}
          </div>
        </div>
      `,
    )
    .join("");

  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <div class="muted-sm" style="font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;">
        Розбір відповіді
      </div>

      <div class="question-card">
        <h3 class="question-card__text">${ctx.escapeHtml(view.question.question)}</h3>
      </div>

      <div class="stack" style="gap: 8px;">${options}</div>

      <div class="sticky-cta" id="feedback-actions"></div>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#feedback-actions").append(
    ctx.actionButton(
      "Продовжити",
      async () => {
        try {
          ctx.state.currentView = await ctx.api("/api/session/next", { method: "POST" });
          ctx.impact("medium");
          ctx.render();
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      },
      "block",
    ),
    ctx.actionButton("Завершити", ctx.leaveCurrentView, "block-ghost"),
  );
}

/* ===================== RESULT ===================== */
function renderResultView(ctx, view) {
  const summary = view.summary || {};
  const blocks = summary.blocks || [];
  const pct = typeof summary.percent === "number" ? summary.percent : null;
  const pctClass = pct == null ? "" : pct >= 60 ? "result-hero__pct--success" : "result-hero__pct--danger";

  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${ctx.escapeHtml(summary.title || "Результат")}</h1>

      <div class="result-hero">
        <div class="result-hero__pct ${pctClass}">${pct == null ? "—" : pct.toFixed(0) + "%"}</div>
        <div class="result-hero__label">
          ${typeof summary.correct === "number" ? `${summary.correct} з ${summary.total} правильно` : "Підсумок"}
        </div>
        ${typeof summary.passed === "boolean"
          ? `<div class="chip ${summary.passed ? "chip--success" : "chip--danger"}" style="margin-top: 6px;">
              ${summary.passed ? "Складено" : "Не складено"} · поріг 60%
            </div>`
          : ""}
      </div>

      <div class="stat-strip">
        ${typeof summary.correct === "number" ? ctx.statPill("Правильно", `${summary.correct}/${summary.total}`) : ""}
        ${typeof summary.percent === "number" ? ctx.statPill("Відсоток", `${summary.percent.toFixed(0)}%`) : ""}
        ${typeof summary.remaining === "number" ? ctx.statPill("Помилок", String(summary.remaining)) : ""}
      </div>

      ${blocks.length
        ? `<div class="group">
            <div class="group__label">По блоках</div>
            <div class="group__list" id="result-blocks"></div>
          </div>`
        : ""}

      <div class="sticky-cta" id="result-actions"></div>
    </section>
  `;

  if (blocks.length) {
    const blocksRoot = ctx.refs.mainPanel.querySelector("#result-blocks");
    blocks.forEach((block) => {
      const row = document.createElement("div");
      row.className = "cell";
      row.style.cursor = "default";
      const blockPct = block.total > 0 ? (block.correct / block.total) * 100 : 0;
      const tint = blockPct >= 60 ? "green" : blockPct >= 40 ? "orange" : "red";
      row.innerHTML = `
        <span class="cell__icon cell__icon--${tint}">${block.correct}</span>
        <span class="cell__body">
          <span class="cell__title">${ctx.escapeHtml(block.name)}</span>
          <span class="cell__subtitle">${block.correct} з ${block.total}</span>
        </span>
        <span class="cell__detail">${blockPct.toFixed(0)}%</span>
      `;
      blocksRoot.append(row);
    });
  }

  const actions = ctx.refs.mainPanel.querySelector("#result-actions");
  if (view.mode === "test_result" && view.wrong_count > 0) {
    actions.append(
      ctx.actionButton(
        "Показати помилки",
        async () => {
          try {
            ctx.state.currentView = await ctx.api("/api/test/review/open", { method: "POST" });
            ctx.render();
          } catch (error) {
            ctx.setMessage("error", error.message);
          }
        },
        "block",
      ),
    );
  }
  actions.append(
    ctx.actionButton(
      "На головну",
      async () => {
        ctx.state.currentView = null;
        ctx.goHome();
        await ctx.loadBootstrap();
      },
      view.mode === "test_result" && view.wrong_count > 0 ? "block-ghost" : "block",
    ),
  );
}

/* ===================== REVIEW (mistakes navigation) ===================== */
function renderReviewView(ctx, view) {
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice ${statusToClass(option.status)}">
          <span class="choice__index">${option.index}</span>
          <div class="choice__text">
            <div>${ctx.escapeHtml(option.text)}</div>
            ${statusToLabel(option.status) ? `<div class="choice__hint">${statusToLabel(option.status)}</div>` : ""}
          </div>
        </div>
      `,
    )
    .join("");

  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <div class="row-between">
        <div class="muted-sm" style="font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;">
          Помилки тесту
        </div>
        <div class="muted-sm">${view.index + 1} / ${view.total}</div>
      </div>

      ${progressBar(view.index + 1, view.total)}

      <div class="question-card">
        <h3 class="question-card__text">${ctx.escapeHtml(view.question.question)}</h3>
        ${view.question.selected_missing ? `<p class="muted">Ваш вибір не зберігся для цього питання.</p>` : ""}
      </div>

      <div class="stack" style="gap: 8px;">${options}</div>

      <div class="sticky-cta" id="review-actions"></div>
    </section>
  `;

  const actions = ctx.refs.mainPanel.querySelector("#review-actions");
  const navRow = document.createElement("div");
  navRow.className = "row";
  navRow.style.gap = "8px";

  if (view.actions.has_prev) {
    const btn = ctx.actionButton(
      "← Попереднє",
      async () => {
        ctx.state.currentView = await ctx.api("/api/test/review/index", {
          method: "POST",
          body: { index: view.index - 1 },
        });
        ctx.render();
      },
      "lg",
    );
    btn.style.flex = "1";
    navRow.append(btn);
  }
  if (view.actions.has_next) {
    const btn = ctx.actionButton(
      "Наступне →",
      async () => {
        ctx.state.currentView = await ctx.api("/api/test/review/index", {
          method: "POST",
          body: { index: view.index + 1 },
        });
        ctx.render();
      },
      "lg",
    );
    btn.classList.add("btn--primary");
    btn.style.flex = "1";
    navRow.append(btn);
  }
  actions.append(navRow);
  actions.append(
    ctx.actionButton(
      "До результату",
      async () => {
        ctx.state.currentView = await ctx.api("/api/test/review/back", { method: "POST" });
        ctx.render();
      },
      "block-ghost",
    ),
  );
}

/* ===================== ROUTER ===================== */
export function renderCurrentView(ctx) {
  const view = ctx.state.currentView;
  if (!view) {
    ctx.render();
    return;
  }

  if (view.mode === "pretest") return renderPretest(ctx, view);
  if (view.screen === "question") return renderQuestionView(ctx, view);
  if (view.screen === "feedback") return renderFeedbackView(ctx, view);
  if (view.screen === "result") return renderResultView(ctx, view);
  if (view.screen === "review") return renderReviewView(ctx, view);

  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <div class="screen-content">
      <div class="empty">
        <h2>Активний стан не знайдено</h2>
        <p>${ctx.escapeHtml(view.message || "Спробуйте повернутися на головну.")}</p>
      </div>
    </div>
  `;
}
