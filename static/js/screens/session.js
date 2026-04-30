function statusToClass(status) {
  if (status === "correct") {
    return "choice-review--correct";
  }
  if (status === "chosen") {
    return "choice-review--chosen";
  }
  return "";
}

function statusToLabel(status) {
  if (status === "correct") {
    return "Правильна відповідь";
  }
  if (status === "chosen") {
    return "Ваш вибір";
  }
  return "Інший варіант";
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return 0;
  }
  return Math.max(0, Math.min(100, number));
}

function progressPercent(progress = {}) {
  const current = Number(progress.current || 1);
  const total = Number(progress.total || 1);
  if (!total) {
    return 0;
  }
  return clampPercent(((current - 1) / total) * 100);
}

function modeLabel(mode) {
  if (mode === "test") {
    return "Тестування";
  }
  if (mode === "learn") {
    return "Навчання";
  }
  if (mode === "mistakes") {
    return "Помилки";
  }
  if (mode === "test_result") {
    return "Тест";
  }
  return "Сесія";
}

function questionMeta(question = {}) {
  return question.ok_label || question.topic || question.section || "Питання";
}

function renderPreviewQuestion(ctx, question, index, total) {
  const options = question.choices
    .map(
      (choice) => `
        <div class="choice-review ${choice.is_correct ? "choice-review--correct" : ""}">
          <span class="choice-review__index">${choice.index}</span>
          <div>
            <div>${ctx.escapeHtml(choice.text)}</div>
            ${choice.is_correct ? `<div class="muted">Правильна відповідь</div>` : ""}
          </div>
        </div>
      `,
    )
    .join("");

  return `
    <span class="eyebrow">Питання ${index}/${total}</span>
    <h3>${ctx.escapeHtml(question.question)}</h3>
    <p class="muted">${ctx.escapeHtml(questionMeta(question))}</p>
    <div class="review-grid">${options}</div>
  `;
}

function renderPretest(ctx, view) {
  const total = view.total || 0;
  const current = Number(view.selected_index || 0) + 1;
  ctx.setChrome({
    eyebrow: "Передстарт",
    title: "Перед стартом",
    subtitle: view.header || "Швидкий перегляд перед стартом.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen test-session-screen pretest-screen">
      <section class="test-hero-card test-hero-card--calm">
        <div class="test-hero-card__top">
          <span class="test-mode-chip">Preview</span>
          <span class="status-chip">${current}/${total}</span>
        </div>
        <div class="test-hero-card__copy">
          <h2>Перед стартом</h2>
          <p>${ctx.escapeHtml(view.header || "Перегляньте питання перед запуском навчання.")}</p>
        </div>
        <div class="button-row" id="pretest-actions"></div>
      </section>

      <section class="pretest-layout">
        <section class="ios-section test-index-panel">
          <div class="section-header">
            <div class="section-copy">
              <h2>Навігація</h2>
              <p>Оберіть номер питання для швидкого перегляду.</p>
            </div>
          </div>
          <div class="numbers-grid test-number-grid" id="pretest-grid"></div>
        </section>
        <section class="question-card test-question-card" id="pretest-preview"></section>
      </section>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#pretest-actions").append(
    ctx.actionButton("Почати навчання", async () => {
      try {
        ctx.state.currentView = await ctx.api("/api/pretest/start", { method: "POST" });
        ctx.impact("medium");
        ctx.render();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "primary"),
    ctx.actionButton("Закрити", ctx.leaveCurrentView),
  );

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

  ctx.refs.mainPanel.querySelector("#pretest-preview").innerHTML = renderPreviewQuestion(
    ctx,
    view.question,
    current,
    total,
  );
}

function renderQuestionView(ctx, view) {
  const question = view.question;
  const progress = view.progress || {};
  const percent = progressPercent(progress);
  const isSkippedPhase = progress.phase === "skipped";
  const choices = question.choices
    .map(
      (choice, index) => `
        <button class="choice-button test-choice-card" type="button" data-choice="${index}">
          <span class="choice-button__index">${ctx.escapeHtml(choice.index)}</span>
          <span class="test-choice-card__text">${ctx.escapeHtml(choice.text)}</span>
        </button>
      `,
    )
    .join("");

  ctx.setChrome({
    eyebrow: modeLabel(view.mode),
    title: view.header || modeLabel(view.mode),
    subtitle: `Питання ${progress.current}/${progress.total}`,
    showBack: true,
    showRefresh: false,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen test-session-screen active-question-screen">
      <section class="test-session-header">
        <div class="test-session-header__line">
          <span class="test-mode-chip">${ctx.escapeHtml(modeLabel(view.mode))}</span>
          <span class="status-chip">${progress.current}/${progress.total}</span>
        </div>
        <div class="test-progress-track" aria-label="Прогрес тесту">
          <span style="width: ${percent}%"></span>
        </div>
        <div class="test-session-header__meta">
          <span>${ctx.escapeHtml(questionMeta(question))}</span>
          ${isSkippedPhase ? `<strong>Повтор пропущених</strong>` : `<strong>Оберіть одну відповідь</strong>`}
        </div>
      </section>

      <section class="question-card test-question-card test-question-card--active">
        <span class="eyebrow">Питання ${progress.current}</span>
        <h3>${ctx.escapeHtml(question.question)}</h3>
        <div class="choice-grid test-choice-grid">${choices}</div>
      </section>

      <section class="test-bottom-actions">
        <div class="button-row" id="question-actions"></div>
      </section>
    </section>
  `;

  ctx.refs.mainPanel.querySelectorAll("[data-choice]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        button.classList.add("is-selected");
        ctx.impact("medium");
        ctx.state.currentView = await ctx.api("/api/session/answer", {
          method: "POST",
          body: { choice: Number(button.dataset.choice) },
        });
        ctx.render();
      } catch (error) {
        button.classList.remove("is-selected");
        ctx.setMessage("error", error.message);
      }
    });
  });

  const actions = ctx.refs.mainPanel.querySelector("#question-actions");
  if (view.actions.allow_skip) {
    actions.append(
      ctx.actionButton("Пропустити", async () => {
        try {
          ctx.state.currentView = await ctx.api("/api/session/skip", { method: "POST" });
          ctx.render();
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      }),
    );
  }
  actions.append(ctx.actionButton("Завершити", ctx.leaveCurrentView, "danger"));
}

function renderFeedbackView(ctx, view) {
  const correctOption = view.question.options.find((option) => option.status === "correct");
  const chosenOption = view.question.options.find((option) => option.status === "chosen");
  const isCorrect = !chosenOption || chosenOption.index === correctOption?.index;
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice-review test-choice-review ${statusToClass(option.status)}">
          <span class="choice-review__index">${ctx.escapeHtml(option.index)}</span>
          <div>
            <div>${ctx.escapeHtml(option.text)}</div>
            <div class="muted">${statusToLabel(option.status)}</div>
          </div>
        </div>
      `,
    )
    .join("");

  ctx.setChrome({
    eyebrow: "Розбір",
    title: "Розбір відповіді",
    subtitle: view.header || "Короткий розбір.",
    showBack: true,
    showRefresh: false,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen test-session-screen feedback-screen">
      <section class="test-hero-card ${isCorrect ? "test-hero-card--success" : "test-hero-card--danger"}">
        <div class="test-hero-card__top">
          <span class="test-mode-chip">Розбір</span>
          <span class="status-chip ${isCorrect ? "is-active" : "is-danger"}">${isCorrect ? "Правильно" : "Помилка"}</span>
        </div>
        <div class="test-hero-card__copy">
          <h2>${isCorrect ? "Відповідь зарахована" : "Правильна відповідь нижче"}</h2>
          <p>${ctx.escapeHtml(view.question.question)}</p>
        </div>
      </section>

      <section class="question-card test-question-card">
        <div class="review-grid test-review-grid">${options}</div>
      </section>

      <section class="test-bottom-actions">
        <div class="button-row" id="feedback-actions"></div>
      </section>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#feedback-actions").append(
    ctx.actionButton("Далі", async () => {
      try {
        ctx.state.currentView = await ctx.api("/api/session/next", { method: "POST" });
        ctx.impact("medium");
        ctx.render();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "primary"),
    ctx.actionButton("Завершити", ctx.leaveCurrentView, "danger"),
  );
}

function renderResultView(ctx, view) {
  const summary = view.summary || {};
  const blocks = summary.blocks || [];
  const percent = clampPercent(summary.percent);
  const passed = summary.passed !== false;
  const resultTone = passed ? "success" : "danger";

  ctx.setChrome({
    eyebrow: "Результат",
    title: summary.title || "Результат",
    subtitle: "Підсумок сесії.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen test-session-screen result-screen">
      <section class="test-result-hero test-result-hero--${resultTone}">
        <div class="score-ring" style="--score: ${percent};">
          <span>${percent.toFixed(0)}%</span>
        </div>
        <div class="test-result-hero__copy">
          <span class="test-mode-chip">Result</span>
          <h2>${ctx.escapeHtml(summary.title || "Результат")}</h2>
          <p>${typeof summary.correct === "number" ? `Правильно ${summary.correct} із ${summary.total}` : "Сесію завершено."}</p>
          ${typeof summary.passed === "boolean" ? `<strong class="result-badge ${passed ? "is-passed" : "is-failed"}">${passed ? "Поріг складено" : "Поріг не складено"}</strong>` : ""}
        </div>
      </section>

      <section class="metrics-grid test-metrics-grid">
        ${typeof summary.correct === "number" ? ctx.metricCard("Правильно", `${summary.correct}/${summary.total}`, "відповідей") : ""}
        ${typeof summary.percent === "number" ? ctx.metricCard("Точність", `${summary.percent.toFixed(1)}%`, "результат") : ""}
        ${typeof summary.remaining === "number" ? ctx.metricCard("У помилках", String(summary.remaining), "залишилось") : ""}
        ${typeof view.wrong_count === "number" ? ctx.metricCard("Помилки", String(view.wrong_count), "для розбору") : ""}
      </section>

      ${
        blocks.length
          ? `
            <section class="ios-section test-blocks-section">
              <div class="section-header">
                <div class="section-copy">
                  <h2>По блоках</h2>
                  <p>Де тест пройдено сильніше або слабше.</p>
                </div>
              </div>
              <div class="list-stack test-block-list">
                ${blocks
                  .map((block) => {
                    const blockPercent = block.total ? clampPercent((block.correct / block.total) * 100) : 0;
                    return `
                      <article class="test-block-row">
                        <div class="test-block-row__top">
                          <strong>${ctx.escapeHtml(block.name)}</strong>
                          <span>${block.correct}/${block.total}</span>
                        </div>
                        <div class="test-progress-track test-progress-track--small">
                          <span style="width: ${blockPercent}%"></span>
                        </div>
                      </article>
                    `;
                  })
                  .join("")}
              </div>
            </section>
          `
          : ""
      }

      <section class="test-bottom-actions">
        <div class="button-row" id="result-actions"></div>
      </section>
    </section>
  `;

  const actions = ctx.refs.mainPanel.querySelector("#result-actions");
  if (view.mode === "test_result" && view.wrong_count > 0) {
    actions.append(
      ctx.actionButton("Розібрати помилки", async () => {
        try {
          ctx.state.currentView = await ctx.api("/api/test/review/open", { method: "POST" });
          ctx.render();
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      }, "primary"),
    );
  }
  actions.append(
    ctx.actionButton("На головну", async () => {
      ctx.state.currentView = null;
      ctx.goHome();
      await ctx.loadBootstrap();
    }),
  );
}

function renderReviewView(ctx, view) {
  const progress = view.total ? clampPercent(((view.index + 1) / view.total) * 100) : 0;
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice-review test-choice-review ${statusToClass(option.status)}">
          <span class="choice-review__index">${ctx.escapeHtml(option.index)}</span>
          <div>
            <div>${ctx.escapeHtml(option.text)}</div>
            <div class="muted">${statusToLabel(option.status)}</div>
          </div>
        </div>
      `,
    )
    .join("");

  ctx.setChrome({
    eyebrow: "Помилки",
    title: "Помилки тесту",
    subtitle: `Питання ${view.index + 1}/${view.total}`,
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen test-session-screen review-screen">
      <section class="test-session-header">
        <div class="test-session-header__line">
          <span class="test-mode-chip">Розбір помилок</span>
          <span class="status-chip">${view.index + 1}/${view.total}</span>
        </div>
        <div class="test-progress-track">
          <span style="width: ${progress}%"></span>
        </div>
        <div class="test-session-header__meta">
          <span>Перегляд неправильних відповідей</span>
          <strong>Знайдіть слабке місце</strong>
        </div>
      </section>

      <section class="question-card test-question-card">
        <span class="eyebrow">Помилка ${view.index + 1}</span>
        <h3>${ctx.escapeHtml(view.question.question)}</h3>
        <div class="review-grid test-review-grid">${options}</div>
        ${view.question.selected_missing ? `<p class="muted">Ваш вибір не зберігся для цього питання.</p>` : ""}
      </section>

      <section class="test-bottom-actions">
        <div class="button-row" id="review-actions"></div>
      </section>
    </section>
  `;

  const actions = ctx.refs.mainPanel.querySelector("#review-actions");
  if (view.actions.has_prev) {
    actions.append(
      ctx.actionButton("Назад", async () => {
        ctx.state.currentView = await ctx.api("/api/test/review/index", { method: "POST", body: { index: view.index - 1 } });
        ctx.render();
      }),
    );
  }
  if (view.actions.has_next) {
    actions.append(
      ctx.actionButton("Наступна", async () => {
        ctx.state.currentView = await ctx.api("/api/test/review/index", { method: "POST", body: { index: view.index + 1 } });
        ctx.render();
      }, "primary"),
    );
  }
  actions.append(
    ctx.actionButton("До результату", async () => {
      ctx.state.currentView = await ctx.api("/api/test/review/back", { method: "POST" });
      ctx.render();
    }),
  );
}

export function renderCurrentView(ctx) {
  const view = ctx.state.currentView;
  if (!view) {
    ctx.render();
    return;
  }

  if (view.mode === "pretest") {
    renderPretest(ctx, view);
    return;
  }

  if (view.screen === "question") {
    renderQuestionView(ctx, view);
    return;
  }

  if (view.screen === "feedback") {
    renderFeedbackView(ctx, view);
    return;
  }

  if (view.screen === "result") {
    renderResultView(ctx, view);
    return;
  }

  if (view.screen === "review") {
    renderReviewView(ctx, view);
    return;
  }

  ctx.setChrome({
    eyebrow: "Сесія",
    title: "Активний стан",
    subtitle: "Не вдалося визначити активний сценарій.",
    showBack: true,
  });
  ctx.refs.mainPanel.innerHTML = `
    <div class="empty-state">
      <h2>Активний стан не знайдено</h2>
      <p>${ctx.escapeHtml(view.message || "Спробуйте повернутися на головну.")}</p>
    </div>
  `;
}
