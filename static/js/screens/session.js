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
    <p class="muted">${ctx.escapeHtml(question.ok_label || question.topic || question.section || "")}</p>
    <div class="review-grid">${options}</div>
  `;
}

function renderPretest(ctx, view) {
  const total = view.total || 0;
  ctx.setChrome({
    eyebrow: "Передстарт",
    title: "Перед стартом",
    subtitle: view.header || "Швидкий перегляд перед стартом.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Перед стартом</h2>
          <p>Оберіть номер для перегляду.</p>
        </div>
      </div>
      <div class="button-row" id="pretest-actions"></div>
    </section>
    <section class="split-layout">
      <section class="surface">
        <div class="section-header">
          <div class="section-copy">
            <h2>Питання</h2>
            <p>Усього: ${total}.</p>
          </div>
        </div>
        <div class="numbers-grid" id="pretest-grid"></div>
      </section>
      <section class="question-card" id="pretest-preview"></section>
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
    view.selected_index + 1,
    total,
  );
}

function renderQuestionView(ctx, view) {
  const question = view.question;
  const choices = question.choices
    .map(
      (choice, index) => `
        <button class="choice-button" type="button" data-choice="${index}">
          <span class="choice-button__index">${choice.index}</span>
          <span>${ctx.escapeHtml(choice.text)}</span>
        </button>
      `,
    )
    .join("");

  ctx.setChrome({
    eyebrow: "Активна сесія",
    title: view.header || "Сесія",
    subtitle: `Питання ${view.progress.current}/${view.progress.total}${view.progress.phase === "skipped" ? " • повтор пропущених" : ""}`,
    showBack: true,
    showRefresh: false,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Питання</h2>
          <p>Оберіть відповідь.</p>
        </div>
      </div>
      <div class="button-row" id="question-actions"></div>
    </section>
    <section class="question-card">
      <h3>${ctx.escapeHtml(question.question)}</h3>
      <div class="choice-grid">${choices}</div>
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
  actions.append(ctx.actionButton("Вийти", ctx.leaveCurrentView, "danger"));
}

function renderFeedbackView(ctx, view) {
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice-review ${statusToClass(option.status)}">
          <span class="choice-review__index">${option.index}</span>
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
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Розбір</h2>
          <p>Перевірте відповідь і рухайтесь далі.</p>
        </div>
      </div>
      <div class="button-row" id="feedback-actions"></div>
    </section>
    <section class="question-card">
      <h3>${ctx.escapeHtml(view.question.question)}</h3>
      <div class="review-grid">${options}</div>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#feedback-actions").append(
    ctx.actionButton("Продовжити", async () => {
      try {
        ctx.state.currentView = await ctx.api("/api/session/next", { method: "POST" });
        ctx.impact("medium");
        ctx.render();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "primary"),
    ctx.actionButton("Вийти", ctx.leaveCurrentView, "danger"),
  );
}

function renderResultView(ctx, view) {
  const summary = view.summary || {};
  const blocks = summary.blocks || [];

  ctx.setChrome({
    eyebrow: "Результат",
    title: summary.title || "Результат",
    subtitle: "Підсумок сесії.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>${ctx.escapeHtml(summary.title || "Результат")}</h2>
          <p>Коротке зведення та дії.</p>
        </div>
      </div>
      <div class="button-row" id="result-actions"></div>
    </section>
    <section class="metrics-grid">
      ${typeof summary.correct === "number" ? ctx.metricCard("Правильно", `${summary.correct}/${summary.total}`) : ""}
      ${typeof summary.percent === "number" ? ctx.metricCard("Відсоток", `${summary.percent.toFixed(1)}%`) : ""}
      ${typeof summary.remaining === "number" ? ctx.metricCard("У помилках", String(summary.remaining)) : ""}
      ${typeof summary.passed === "boolean" ? ctx.metricCard("Поріг 60%", summary.passed ? "Складено" : "Не складено") : ""}
    </section>
    ${
      blocks.length
        ? `
          <section class="surface">
            <div class="section-header">
              <div class="section-copy">
                <h2>По блоках</h2>
                <p>Результати по блоках.</p>
              </div>
            </div>
            <div class="list-stack">
              ${blocks
                .map(
                  (block) => `
                    <article class="list-item">
                      <div class="list-item__main">
                        <strong>${ctx.escapeHtml(block.name)}</strong>
                        <span class="list-item__meta">${block.correct} із ${block.total}</span>
                      </div>
                    </article>
                  `,
                )
                .join("")}
            </div>
          </section>
        `
        : ""
    }
  `;

  const actions = ctx.refs.mainPanel.querySelector("#result-actions");
  if (view.mode === "test_result" && view.wrong_count > 0) {
    actions.append(
      ctx.actionButton("Показати помилки", async () => {
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
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice-review ${statusToClass(option.status)}">
          <span class="choice-review__index">${option.index}</span>
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
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Помилки</h2>
          <p>Гортайте питання і повертайтесь до результату.</p>
        </div>
      </div>
      <div class="button-row" id="review-actions"></div>
    </section>
    <section class="question-card">
      <h3>${ctx.escapeHtml(view.question.question)}</h3>
      <div class="review-grid">${options}</div>
      ${view.question.selected_missing ? `<p class="muted">Ваш вибір не зберігся для цього питання.</p>` : ""}
    </section>
  `;

  const actions = ctx.refs.mainPanel.querySelector("#review-actions");
  if (view.actions.has_prev) {
    actions.append(
      ctx.actionButton("Попереднє", async () => {
        ctx.state.currentView = await ctx.api("/api/test/review/index", { method: "POST", body: { index: view.index - 1 } });
        ctx.render();
      }),
    );
  }
  if (view.actions.has_next) {
    actions.append(
      ctx.actionButton("Наступне", async () => {
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
