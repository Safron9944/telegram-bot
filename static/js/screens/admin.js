/* ===================== ADMIN HUB ===================== */
export function renderAdminHub(ctx) {
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Адмін</h1>
      <p class="page-subtitle">Сервісні інструменти.</p>

      ${ctx.group({
        children: [
          ctx.cell({
            title: "Користувачі",
            subtitle: "Доступ і статус",
            icon: "👥",
            tint: "blue",
            screen: "admin-users",
          }),
          ctx.cell({
            title: "Банк питань",
            subtitle: "Пошук і редагування",
            icon: "✎",
            tint: "purple",
            screen: "admin-questions",
          }),
        ].join(""),
      })}
    </section>
  `;

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

/* ===================== ADMIN USERS ===================== */
export function renderAdminUsers(ctx) {
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Користувачі</h1>

      <div id="admin-users-summary"></div>

      <div class="group">
        <div class="group__list" id="admin-users-list">
          <div class="empty empty--inline">
            <h2>Завантажуємо…</h2>
          </div>
        </div>
      </div>

      <div class="row" id="admin-users-pagination" style="justify-content: center; gap: 8px;"></div>

      <div id="admin-user-detail"></div>
    </section>
  `;
}

export async function loadAdminUsers(ctx, offset = 0) {
  if (ctx.state.currentScreen !== "admin-users") return;

  try {
    const payload = await ctx.api(`/api/admin/users?offset=${offset}&limit=10`);
    if (ctx.state.currentScreen !== "admin-users") return;

    ctx.state.adminUsersOffset = payload.offset;

    const summary = document.querySelector("#admin-users-summary");
    if (summary) {
      summary.innerHTML = `
        <div class="stat-strip">
          ${ctx.statPill("Активні", String(payload.counts.active))}
          ${ctx.statPill("Тріал", String(payload.counts.trial))}
          ${ctx.statPill("Без доступу", String(payload.counts.expired))}
        </div>
      `;
    }

    const list = document.querySelector("#admin-users-list");
    if (!list) return;

    if (!payload.items.length) {
      list.innerHTML = `
        <div class="empty empty--inline">
          <h2>Порожньо</h2>
          <p>У цьому діапазоні немає користувачів.</p>
        </div>
      `;
    } else {
      list.innerHTML = "";
      payload.items.forEach((item) => {
        const isSelected = ctx.state.selectedAdminUserId === item.user_id;
        const tint = item.access.label.toLowerCase().includes("актив")
          ? "green"
          : item.access.label.toLowerCase().includes("тріал")
            ? "orange"
            : "gray";

        const row = document.createElement("button");
        row.type = "button";
        row.className = "cell";
        row.innerHTML = `
          <span class="cell__icon cell__icon--${tint}">${ctx.escapeHtml((item.display_name || "U").slice(0, 1).toUpperCase())}</span>
          <span class="cell__body">
            <span class="cell__title">${ctx.escapeHtml(item.display_name)}</span>
            <span class="cell__subtitle">ID ${item.user_id} · ${ctx.escapeHtml(item.access.label)}</span>
          </span>
          <span class="cell__chevron" aria-hidden="true"></span>
        `;
        if (isSelected) row.style.background = "var(--bg-fill-soft)";
        row.addEventListener("click", () => loadAdminUserDetail(ctx, item.user_id));
        list.append(row);
      });
    }

    const pagination = document.querySelector("#admin-users-pagination");
    if (pagination) {
      pagination.innerHTML = "";
      if (payload.has_prev) {
        pagination.append(
          ctx.actionButton(
            "← Назад",
            async () => loadAdminUsers(ctx, Math.max(0, payload.offset - payload.limit)),
            "sm",
          ),
        );
      }
      if (payload.has_next) {
        pagination.append(
          ctx.actionButton(
            "Далі →",
            async () => loadAdminUsers(ctx, payload.offset + payload.limit),
            "sm",
          ),
        );
      }
    }

    if (ctx.state.selectedAdminUserId) {
      void loadAdminUserDetail(ctx, ctx.state.selectedAdminUserId);
    }
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export async function loadAdminUserDetail(ctx, userId) {
  if (ctx.state.currentScreen !== "admin-users") return;

  try {
    ctx.state.selectedAdminUserId = userId;
    const payload = await ctx.api(`/api/admin/users/${userId}`);
    if (ctx.state.currentScreen !== "admin-users") return;

    const root = document.querySelector("#admin-user-detail");
    if (!root) return;

    const name = [payload.first_name, payload.last_name].filter(Boolean).join(" ") || "—";
    const isInfinite = payload.access.state === "sub_infinite";

    root.innerHTML = `
      <div style="height: 6px"></div>
      <div class="group">
        <div class="group__label">Деталі</div>
        <div class="group__list">
          <div class="cell" style="cursor: default;">
            <span class="cell__icon cell__icon--blue">i</span>
            <span class="cell__body">
              <span class="cell__title">${ctx.escapeHtml(name)}</span>
              <span class="cell__subtitle">ID ${payload.user_id} · ${ctx.escapeHtml(payload.access.label)}</span>
            </span>
          </div>
          <div class="cell" style="cursor: default;">
            <span class="cell__icon cell__icon--gray">📅</span>
            <span class="cell__body">
              <span class="cell__title">Створено</span>
              <span class="cell__subtitle">${ctx.escapeHtml(payload.created_at || "—")}</span>
            </span>
          </div>
        </div>
      </div>
      <div class="sticky-cta" id="admin-user-actions"></div>
    `;

    const actions = root.querySelector("#admin-user-actions");
    actions.append(
      ctx.actionButton(
        isInfinite ? "Скасувати безстроковий доступ" : "Дати безстроковий доступ",
        async () => {
          try {
            await ctx.api(`/api/admin/users/${userId}/subscription`, {
              method: "POST",
              body: { infinite: !isInfinite },
            });
            ctx.impact("medium");
            ctx.setMessage("success", "Доступ оновлено.");
            await loadAdminUsers(ctx, ctx.state.adminUsersOffset);
            await loadAdminUserDetail(ctx, userId);
          } catch (error) {
            ctx.setMessage("error", error.message);
          }
        },
        "block",
      ),
    );
    actions.append(
      ctx.actionButton(
        "Забрати доступ",
        async () => {
          try {
            await ctx.api(`/api/admin/users/${userId}/subscription`, {
              method: "POST",
              body: { infinite: false },
            });
            ctx.setMessage("success", "Доступ оновлено.");
            await loadAdminUsers(ctx, ctx.state.adminUsersOffset);
            await loadAdminUserDetail(ctx, userId);
          } catch (error) {
            ctx.setMessage("error", error.message);
          }
        },
        "block-ghost",
      ),
    );
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

/* ===================== ADMIN QUESTIONS ===================== */
export function renderAdminQuestions(ctx) {
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Банк питань</h1>

      <div class="field">
        <input id="question-search-input" class="input" type="text"
               value="${ctx.escapeHtml(ctx.state.questionSearchQuery)}"
               placeholder="Пошук за текстом (від 3 символів)" />
      </div>

      <div class="row" id="question-search-actions" style="gap: 8px;"></div>

      <div class="group">
        <div class="group__list" id="question-list">
          <div class="empty empty--inline">
            <h2>Завантажуємо…</h2>
          </div>
        </div>
      </div>

      <div class="row" id="question-pagination" style="justify-content: center; gap: 8px;"></div>

      <div id="admin-question-editor"></div>
    </section>
  `;
}

export async function loadAdminQuestions(ctx, page = 0) {
  if (ctx.state.currentScreen !== "admin-questions") return;

  try {
    const payload = await ctx.api(`/api/admin/questions?page=${page}&page_size=10`);
    if (ctx.state.currentScreen !== "admin-questions") return;

    ctx.state.adminQuestionsPage = payload.page;
    if (!ctx.state.questionSearchQuery) {
      ctx.state.searchResults = null;
    }

    // Search bar wiring
    const searchActions = document.querySelector("#question-search-actions");
    if (searchActions) {
      searchActions.innerHTML = "";
      searchActions.append(
        ctx.actionButton(
          "Шукати",
          async () => {
            const query = document.querySelector("#question-search-input").value.trim();
            await runQuestionSearch(ctx, query);
          },
          "primary",
        ),
      );
      if (ctx.state.questionSearchQuery) {
        searchActions.append(
          ctx.actionButton(
            "Скинути",
            async () => {
              ctx.state.questionSearchQuery = "";
              ctx.state.searchResults = null;
              const input = document.querySelector("#question-search-input");
              if (input) input.value = "";
              await loadAdminQuestions(ctx, 0);
            },
          ),
        );
      }
    }

    document.querySelector("#question-search-input")?.addEventListener("keydown", async (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        const query = event.currentTarget.value.trim();
        await runQuestionSearch(ctx, query);
      }
    });

    renderQuestionList(ctx, ctx.state.searchResults || payload.items);

    const pagination = document.querySelector("#question-pagination");
    if (pagination) {
      pagination.innerHTML = "";
      if (!ctx.state.questionSearchQuery) {
        if (payload.page > 0) {
          pagination.append(
            ctx.actionButton("← Назад", async () => loadAdminQuestions(ctx, payload.page - 1), "sm"),
          );
        }
        if (payload.page + 1 < payload.pages) {
          pagination.append(
            ctx.actionButton("Далі →", async () => loadAdminQuestions(ctx, payload.page + 1), "sm"),
          );
        }
      }
    }

    if (ctx.state.selectedQuestionId) {
      void loadQuestionDetail(ctx, ctx.state.selectedQuestionId);
    }
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export async function runQuestionSearch(ctx, query) {
  if (!query || query.length < 3) {
    ctx.setMessage("error", "Введіть щонайменше 3 символи для пошуку.");
    return;
  }

  try {
    const result = await ctx.api(`/api/admin/questions/search?q=${encodeURIComponent(query)}`);
    if (ctx.state.currentScreen !== "admin-questions") return;

    ctx.state.questionSearchQuery = query;
    ctx.state.searchResults = result.items;
    renderQuestionList(ctx, result.items);
    document.querySelector("#question-pagination").innerHTML = "";
    ctx.impact("light");

    // re-render search bar so "Скинути" appears
    const searchActions = document.querySelector("#question-search-actions");
    if (searchActions) {
      searchActions.innerHTML = "";
      searchActions.append(
        ctx.actionButton(
          "Шукати",
          async () => {
            const q = document.querySelector("#question-search-input").value.trim();
            await runQuestionSearch(ctx, q);
          },
          "primary",
        ),
        ctx.actionButton(
          "Скинути",
          async () => {
            ctx.state.questionSearchQuery = "";
            ctx.state.searchResults = null;
            const input = document.querySelector("#question-search-input");
            if (input) input.value = "";
            await loadAdminQuestions(ctx, 0);
          },
        ),
      );
    }
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

function renderQuestionList(ctx, items) {
  const list = document.querySelector("#question-list");
  if (!list) return;

  list.innerHTML = "";
  if (!items.length) {
    list.innerHTML = `
      <div class="empty empty--inline">
        <h2>Нічого не знайдено</h2>
        <p>Спробуйте інший фрагмент або поверніться до пагінованого списку.</p>
      </div>
    `;
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell";
    row.innerHTML = `
      <span class="cell__icon cell__icon--purple">#${item.id}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(item.question)}</span>
        <span class="cell__subtitle">${ctx.escapeHtml(item.ok || item.topic || "Без модуля")}</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", () => loadQuestionDetail(ctx, item.id));
    list.append(row);
  });
}

export async function loadQuestionDetail(ctx, questionId) {
  if (ctx.state.currentScreen !== "admin-questions") return;

  try {
    ctx.state.selectedQuestionId = questionId;
    const payload = await ctx.api(`/api/admin/questions/${questionId}`);
    if (ctx.state.currentScreen !== "admin-questions") return;

    const question = payload.question;
    const root = document.querySelector("#admin-question-editor");
    if (!root) return;

    root.innerHTML = `
      <div style="height: 6px"></div>
      <div class="group">
        <div class="group__label">Питання #${question.id} — ${ctx.escapeHtml(question.ok_label || question.topic || question.section || "Без групи")}</div>
        <div class="group__list" style="padding: 14px;">
          <form id="question-edit-form" class="stack" style="gap: 12px;">
            <div class="field">
              <label class="field__label" for="question-text">Текст питання</label>
              <textarea id="question-text" class="textarea">${ctx.escapeHtml(question.question)}</textarea>
            </div>
            <div id="choices-editor" class="stack"></div>
            <div class="row" style="gap: 8px; margin-top: 4px;">
              <button class="btn btn--primary btn--lg" type="submit" style="flex: 1;">Зберегти</button>
              <button class="btn btn--lg" type="button" id="reload-question">Скинути</button>
            </div>
          </form>
        </div>
      </div>
    `;

    const choicesEditor = document.querySelector("#choices-editor");
    question.choices.forEach((choice) => {
      const block = document.createElement("div");
      block.className = "stack";
      block.style.gap = "6px";
      block.style.padding = "10px";
      block.style.borderRadius = "10px";
      block.style.background = "var(--bg-fill-soft)";
      block.innerHTML = `
        <div class="field">
          <label class="field__label" for="choice-${choice.index}">Варіант ${choice.index}</label>
          <textarea id="choice-${choice.index}" class="textarea" style="min-height: 60px;">${ctx.escapeHtml(choice.text)}</textarea>
        </div>
        <label class="row" style="gap: 10px; cursor: pointer;">
          <span class="switch">
            <input id="correct-${choice.index}" type="checkbox" ${choice.is_correct ? "checked" : ""} />
            <span class="switch__track"></span>
          </span>
          <span style="font-size: 14px; font-weight: 500;">Правильна відповідь</span>
        </label>
      `;
      choicesEditor.append(block);
    });

    document.querySelector("#reload-question").addEventListener("click", async () => {
      await loadQuestionDetail(ctx, questionId);
    });

    document.querySelector("#question-edit-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const updatedChoices = [];
      const correct = [];
      question.choices.forEach((choice) => {
        updatedChoices.push(document.querySelector(`#choice-${choice.index}`).value.trim());
        if (document.querySelector(`#correct-${choice.index}`).checked) {
          correct.push(choice.index);
        }
      });

      try {
        const updated = await ctx.api(`/api/admin/questions/${questionId}`, {
          method: "PATCH",
          body: {
            question: document.querySelector("#question-text").value.trim(),
            choices: updatedChoices,
            correct,
          },
        });
        ctx.setMessage("success", "Питання збережено.");
        ctx.impact("medium");
        await loadQuestionDetail(ctx, updated.question.id);

        if (ctx.state.questionSearchQuery) {
          await runQuestionSearch(ctx, ctx.state.questionSearchQuery);
        } else {
          await loadAdminQuestions(ctx, ctx.state.adminQuestionsPage);
        }
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    });
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}
