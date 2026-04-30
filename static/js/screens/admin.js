export function renderAdminHub(ctx) {
  ctx.setChrome({
    eyebrow: "Адмін режим",
    title: "Адміністрування",
    subtitle: "Сервісні інструменти відокремлені від основного користувацького потоку.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Інструменти адміністратора</h2>
          <p>Оберіть потрібний напрям: користувачі або банк питань.</p>
        </div>
      </div>
      <div class="dashboard-grid">
        ${ctx.actionCard({
          code: "US",
          title: "Користувачі",
          body: "Безстроковий доступ, статус і перевірка поточних підписок.",
          meta: "Відкрити",
          screen: "admin-users",
        })}
        ${ctx.actionCard({
          code: "QA",
          title: "Питання",
          body: "Пошук і редагування текстів, варіантів та правильних відповідей.",
          meta: "Відкрити",
          screen: "admin-questions",
        })}
      </div>
    </section>
  `;

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

export function renderAdminUsers(ctx) {
  ctx.setChrome({
    eyebrow: "Адмін режим",
    title: "Користувачі",
    subtitle: "Керуйте доступом і переглядайте поточний стан користувачів.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="split-layout">
      <section class="surface" id="admin-users-list">
        <div class="empty-state">
          <h2>Завантажуємо список</h2>
          <p>Поточні користувачі з’являться тут за мить.</p>
        </div>
      </section>
      <section class="surface" id="admin-user-detail">
        <div class="empty-state">
          <h2>Оберіть користувача</h2>
          <p>Деталі і кнопки керування з’являться після вибору запису зі списку.</p>
        </div>
      </section>
    </section>
  `;
}

export function renderAdminQuestions(ctx) {
  ctx.setChrome({
    eyebrow: "Адмін режим",
    title: "Банк питань",
    subtitle: "Пошук і редагування в одному режимі без зайвої навігації.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="split-layout">
      <section class="surface" id="admin-question-browser">
        <div class="empty-state">
          <h2>Завантажуємо питання</h2>
          <p>Список і пошук з’являться тут за мить.</p>
        </div>
      </section>
      <section class="surface" id="admin-question-editor">
        <div class="empty-state">
          <h2>Оберіть питання</h2>
          <p>Форма редагування з’явиться після вибору зі списку або з результатів пошуку.</p>
        </div>
      </section>
    </section>
  `;
}

export async function loadAdminUsers(ctx, offset = 0) {
  if (ctx.state.currentScreen !== "admin-users") {
    return;
  }

  try {
    const payload = await ctx.api(`/api/admin/users?offset=${offset}&limit=10`);
    if (ctx.state.currentScreen !== "admin-users") {
      return;
    }

    ctx.state.adminUsersOffset = payload.offset;
    const listNode = document.querySelector("#admin-users-list");
    if (!listNode) {
      return;
    }

    listNode.innerHTML = `
      <div class="section-header">
        <div class="section-copy">
          <h2>Список користувачів</h2>
          <p>На цій сторінці: активні ${payload.counts.active}, тріал ${payload.counts.trial}, без доступу ${payload.counts.expired}.</p>
        </div>
      </div>
      <div class="list-stack" id="admin-users-items"></div>
      <div class="button-row" id="admin-users-pagination"></div>
    `;

    const items = document.querySelector("#admin-users-items");
    if (!payload.items.length) {
      items.innerHTML = `
        <div class="empty-state">
          <h2>Порожньо</h2>
          <p>У списку немає користувачів для цього діапазону.</p>
        </div>
      `;
    } else {
      payload.items.forEach((item) => {
        const row = document.createElement("article");
        row.className = "list-item";
        row.innerHTML = `
          <div class="list-item__main">
            <strong>${ctx.escapeHtml(item.display_name)}</strong>
            <span class="list-item__meta">ID ${item.user_id} • ${ctx.escapeHtml(item.access.label)}</span>
          </div>
        `;
        row.append(ctx.actionButton("Відкрити", async () => loadAdminUserDetail(ctx, item.user_id), "primary"));
        items.append(row);
      });
    }

    const pagination = document.querySelector("#admin-users-pagination");
    if (payload.has_prev) {
      pagination.append(ctx.actionButton("Попередні", async () => loadAdminUsers(ctx, Math.max(0, payload.offset - payload.limit))));
    }
    if (payload.has_next) {
      pagination.append(ctx.actionButton("Наступні", async () => loadAdminUsers(ctx, payload.offset + payload.limit)));
    }

    if (ctx.state.selectedAdminUserId) {
      void loadAdminUserDetail(ctx, ctx.state.selectedAdminUserId);
    }
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export async function loadAdminUserDetail(ctx, userId) {
  if (ctx.state.currentScreen !== "admin-users") {
    return;
  }

  try {
    ctx.state.selectedAdminUserId = userId;
    const payload = await ctx.api(`/api/admin/users/${userId}`);
    if (ctx.state.currentScreen !== "admin-users") {
      return;
    }

    const detailNode = document.querySelector("#admin-user-detail");
    if (!detailNode) {
      return;
    }

    detailNode.innerHTML = `
      <div class="section-header">
        <div class="section-copy">
          <h2>Користувач ${payload.user_id}</h2>
          <p>${ctx.escapeHtml(payload.access.label)}</p>
        </div>
      </div>
      <div class="list-stack">
        <article class="list-item">
          <div class="list-item__main">
            <strong>${ctx.escapeHtml([payload.first_name, payload.last_name].filter(Boolean).join(" ") || "—")}</strong>
            <span class="list-item__meta">Створено: ${ctx.escapeHtml(payload.created_at || "—")}</span>
          </div>
        </article>
      </div>
      <div class="button-row" id="admin-user-detail-actions"></div>
    `;

    const actions = document.querySelector("#admin-user-detail-actions");
    actions.append(
      ctx.actionButton(
        payload.access.state === "sub_infinite" ? "Скасувати безстроковий доступ" : "Дати безстроковий доступ",
        async () => {
          try {
            await ctx.api(`/api/admin/users/${userId}/subscription`, {
              method: "POST",
              body: { infinite: payload.access.state !== "sub_infinite" },
            });
            ctx.impact("medium");
            ctx.setMessage("success", "Доступ користувача оновлено.");
            await loadAdminUsers(ctx, ctx.state.adminUsersOffset);
            await loadAdminUserDetail(ctx, userId);
          } catch (error) {
            ctx.setMessage("error", error.message);
          }
        },
        "primary",
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
            ctx.setMessage("success", "Доступ користувача оновлено.");
            await loadAdminUsers(ctx, ctx.state.adminUsersOffset);
            await loadAdminUserDetail(ctx, userId);
          } catch (error) {
            ctx.setMessage("error", error.message);
          }
        },
        "danger",
      ),
    );
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export async function loadAdminQuestions(ctx, page = 0) {
  if (ctx.state.currentScreen !== "admin-questions") {
    return;
  }

  try {
    const payload = await ctx.api(`/api/admin/questions?page=${page}&page_size=10`);
    if (ctx.state.currentScreen !== "admin-questions") {
      return;
    }

    ctx.state.adminQuestionsPage = payload.page;
    if (!ctx.state.questionSearchQuery) {
      ctx.state.searchResults = null;
    }

    const browser = document.querySelector("#admin-question-browser");
    if (!browser) {
      return;
    }

    browser.innerHTML = `
      <div class="field">
        <label for="question-search-input">Пошук за текстом питання</label>
        <input id="question-search-input" type="text" value="${ctx.escapeHtml(ctx.state.questionSearchQuery)}" placeholder="Введіть щонайменше 3 символи" />
      </div>
      <div class="button-row" id="question-search-actions"></div>
      <div class="list-stack" id="question-list"></div>
      <div class="button-row" id="question-pagination"></div>
    `;

    const searchActions = document.querySelector("#question-search-actions");
    searchActions.append(
      ctx.actionButton("Шукати", async () => {
        const query = document.querySelector("#question-search-input").value.trim();
        await runQuestionSearch(ctx, query);
      }, "primary"),
    );

    if (ctx.state.questionSearchQuery) {
      searchActions.append(
        ctx.actionButton("Скинути пошук", async () => {
          ctx.state.questionSearchQuery = "";
          ctx.state.searchResults = null;
          await loadAdminQuestions(ctx, ctx.state.adminQuestionsPage);
        }),
      );
    }

    document.querySelector("#question-search-input").addEventListener("keydown", async (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        const query = event.currentTarget.value.trim();
        await runQuestionSearch(ctx, query);
      }
    });

    renderQuestionList(ctx, ctx.state.searchResults || payload.items);

    const pagination = document.querySelector("#question-pagination");
    if (!ctx.state.questionSearchQuery) {
      if (payload.page > 0) {
        pagination.append(ctx.actionButton("Попередні", async () => loadAdminQuestions(ctx, payload.page - 1)));
      }
      if (payload.page + 1 < payload.pages) {
        pagination.append(ctx.actionButton("Наступні", async () => loadAdminQuestions(ctx, payload.page + 1)));
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
    if (ctx.state.currentScreen !== "admin-questions") {
      return;
    }

    ctx.state.questionSearchQuery = query;
    ctx.state.searchResults = result.items;
    renderQuestionList(ctx, result.items);
    document.querySelector("#question-pagination").innerHTML = "";
    ctx.impact("light");
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

function renderQuestionList(ctx, items) {
  const list = document.querySelector("#question-list");
  if (!list) {
    return;
  }

  list.innerHTML = "";
  if (!items.length) {
    list.innerHTML = `
      <div class="empty-state">
        <h2>Нічого не знайдено</h2>
        <p>Спробуйте інший фрагмент тексту або поверніться до пагінованого списку.</p>
      </div>
    `;
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = "list-item";
    row.innerHTML = `
      <div class="list-item__main">
        <span class="list-item__eyebrow">Питання #${item.id}</span>
        <strong>${ctx.escapeHtml(item.question)}</strong>
        <span class="list-item__meta">${ctx.escapeHtml(item.ok || item.topic || "Без модуля")}</span>
      </div>
    `;
    row.append(ctx.actionButton("Редагувати", async () => loadQuestionDetail(ctx, item.id), "primary"));
    list.append(row);
  });
}

export async function loadQuestionDetail(ctx, questionId) {
  if (ctx.state.currentScreen !== "admin-questions") {
    return;
  }

  try {
    ctx.state.selectedQuestionId = questionId;
    const payload = await ctx.api(`/api/admin/questions/${questionId}`);
    if (ctx.state.currentScreen !== "admin-questions") {
      return;
    }

    const question = payload.question;
    const editor = document.querySelector("#admin-question-editor");
    if (!editor) {
      return;
    }

    editor.innerHTML = `
      <div class="section-header">
        <div class="section-copy">
          <h2>Питання #${question.id}</h2>
          <p>${ctx.escapeHtml(question.ok_label || question.topic || question.section || "Без групи")}</p>
        </div>
      </div>
      <form id="question-edit-form" class="section-stack">
        <div class="field">
          <label for="question-text">Текст питання</label>
          <textarea id="question-text">${ctx.escapeHtml(question.question)}</textarea>
        </div>
        <div id="choices-editor" class="list-stack"></div>
        <div class="button-row">
          <button class="primary-button" type="submit">Зберегти</button>
          <button class="secondary-button" type="button" id="reload-question">Оновити форму</button>
        </div>
      </form>
    `;

    const choicesEditor = document.querySelector("#choices-editor");
    question.choices.forEach((choice) => {
      const block = document.createElement("article");
      block.className = "inline-form";
      block.innerHTML = `
        <div class="field">
          <label for="choice-${choice.index}">Варіант ${choice.index}</label>
          <textarea id="choice-${choice.index}">${ctx.escapeHtml(choice.text)}</textarea>
        </div>
        <div class="check-row">
          <input id="correct-${choice.index}" type="checkbox" ${choice.is_correct ? "checked" : ""} />
          <label for="correct-${choice.index}">Правильна відповідь</label>
        </div>
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
