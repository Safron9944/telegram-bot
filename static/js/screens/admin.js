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
            iconName: "users",
            tint: "blue",
            screen: "admin-users",
          }),
          ctx.cell({
            title: "Банк питань",
            subtitle: "Пошук і редагування",
            iconName: "edit",
            tint: "purple",
            screen: "admin-questions",
          }),
          ctx.cell({
            title: "Розділи атестації",
            subtitle: "Видимість, порядок і видалення",
            iconName: "document",
            tint: "purple",
            screen: "admin-attestation-banks",
          }),
          ctx.cell({
            title: "Кейси",
            subtitle: "Імпорт Keys.db",
            iconName: "folder",
            tint: "green",
            screen: "admin-cases",
          }),
          ctx.cell({
            title: "Тестові питання",
            subtitle: "База питань тестування",
            iconName: "document",
            tint: "orange",
            screen: "admin-test-questions",
          }),
          ctx.cell({
            title: "Налаштування",
            subtitle: "Головний екран і тарифи",
            iconName: "settings",
            tint: "teal",
            screen: "admin-settings",
          }),
          ctx.cell({
            title: "Пошук питань",
            subtitle: "По всіх банках одразу",
            iconName: "search",
            tint: "gray",
            screen: "admin-global-search",
          }),
        ].join(""),
      })}
    </section>
  `;

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

/* ===================== ADMIN ATTESTATION BANKS ===================== */
export function renderAdminAttestationBanks(ctx) {
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Розділи атестації</h1>
      <p class="page-subtitle">Перший етап захищений. Нові розділи можна впорядковувати, приховувати й видаляти.</p>
      <div class="group">
        <div class="group__list" id="admin-attestation-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
      </div>
    </section>
  `;
}

export async function loadAdminAttestationBanks(ctx) {
  if (ctx.state.currentScreen !== "admin-attestation-banks") return;
  const list = document.querySelector("#admin-attestation-list");
  try {
    const payload = await ctx.api("/api/admin/attestation-banks");
    if (!list || ctx.state.currentScreen !== "admin-attestation-banks") return;
    list.innerHTML = "";
    payload.items.forEach((bank, index) => {
      const row = document.createElement("div");
      row.className = "admin-attestation-row";
      row.innerHTML = `
        <div class="admin-attestation-row__info">
          <strong>${ctx.escapeHtml(bank.title)}</strong>
          <span>${ctx.escapeHtml(bank.questions_count || 0)} питань · ${bank.system ? "системний розділ" : `позиція ${index}`}</span>
        </div>
        <div class="admin-attestation-row__actions"></div>
      `;
      const actions = row.querySelector(".admin-attestation-row__actions");
      if (bank.system) {
        actions.innerHTML = '<span class="apk-badge">Захищено</span>';
      } else {
        const visibility = document.createElement("label");
        visibility.className = "admin-attestation-visibility";
        visibility.innerHTML = `<input type="checkbox" ${bank.visible ? "checked" : ""}><span>Показувати</span>`;
        visibility.querySelector("input").addEventListener("change", async (event) => {
          try {
            await ctx.api(`/api/admin/attestation-banks/${bank.id}/visibility`, {
              method: "PATCH", body: { visible: event.currentTarget.checked },
            });
            await loadAdminAttestationBanks(ctx);
          } catch (error) { event.currentTarget.checked = !event.currentTarget.checked; ctx.setMessage("error", error.message); }
        });
        const move = (label, direction) => {
          const button = ctx.actionButton(label, async () => {
            try {
              await ctx.api(`/api/admin/attestation-banks/${bank.id}/move`, { method: "POST", body: { direction } });
              await loadAdminAttestationBanks(ctx);
            } catch (error) { ctx.setMessage("error", error.message); }
          }, "sm");
          return button;
        };
        const up = move("↑ Вище", "up");
        up.dataset.direction = "up"; // direction: "up"
        const down = move("↓ Нижче", "down");
        down.dataset.direction = "down"; // direction: "down"
        const remove = ctx.actionButton("Видалити", async () => {
          if (!window.confirm(`Видалити «${bank.title}» разом із питаннями?`)) return;
          try {
            await ctx.api(`/api/admin/attestation-banks/${bank.id}`, { method: "DELETE" });
            await loadAdminAttestationBanks(ctx);
          } catch (error) { ctx.setMessage("error", error.message); }
        }, "sm");
        remove.classList.add("btn--danger");
        actions.append(visibility, up, down, remove);
      }
      list.append(row);
    });
  } catch (error) {
    if (list) list.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
  }
}

function adminUserStatus(access, isAdmin = false) {
  if (isAdmin) return { label: "Адмін", tone: "active", tint: "blue" };
  if (access?.state === "trial") return { label: "Тріал", tone: "trial", tint: "orange" };
  if (access?.has_access) return { label: "Активний", tone: "active", tint: "green" };
  return { label: "Без доступу", tone: "none", tint: "gray" };
}

function formatAdminDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("uk-UA", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
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
        const status = adminUserStatus(item.access, item.is_admin);

        const row = document.createElement("button");
        row.type = "button";
        row.className = "cell cell--admin-user";
        row.innerHTML = `
          <span class="cell__icon cell__icon--${status.tint}">${ctx.escapeHtml((item.display_name || "U").slice(0, 1).toUpperCase())}</span>
          <span class="cell__body">
            <span class="cell__title">${ctx.escapeHtml(item.display_name)}</span>
            <span class="cell__subtitle">ID ${item.user_id} · ${ctx.escapeHtml(item.access.label)}</span>
          </span>
          <span class="admin-user-status admin-user-status--${status.tone}">${status.label}</span>
          <span class="cell__chevron" aria-hidden="true"></span>
        `;
        if (isSelected) row.style.background = "var(--bg-fill-soft)";
        row.addEventListener("click", () => {
          ctx.state.selectedAdminUserId = item.user_id;
          ctx.state.adminUserDetail = null;
          ctx.navigate("admin-user-detail");
        });
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

  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export function renderAdminUserDetail(ctx) {
  ctx.setChrome({ showBack: true });
  const payload = ctx.state.adminUserDetail;

  if (!payload) {
    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <h1 class="page-title">Користувач</h1>
        <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
      </section>
    `;
    return;
  }

  const name = [payload.first_name, payload.last_name].filter(Boolean).join(" ") || "Без імені";
  const initials = name === "Без імені"
    ? "U"
    : name.split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
  const status = adminUserStatus(payload.access, payload.is_admin);
  const stats = payload.stats || { count: 0, avg: 0, last: null };
  const protectedAccount = payload.is_admin || payload.user_id === ctx.state.bootstrap.user.id;

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content admin-user-detail">
      <h1 class="page-title">Користувач</h1>

      <section class="admin-user-profile">
        <div class="admin-user-profile__avatar">${ctx.escapeHtml(initials)}</div>
        <div class="admin-user-profile__body">
          <div class="admin-user-profile__name">${ctx.escapeHtml(name)}</div>
          <div class="admin-user-profile__id">Telegram ID ${payload.user_id}</div>
          <span class="admin-user-status admin-user-status--${status.tone}">${status.label}</span>
        </div>
      </section>

      <div class="group">
        <div class="group__label">Обліковий запис</div>
        <div class="group__list">
          <div class="cell admin-user-detail__row">
            <span class="cell__body">
              <span class="cell__title">Поточний доступ</span>
              <span class="cell__subtitle">${ctx.escapeHtml(payload.access.label)}</span>
            </span>
          </div>
          <div class="cell admin-user-detail__row">
            <span class="cell__body">
              <span class="cell__title">Дата реєстрації</span>
              <span class="cell__subtitle">${ctx.escapeHtml(formatAdminDate(payload.created_at))}</span>
            </span>
          </div>
          <div class="cell admin-user-detail__row">
            <span class="cell__body">
              <span class="cell__title">Обрані компетенції</span>
              <span class="cell__subtitle">${payload.ok_modules?.length || 0} модулів</span>
            </span>
          </div>
        </div>
      </div>

      <div class="group">
        <div class="group__label">Результати</div>
        <div class="stat-strip admin-user-detail__stats">
          ${ctx.statPill("Тестів", String(stats.count || 0))}
          ${ctx.statPill("Середній", `${Number(stats.avg || 0).toFixed(0)}%`)}
          ${ctx.statPill("Останній", stats.last ? `${stats.last.correct}/${stats.last.total}` : "—")}
        </div>
      </div>

      <section class="admin-access-controls">
        <div class="admin-access-controls__header">
          <span class="admin-access-controls__title">Керування доступом</span>
          <span class="admin-access-controls__hint">Зміни застосовуються одразу</span>
        </div>
        <div id="admin-user-actions" class="admin-access-actions"></div>
      </section>

      <section class="admin-danger-zone">
        <div class="admin-danger-zone__title">Видалення користувача</div>
        <p>Буде видалено профіль, підписку, прогрес, помилки, результати тестів і збережену сесію.</p>
        <div id="admin-user-delete-wrap"></div>
        ${protectedAccount ? '<div class="admin-danger-zone__note">Адміністраторські облікові записи захищені від видалення.</div>' : ""}
      </section>
    </section>
  `;

  const updateAccess = async (access, message) => {
    try {
      ctx.state.adminUserDetail = await ctx.api(`/api/admin/users/${payload.user_id}/access`, {
        method: "POST",
        body: { access },
      });
      ctx.impact("medium");
      ctx.setMessage("success", message);
      ctx.render();
    } catch (error) {
      ctx.setMessage("error", error.message);
    }
  };

  const actions = ctx.refs.mainPanel.querySelector("#admin-user-actions");
  const trialButton = ctx.actionButton("⏳ Тріал на 3 дні", () => updateAccess("trial", "Тріал активовано на 3 дні."), "block");
  trialButton.classList.add("btn--admin-trial");
  const casesButton = ctx.actionButton("⭐ Кейси й атестація", () => updateAccess("cases", "Доступ до кейсів і атестації активовано."), "block");
  casesButton.classList.add("btn--admin-cases");
  const fullButton = ctx.actionButton("✓ Повний доступ", () => updateAccess("full", "Повний доступ активовано."), "block");
  const removeButton = ctx.actionButton("✕ Забрати доступ", async () => {
    if (!window.confirm("Забрати у користувача тріал і всі види доступу?")) return;
    await updateAccess("none", "Доступ скасовано.");
  }, "block");
  removeButton.classList.add("btn--admin-remove");
  actions?.append(trialButton, casesButton, fullButton, removeButton);

  const deleteWrap = ctx.refs.mainPanel.querySelector("#admin-user-delete-wrap");
  if (deleteWrap && !protectedAccount) {
    const deleteButton = ctx.actionButton("Видалити користувача", async () => {
      const confirmed = window.confirm(
        `Видалити користувача «${name}» (ID ${payload.user_id}) разом з усіма його даними? Цю дію неможливо скасувати.`,
      );
      if (!confirmed) return;
      try {
        await ctx.api(`/api/admin/users/${payload.user_id}`, { method: "DELETE" });
        ctx.state.selectedAdminUserId = null;
        ctx.state.adminUserDetail = null;
        ctx.state.adminUsersOffset = 0;
        ctx.impact("medium");
        ctx.setMessage("success", "Користувача та його дані видалено.");
        await ctx.goBack();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "block");
    deleteButton.classList.add("btn--danger");
    deleteWrap.append(deleteButton);
  }
}

export async function loadAdminUserDetail(ctx, userId = ctx.state.selectedAdminUserId) {
  if (ctx.state.currentScreen !== "admin-user-detail" || !userId) return;
  try {
    const payload = await ctx.api(`/api/admin/users/${userId}`);
    if (ctx.state.currentScreen !== "admin-user-detail") return;
    ctx.state.selectedAdminUserId = userId;
    ctx.state.adminUserDetail = payload;
    ctx.render();
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
    row.addEventListener("click", () => {
      ctx.state.selectedQuestionId = item.id;
      ctx.navigate("admin-question-detail");
    });
    list.append(row);
  });
}

export function renderAdminQuestionDetail(ctx) {
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
    </section>
  `;
  if (ctx.state.selectedQuestionId) {
    void loadQuestionDetail(ctx, ctx.state.selectedQuestionId);
  }
}

export async function loadQuestionDetail(ctx, questionId) {
  if (ctx.state.currentScreen !== "admin-question-detail") return;

  try {
    ctx.state.selectedQuestionId = questionId;
    const payload = await ctx.api(`/api/admin/questions/${questionId}`);
    if (ctx.state.currentScreen !== "admin-question-detail") return;

    const question = payload.question;

    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <h1 class="page-title">Питання #${question.id}</h1>
        <p class="page-subtitle">${ctx.escapeHtml(question.ok_label || question.topic || question.section || "Без групи")}</p>
        <div class="group">
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
      </section>
    `;

    const panel = ctx.refs.mainPanel;
    const choicesEditor = panel.querySelector("#choices-editor");
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

    panel.querySelector("#reload-question").addEventListener("click", async () => {
      await loadQuestionDetail(ctx, questionId);
    });

    panel.querySelector("#question-edit-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const updatedChoices = [];
      const correct = [];
      question.choices.forEach((choice) => {
        updatedChoices.push(panel.querySelector(`#choice-${choice.index}`).value.trim());
        if (panel.querySelector(`#correct-${choice.index}`).checked) {
          correct.push(choice.index);
        }
      });

      try {
        const updated = await ctx.api(`/api/admin/questions/${questionId}`, {
          method: "PATCH",
          body: {
            question: panel.querySelector("#question-text").value.trim(),
            choices: updatedChoices,
            correct,
          },
        });
        ctx.setMessage("success", "Питання збережено.");
        ctx.impact("medium");
        await loadQuestionDetail(ctx, updated.question.id);
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    });
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

/* ===================== ADMIN CASES ===================== */
export function renderAdminCases(ctx) {
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Кейси</h1>
      <p class="page-subtitle">Завантажте Keys.db — бот сам витягне номер кейсу, питання і правильні відповіді.</p>

      <div class="group">
        <div class="group__label">Імпорт Keys.db</div>
        <div class="group__list admin-upload-box">
          <input class="input" id="case-db-file" type="file" accept=".db,.zip" multiple />
          <div id="case-upload-action"></div>
          <div class="group__footer">Можна вибрати кілька Keys.db або ZIP-архів. Кожен .db зберігається як окремий кейс із питаннями.</div>
        </div>
      </div>

      <div class="group">
        <div class="group__label">Завантажені кейси</div>
        <div class="group__list" id="admin-cases-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
      </div>
    </section>
  `;

  const action = ctx.refs.mainPanel.querySelector("#case-upload-action");
  action.append(
    ctx.actionButton(
      "Завантажити кейси",
      async () => {
        const input = ctx.refs.mainPanel.querySelector("#case-db-file");
        const files = Array.from(input?.files || []);
        if (!files.length) {
          ctx.setMessage("error", "Спочатку виберіть Keys.db або ZIP-архів.");
          return;
        }
        const form = new FormData();
        files.forEach((file) => form.append("files", file));
        try {
          const response = await ctx.api("/api/admin/cases/import-batch", {
            method: "POST",
            body: form,
          });
          const imported = response.imported_count || 0;
          const failed = response.failed_count || 0;
          const questions = response.questions_count || 0;
          const suffix = failed ? ` Не імпортовано: ${failed}.` : "";
          ctx.setMessage(
            imported ? "success" : "error",
            `Імпортовано кейсів: ${imported}, питань: ${questions}.${suffix}`,
          );
          ctx.impact("medium");
          input.value = "";
          await loadAdminCases(ctx);
          await ctx.loadBootstrap(false);
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      },
      "block",
    ),
  );
}

export async function loadAdminCases(ctx) {
  if (ctx.state.currentScreen !== "admin-cases") return;
  try {
    const payload = await ctx.api("/api/cases");
    const list = document.querySelector("#admin-cases-list");
    if (!list) return;
    const items = payload.items || [];
    if (!items.length) {
      list.innerHTML = `
        <div class="empty empty--inline"><h2>Кейсів ще немає</h2><p>Завантажте перший Keys.db.</p></div>
      `;
      return;
    }
    list.innerHTML = "";
    items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "cell";
      row.style.cursor = "default";
      row.innerHTML = `
        <span class="cell__icon cell__icon--green">${ctx.escapeHtml((item.case_number || "К").slice(0, 2))}</span>
        <span class="cell__body">
          <span class="cell__title">Кейс ${ctx.escapeHtml(item.case_number || "—")}</span>
          <span class="cell__subtitle">${ctx.escapeHtml(item.questions_count)} питань · ${ctx.escapeHtml(item.correct_count)} правильних</span>
        </span>
        <span class="row-actions"></span>
      `;
      const actions = row.querySelector(".row-actions");
      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "pill";
      openBtn.textContent = "Відкрити";
      openBtn.addEventListener("click", () => {
        ctx.state.selectedCase = item;
        ctx.state.caseOffset = 0;
        ctx.state.caseQuery = "";
        ctx.navigate("case-detail");
      });
      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "pill pill--danger";
      delBtn.textContent = "Видалити";
      delBtn.addEventListener("click", async () => {
        if (!confirm(`Видалити кейс ${item.case_number}?`)) return;
        try {
          await ctx.api(`/api/admin/cases/${item.id}`, { method: "DELETE" });
          ctx.setMessage("success", "Кейс видалено.");
          await loadAdminCases(ctx);
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      });
      actions.append(openBtn, delBtn);
      list.append(row);
    });
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

/* ===================== ADMIN SETTINGS ===================== */
export function renderAdminSettings(ctx) {
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Налаштування</h1>
      <p class="page-subtitle">Головний екран, тарифи та загальні параметри застосунку.</p>

      <div class="group">
        <div class="group__label">Головний екран для всіх користувачів</div>
        <div class="group__list" id="home-visibility-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
        <div class="group__footer">Приховування прибирає пункт із головного екрана, але не видаляє дані розділу.</div>
      </div>

      <div class="group" style="margin-top: 16px;">
        <div class="group__label">Тарифи</div>
        <div class="group__list" style="padding: 16px; display: flex; flex-direction: column; gap: 12px;">
          <div>
            <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: var(--text-secondary);">Тільки кейси (⭐)</label>
            <input id="price-cases" class="input" type="number" min="1" placeholder="100" style="width: 100%;" />
          </div>
          <div>
            <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: var(--text-secondary);">Повний доступ (⭐)</label>
            <input id="price-full" class="input" type="number" min="1" placeholder="250" style="width: 100%;" />
          </div>
          <div id="settings-save-wrap"></div>
        </div>
      </div>

      <div class="group" style="margin-top: 16px;">
        <div class="group__label">Контакт адміністратора</div>
        <div class="group__list" style="padding: 16px; display: flex; flex-direction: column; gap: 12px;">
          <div>
            <label style="display: block; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: var(--text-secondary);">Username або посилання</label>
            <input id="admin-contact-url" class="input" type="text" placeholder="@username або https://t.me/username" style="width: 100%;" />
          </div>
          <div id="admin-contact-save-wrap"></div>
        </div>
        <div class="group__footer">Посилання відображається у кнопці "Адміністратор" у розділі Підтримка.</div>
      </div>

      <div class="group" style="margin-top: 16px;">
        <div class="group__label">Тестові питання</div>
        <div class="group__list">
          <div class="cell" style="cursor: default;">
            <span class="cell__body">
              <span class="cell__title">Видимість для підписників</span>
              <span class="cell__subtitle">Показати вкладку користувачам з повним доступом</span>
            </span>
            <label class="switch" id="tq-switch-label">
              <input id="test-questions-visible" type="checkbox" />
              <span class="switch__track"></span>
            </label>
          </div>
        </div>
        <div class="group__footer">Увімкніть після того, як заповните всі тестові питання.</div>
      </div>
    </section>
  `;
}

export async function loadAdminSettings(ctx) {
  if (ctx.state.currentScreen !== "admin-settings") return;
  try {
    const payload = await ctx.api("/api/admin/settings");
    const casesInput = document.querySelector("#price-cases");
    const fullInput = document.querySelector("#price-full");
    const tqCheckbox = document.querySelector("#test-questions-visible");
    const homeVisibilityList = document.querySelector("#home-visibility-list");

    if (homeVisibilityList) {
      const definitions = [
        ["attestation", "Атестація посадових осіб", "Основна картка першого етапу атестації"],
        ["customs", "Митні компетенції", "Основна картка навчання і тестування"],
        ["cases", "Кейси", "Пункт у блоці матеріалів"],
        ["customs_code", "Митний кодекс", "Пункт у блоці матеріалів"],
        ["question_search", "Пошук питань", "Відображається користувачам із повним доступом"],
        ["support", "Підтримка", "Контакт адміністратора і довідкова інформація"],
      ];
      const current = { ...(payload.home_visibility || {}) };
      homeVisibilityList.innerHTML = definitions.map(([key, title, subtitle]) => `
        <div class="cell admin-setting-row">
          <span class="cell__body">
            <span class="cell__title">${ctx.escapeHtml(title)}</span>
            <span class="cell__subtitle">${ctx.escapeHtml(subtitle)}</span>
          </span>
          <label class="switch">
            <input type="checkbox" data-home-visibility="${key}" ${current[key] !== false ? "checked" : ""} />
            <span class="switch__track"></span>
          </label>
        </div>
      `).join("");

      homeVisibilityList.querySelectorAll("[data-home-visibility]").forEach((checkbox) => {
        checkbox.addEventListener("change", async () => {
          const key = checkbox.dataset.homeVisibility;
          const visible = checkbox.checked;
          checkbox.disabled = true;
          try {
            await ctx.api("/api/admin/settings", {
              method: "POST",
              body: { home_visibility: { [key]: visible } },
            });
            current[key] = visible;
            ctx.state.bootstrap.home_visibility = { ...ctx.state.bootstrap.home_visibility, [key]: visible };
            ctx.impact("medium");
            ctx.setMessage("success", visible ? "Пункт показано на головному екрані." : "Пункт приховано з головного екрана.");
          } catch (error) {
            checkbox.checked = !visible;
            ctx.setMessage("error", error.message);
          } finally {
            checkbox.disabled = false;
          }
        });
      });
    }

    if (casesInput) casesInput.value = String(payload.price_cases);
    if (fullInput) fullInput.value = String(payload.price_full);
    if (tqCheckbox) {
      tqCheckbox.checked = Boolean(payload.test_questions_visible);
      tqCheckbox.addEventListener("change", async () => {
        const visible = tqCheckbox.checked;
        tqCheckbox.disabled = true;
        try {
          await ctx.api("/api/admin/settings", {
            method: "POST",
            body: { test_questions_visible: visible },
          });
          ctx.impact("medium");
          ctx.setMessage("success", visible ? "Тестові питання видимі для підписників." : "Тестові питання приховані.");
          await ctx.loadBootstrap();
        } catch (error) {
          tqCheckbox.checked = !visible;
          ctx.setMessage("error", error.message);
        } finally {
          tqCheckbox.disabled = false;
        }
      });
    }

    const contactInput = document.querySelector("#admin-contact-url");
    if (contactInput) {
      contactInput.value = payload.admin_contact_url || "";
      const contactWrap = document.querySelector("#admin-contact-save-wrap");
      if (contactWrap) {
        contactWrap.innerHTML = "";
        contactWrap.append(
          ctx.actionButton("Зберегти контакт", async () => {
            const val = contactInput.value.trim();
            try {
              await ctx.api("/api/admin/settings", { method: "POST", body: { admin_contact_url: val } });
              ctx.impact("medium");
              ctx.setMessage("success", "Контакт збережено.");
              await ctx.loadBootstrap();
            } catch (error) {
              ctx.setMessage("error", error.message);
            }
          }, "block"),
        );
      }
    }

    const wrap = document.querySelector("#settings-save-wrap");
    if (wrap) {
      wrap.innerHTML = "";
      wrap.append(
        ctx.actionButton(
          "Зберегти ціни",
          async () => {
            const cases = parseInt(document.querySelector("#price-cases")?.value, 10);
            const full = parseInt(document.querySelector("#price-full")?.value, 10);
            if (!cases || cases < 1 || !full || full < 1) {
              ctx.setMessage("error", "Введіть коректні значення (ціле число ≥ 1).");
              return;
            }
            try {
              await ctx.api("/api/admin/settings", {
                method: "POST",
                body: { price_cases: cases, price_full: full },
              });
              ctx.impact("medium");
              ctx.setMessage("success", "Ціни збережено.");
              await ctx.loadBootstrap();
            } catch (error) {
              ctx.setMessage("error", error.message);
            }
          },
          "block",
        ),
      );
    }
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

/* ===================== ADMIN TEST EXAM QUESTIONS ===================== */
let testQSearchTimer = 0;
let testQRequestId = 0;

export function renderAdminTestQuestions(ctx) {
  ctx.state.testQSearchQuery = "";
  ctx.state.testQOffset = 0;
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Тестові питання</h1>
      <p class="page-subtitle">Питання та відповіді підсумкового тестування.</p>

      <div class="case-search">
        <span class="case-search__icon" aria-hidden="true"></span>
        <input class="case-search__input" id="test-q-input" type="search"
               placeholder="Пошук по питанню або відповіді" />
      </div>

      <section class="case-questions">
        <h2 class="case-questions__title">Питання та правильні відповіді</h2>
        <div class="case-answer-list" id="test-q-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
      </section>

      <div class="row" id="test-q-pagination" style="justify-content:center; gap:8px; margin-top:12px;"></div>
    </section>
  `;

  const input = ctx.refs.mainPanel.querySelector("#test-q-input");
  const run = () => {
    ctx.state.testQSearchQuery = input.value.trim();
    ctx.state.testQOffset = 0;
    void loadAdminTestQuestions(ctx, 0);
  };
  const runLive = () => {
    window.clearTimeout(testQSearchTimer);
    testQSearchTimer = window.setTimeout(run, 350);
  };
  input?.addEventListener("input", runLive);
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      window.clearTimeout(testQSearchTimer);
      run();
    }
  });
}

export async function loadAdminTestQuestions(ctx, offset = ctx.state.testQOffset || 0) {
  if (ctx.state.currentScreen !== "admin-test-questions") return;
  const query = ctx.state.testQSearchQuery || "";
  const requestId = ++testQRequestId;
  const list = document.querySelector("#test-q-list");
  const pagination = document.querySelector("#test-q-pagination");
  if (list) list.innerHTML = `<div class="empty empty--inline"><h2>Шукаємо…</h2></div>`;

  try {
    const payload = await ctx.api(
      `/api/admin/test-exam-questions?q=${encodeURIComponent(query)}&offset=${offset}&limit=20`,
    );
    if (requestId !== testQRequestId || ctx.state.currentScreen !== "admin-test-questions") return;
    ctx.state.testQOffset = offset;
    if (!list) return;

    if (!payload.items?.length) {
      list.innerHTML = query
        ? `<div class="empty empty--inline"><h2>Нічого не знайдено</h2><p>Спробуйте інший запит.</p></div>`
        : `<div class="empty empty--inline"><h2>Питань ще немає</h2></div>`;
    } else {
      list.innerHTML = "";
      payload.items.forEach((item) => {
        const block = document.createElement("article");
        block.className = "case-answer";
        block.innerHTML = `
          <div class="case-answer__head">
            <span class="case-answer__number">${ctx.escapeHtml(item.num || "")}</span>
            ${item.module ? `<span class="case-answer__count">${ctx.escapeHtml(item.module)}</span>` : `<span class="case-answer__count">${ctx.escapeHtml(item.source || "")}</span>`}
          </div>
          <h2 class="case-answer__question">${ctx.escapeHtml(item.question)}</h2>
          <div class="case-answer__label">Правильна відповідь</div>
          <div class="case-answer__correct">
            <span class="case-answer__check" aria-hidden="true">✓</span>
            <div class="case-answer__correct-body">
              <div class="case-answer__correct-text">${ctx.escapeHtml(item.correct_answer || "—")}</div>
            </div>
          </div>
        `;
        list.append(block);
      });
    }

    if (pagination) {
      pagination.innerHTML = "";
      if (payload.has_prev) {
        pagination.append(ctx.actionButton("← Назад", () => void loadAdminTestQuestions(ctx, Math.max(0, offset - payload.limit)), "sm"));
      }
      if (payload.has_next) {
        pagination.append(ctx.actionButton("Далі →", () => void loadAdminTestQuestions(ctx, offset + payload.limit), "sm"));
      }
    }
  } catch (error) {
    if (list) list.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
  }
}

/* ===================== ADMIN GLOBAL SEARCH ===================== */
export function renderAdminGlobalSearch(ctx) {
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Пошук питань</h1>

      <div class="field">
        <input id="global-search-input" class="input" type="search"
               placeholder="Почніть вводити текст…" autocomplete="off" autofocus />
      </div>
      <div id="global-search-results">
        <div class="empty empty--inline">
          <h2>Введіть запит</h2>
          <p>Шукає одночасно в ОК-модулях, кейсах і тестових питаннях.</p>
        </div>
      </div>
    </section>
  `;

  const panel = ctx.refs.mainPanel;
  const results = panel.querySelector("#global-search-results");
  let debounceTimer = null;
  let currentQuery = "";

  const runSearch = async (q) => {
    currentQuery = q;
    try {
      const data = await ctx.api(`/api/admin/global-search?q=${encodeURIComponent(q)}&limit=15`);
      if (q !== currentQuery) return;
      renderGlobalSearchResults(ctx, data, results);
    } catch (error) {
      if (q !== currentQuery) return;
      results.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
    }
  };

  panel.querySelector("#global-search-input").addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    const q = e.target.value.trim();
    if (q.length < 3) {
      currentQuery = "";
      results.innerHTML = `<div class="empty empty--inline"><h2>Введіть запит</h2><p>Шукає в ОК-модулях, кейсах і тестових питаннях.</p></div>`;
      return;
    }
    results.innerHTML = `<div class="empty empty--inline"><h2>Шукаємо…</h2></div>`;
    debounceTimer = setTimeout(() => void runSearch(q), 350);
  });
}

function renderGlobalSearchResults(ctx, data, container) {
  container.innerHTML = "";

  const total = data.ok.length + data.cases.length + data.test.length;
  if (!total) {
    container.innerHTML = `<div class="empty empty--inline"><h2>Нічого не знайдено</h2><p>Спробуйте інший текст.</p></div>`;
    return;
  }

  const makeRow = (iconName, tint, title, subtitle, onClick) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell";
    row.innerHTML = `
      <span class="cell__icon cell__icon--${tint}">${ctx.lineIcon(iconName)}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(title)}</span>
        <span class="cell__subtitle">${ctx.escapeHtml(subtitle)}</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", onClick);
    return row;
  };

  const addSection = (label, items, buildRow) => {
    const section = document.createElement("div");
    section.className = "group";
    const listId = `gs-${Math.random().toString(36).slice(2)}`;
    section.innerHTML = `<div class="group__label">${label} (${items.length})</div><div class="group__list" id="${listId}"></div>`;
    const list = section.querySelector(`#${listId}`);
    items.forEach((item) => list.append(buildRow(item)));
    container.append(section);
  };

  if (data.ok.length) {
    addSection("ОК-модулі", data.ok, (item) =>
      makeRow("edit", "purple", item.question, item.ok || item.topic || "Без модуля", () => {
        ctx.state.adminQuestionViewItem = { type: "ok", id: item.id, source: item.ok || item.topic || "" };
        ctx.navigate("admin-question-view");
      })
    );
  }

  if (data.cases.length) {
    addSection("Кейси", data.cases, (item) =>
      makeRow("folder", "green", item.question, `Кейс ${item.case_number}`, () => {
        ctx.state.adminQuestionViewItem = { type: "case", ...item };
        ctx.navigate("admin-question-view");
      })
    );
  }

  if (data.test.length) {
    addSection("Тестові питання", data.test, (item) =>
      makeRow("document", "orange", item.question, item.num || item.module || "", () => {
        ctx.state.adminQuestionViewItem = { type: "test", ...item };
        ctx.navigate("admin-question-view");
      })
    );
  }
}

/* ===================== ADMIN QUESTION VIEW ===================== */
export function renderAdminQuestionView(ctx) {
  ctx.setChrome({ showBack: true });
  const item = ctx.state.adminQuestionViewItem;
  if (!item) { ctx.goBack(); return; }

  if (item.type === "ok") {
    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
      </section>
    `;
    void ctx.api(`/api/admin/questions/${item.id}`).then((payload) => {
      const q = payload.question;
      const panel = ctx.refs.mainPanel;
      panel.innerHTML = `
        <section class="screen-content">
          <p class="page-subtitle" style="margin-bottom: 4px;">${ctx.escapeHtml(item.source)}</p>
          <h1 class="page-title">${ctx.escapeHtml(q.question)}</h1>
          <div class="group" id="qv-choices"></div>
        </section>
      `;
      const choicesEl = panel.querySelector("#qv-choices");
      const list = document.createElement("div");
      list.className = "group__list";
      q.choices.forEach((choice) => {
        const row = document.createElement("div");
        row.className = "cell" + (choice.is_correct ? " cell--accent" : "");
        row.style.cursor = "default";
        row.innerHTML = `
          <span class="cell__icon cell__icon--${choice.is_correct ? "green" : "gray"}">${choice.is_correct ? "✓" : choice.index}</span>
          <span class="cell__body"><span class="cell__title">${ctx.escapeHtml(choice.text)}</span></span>
        `;
        list.append(row);
      });
      choicesEl.append(list);
    }).catch((err) => {
      ctx.refs.mainPanel.innerHTML = `<section class="screen-content"><div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(err.message)}</p></div></section>`;
    });
    return;
  }

  if (item.type === "case") {
    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <p class="page-subtitle" style="margin-bottom: 4px;">Кейс ${ctx.escapeHtml(String(item.case_number))}</p>
        <h1 class="page-title">${ctx.escapeHtml(item.question)}</h1>
        <div class="group">
          <div class="group__label">Варіанти відповіді</div>
          <div class="group__list" id="qv-answers"></div>
        </div>
      </section>
    `;
    const list = ctx.refs.mainPanel.querySelector("#qv-answers");
    const answers = Array.isArray(item.answers) ? item.answers : [];
    if (answers.length) {
      answers.forEach((text, idx) => {
        const isCorrect = text === item.correct_answer;
        const row = document.createElement("div");
        row.className = "cell" + (isCorrect ? " cell--accent" : "");
        row.style.cursor = "default";
        row.innerHTML = `
          <span class="cell__icon cell__icon--${isCorrect ? "green" : "gray"}">${isCorrect ? "✓" : idx + 1}</span>
          <span class="cell__body"><span class="cell__title">${ctx.escapeHtml(text)}</span></span>
        `;
        list.append(row);
      });
    } else {
      list.innerHTML = `<div class="cell" style="cursor:default;"><span class="cell__body"><span class="cell__title">${ctx.escapeHtml(item.correct_answer || "—")}</span></span></div>`;
    }
    return;
  }

  if (item.type === "test") {
    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <p class="page-subtitle" style="margin-bottom: 4px;">${ctx.escapeHtml(item.num ? item.num + (item.module ? " · " + item.module : "") : item.module || "")}</p>
        <h1 class="page-title">${ctx.escapeHtml(item.question)}</h1>
        <div class="group">
          <div class="group__label">Правильна відповідь</div>
          <div class="group__list">
            <div class="cell cell--accent" style="cursor:default;">
              <span class="cell__icon cell__icon--green">✓</span>
              <span class="cell__body"><span class="cell__title">${ctx.escapeHtml(item.correct_answer || "—")}</span></span>
            </div>
          </div>
        </div>
        ${item.justification ? `
        <div class="group">
          <div class="group__label">Обґрунтування</div>
          <div class="group__list">
            <div style="padding: 14px; font-size: 14px; line-height: 1.5;">${ctx.escapeHtml(item.justification)}</div>
          </div>
        </div>` : ""}
      </section>
    `;
  }
}
