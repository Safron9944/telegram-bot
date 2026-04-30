const state = {
  bootstrap: null,
  currentTab: "home",
  currentView: null,
  adminUsersOffset: 0,
  adminQuestionsPage: 0,
  selectedAdminUserId: null,
  selectedQuestionId: null,
  searchResults: null,
};

const tg = window.Telegram?.WebApp;
if (tg) {
  tg.ready();
  tg.expand();
}

const mainPanel = document.querySelector("#main-panel");
const messagesPanel = document.querySelector("#messages-panel");
const tabsNode = document.querySelector("#tabs");
const refreshButton = document.querySelector("#refresh-button");

refreshButton.addEventListener("click", () => loadBootstrap(true));

function getAuthHeaders() {
  const params = new URLSearchParams(window.location.search);
  const headers = {};
  const initData = tg?.initData || params.get("initData") || "";
  if (initData) {
    headers["X-Telegram-Init-Data"] = initData;
  }

  const devUserId = params.get("dev_user_id");
  if (devUserId) {
    headers["X-Debug-User-Id"] = devUserId;
    headers["X-Debug-First-Name"] = params.get("dev_first_name") || "Dev";
    headers["X-Debug-Last-Name"] = params.get("dev_last_name") || "User";
    headers["X-Debug-Username"] = params.get("dev_username") || "dev_user";
  }
  return headers;
}

async function api(path, options = {}) {
  const config = {
    method: options.method || "GET",
    headers: {
      ...getAuthHeaders(),
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  };

  if (options.body) {
    config.body = JSON.stringify(options.body);
  }

  const response = await fetch(path, config);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message =
      detail?.message ||
      (typeof detail === "string" ? detail : null) ||
      payload?.message ||
      "Сталася помилка.";
    const error = new Error(message);
    error.code = detail?.code || payload?.code || "request_failed";
    throw error;
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function metricCard(label, value, note = "") {
  return `
    <article class="stat-card">
      <span class="eyebrow">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${note ? `<p class="muted">${escapeHtml(note)}</p>` : ""}
    </article>
  `;
}

function actionButton(label, handler, kind = "secondary") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `${kind}-button`;
  button.textContent = label;
  button.addEventListener("click", handler);
  return button;
}

function setMessage(kind, text) {
  if (!text) {
    messagesPanel.hidden = true;
    messagesPanel.innerHTML = "";
    return;
  }
  messagesPanel.hidden = false;
  messagesPanel.innerHTML = `
    <div class="message message--${kind}">
      <div>${escapeHtml(text)}</div>
      <button class="ghost-button" type="button" id="dismiss-message">Закрити</button>
    </div>
  `;
  document.querySelector("#dismiss-message").addEventListener("click", () => setMessage("", ""));
}

function setHero() {
  const user = state.bootstrap?.user;
  if (!user) {
    return;
  }

  document.querySelector("#hero-title").textContent = user.display_name;
  document.querySelector("#hero-subtitle").textContent = user.access.label;
  document.querySelector("#access-chip").textContent = user.access.has_access
    ? "Доступ активний"
    : "Доступ завершився";
  document.querySelector("#role-chip").textContent = user.is_admin ? "Адмін" : "Користувач";
}

function setTabs() {
  const user = state.bootstrap?.user;
  if (!user) {
    tabsNode.innerHTML = "";
    return;
  }

  const tabs = [
    { key: "home", label: "Головна" },
    { key: "learning", label: "Навчання" },
    { key: "testing", label: "Тестування" },
    { key: "stats", label: "Статистика" },
    { key: "help", label: "Допомога" },
  ];

  if (user.is_admin) {
    tabs.push({ key: "admin-users", label: "Користувачі" });
    tabs.push({ key: "admin-questions", label: "Питання" });
  }

  tabsNode.innerHTML = tabs
    .map(
      (tab) => `
        <button class="tab-button ${state.currentTab === tab.key ? "is-active" : ""}" type="button" data-tab="${tab.key}">
          ${escapeHtml(tab.label)}
        </button>
      `,
    )
    .join("");

  tabsNode.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.currentTab = button.dataset.tab;
      render();
      if (state.currentTab === "admin-users") {
        loadAdminUsers(state.adminUsersOffset);
      }
      if (state.currentTab === "admin-questions") {
        loadAdminQuestions(state.adminQuestionsPage);
      }
    });
  });
}

function render() {
  if (!state.bootstrap) {
    mainPanel.innerHTML = document.querySelector("#empty-state-template").innerHTML;
    return;
  }

  setHero();
  setTabs();

  if (state.currentView) {
    renderCurrentView();
    return;
  }

  switch (state.currentTab) {
    case "home":
      renderHome();
      break;
    case "learning":
      renderLearning();
      break;
    case "testing":
      renderTesting();
      break;
    case "stats":
      renderStats();
      break;
    case "help":
      renderHelp();
      break;
    case "admin-users":
      renderAdminUsers();
      break;
    case "admin-questions":
      renderAdminQuestions();
      break;
    default:
      renderHome();
      break;
  }
}

function renderHome() {
  const { user, catalog, stats } = state.bootstrap;
  const modulesSelected = catalog.ok_modules.filter((item) => item.selected).length;

  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Один інтерфейс для користувача й адміна</h2>
        <p>FastAPI віддає Mini App, а поточна база і банк питань працюють без окремого фронтенд-хостингу.</p>
      </div>
    </div>
    <div class="stats-grid">
      ${metricCard("Доступ", user.access.has_access ? "Активний" : "Завершився", user.access.label)}
      ${metricCard("Законодавство", String(catalog.counts.law), "Питань у банку")}
      ${metricCard("Модулі ОК", String(modulesSelected), "Обрано для навчання")}
      ${metricCard("Тести", String(stats.count), stats.last ? `Останній: ${stats.last.percent.toFixed(1)}%` : "Ще без результатів")}
    </div>
    <div class="cards-grid" style="margin-top:18px">
      <article class="grid-card">
        <span class="eyebrow">Навчання</span>
        <strong>${catalog.counts.questions}</strong>
        <p class="muted">Перейдіть до законодавства, модулів ОК або блоку помилок.</p>
      </article>
      <article class="grid-card">
        <span class="eyebrow">Тести</span>
        <strong>50 + 20×N</strong>
        <p class="muted">Законодавство бере 50 випадкових питань, кожен модуль ОК додає до 20.</p>
      </article>
      <article class="grid-card">
        <span class="eyebrow">Адмінка</span>
        <strong>${user.is_admin ? "Увімкнена" : "Недоступна"}</strong>
        <p class="muted">Керування доступом користувачів і редагування питань живуть у цій самій Mini App.</p>
      </article>
    </div>
  `;
}

function renderLearning() {
  const { user, catalog } = state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);

  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Навчання</h2>
        <p>Запускайте підготовку за законодавством або обраними модулями ОК. Перед стартом можна переглянути питання у префлайті.</p>
      </div>
      <div class="button-row" id="learning-actions"></div>
    </div>
    <div class="cards-grid">
      <article class="grid-card">
        <span class="eyebrow">Законодавство</span>
        <strong>${catalog.law_groups.length}</strong>
        <p class="muted">Розділів доступно в банку.</p>
      </article>
      <article class="grid-card">
        <span class="eyebrow">Модулі ОК</span>
        <strong>${selectedModules.length}</strong>
        <p class="muted">Вибрано у вашому профілі для навчання.</p>
      </article>
      <article class="grid-card">
        <span class="eyebrow">Статус доступу</span>
        <strong>${user.access.has_access ? "Можна стартувати" : "Лише перегляд"}</strong>
        <p class="muted">${escapeHtml(user.access.label)}</p>
      </article>
    </div>
    <div class="two-column" style="margin-top:18px">
      <section class="inline-form">
        <div class="section-heading">
          <div>
            <h3>Розділи законодавства</h3>
            <p>Клік по розділу відкриє стартову підготовку; великі розділи автоматично діляться на частини по 50 питань.</p>
          </div>
        </div>
        <div class="list-stack" id="law-groups-stack"></div>
      </section>
      <section class="inline-form">
        <div class="section-heading">
          <div>
            <h3>Модулі ОК</h3>
            <p>Позначте потрібні модулі й запускайте конкретний рівень.</p>
          </div>
        </div>
        <div class="list-stack" id="ok-modules-stack"></div>
      </section>
    </div>
  `;

  document.querySelector("#learning-actions").append(
    actionButton("Перевірити помилки", async () => {
      try {
        state.currentView = await api("/api/mistakes/start", { method: "POST" });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    }),
  );

  const lawStack = document.querySelector("#law-groups-stack");
  catalog.law_groups.forEach((group) => {
    const node = document.createElement("article");
    node.className = "list-item";
    node.innerHTML = `
      <div class="list-item__main">
        <strong>${escapeHtml(group.title)}</strong>
        <span class="muted">${group.count} питань</span>
      </div>
      <div class="button-row"></div>
    `;
    const buttons = node.querySelector(".button-row");
    const partCount = Math.ceil(group.count / 50);
    if (partCount <= 1) {
      buttons.append(actionButton("Підготовка", () => startLearning({ kind: "law", group_key: group.key, part: 1 }), "primary"));
    } else {
      buttons.append(actionButton(`Частини (${partCount})`, () => showLawParts(group)));
    }
    buttons.append(actionButton("Рандом 50", () => startLearning({ kind: "lawrand", group_key: group.key })));
    lawStack.append(node);
  });

  const okStack = document.querySelector("#ok-modules-stack");
  const editor = document.createElement("section");
  editor.className = "inline-form";
  editor.innerHTML = `
    <div class="field">
      <label>Оберіть модулі ОК</label>
      <div class="chips-row" id="module-selector"></div>
    </div>
    <div class="button-row" id="module-selector-actions"></div>
  `;
  okStack.append(editor);

  const selectedNames = new Set(selectedModules.map((item) => item.name));
  const selector = editor.querySelector("#module-selector");
  catalog.ok_modules.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `pill-button ${selectedNames.has(item.name) ? "is-selected" : ""}`;
    button.textContent = item.label;
    button.addEventListener("click", () => {
      if (selectedNames.has(item.name)) {
        selectedNames.delete(item.name);
        button.classList.remove("is-selected");
      } else {
        selectedNames.add(item.name);
        button.classList.add("is-selected");
      }
    });
    selector.append(button);
  });

  editor.querySelector("#module-selector-actions").append(
    actionButton("Зберегти модулі", async () => {
      try {
        const response = await api("/api/preferences/ok-modules", {
          method: "POST",
          body: { modules: Array.from(selectedNames) },
        });
        state.bootstrap.user = response.user;
        state.bootstrap.catalog = response.catalog;
        setMessage("success", "Модулі збережено.");
        renderLearning();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
  );

  selectedModules.forEach((item) => {
    const card = document.createElement("article");
    card.className = "list-item";
    card.innerHTML = `
      <div class="list-item__main">
        <strong>${escapeHtml(item.label)}</strong>
        <span class="muted">Доступні рівні: ${item.levels.map((entry) => entry.level).join(", ")}</span>
      </div>
      <div class="button-row"></div>
    `;
    const buttons = card.querySelector(".button-row");
    item.levels.forEach((levelEntry) => {
      buttons.append(
        actionButton(
          `Рівень ${levelEntry.level}`,
          () => startLearning({ kind: "ok", module: item.name, level: levelEntry.level }),
          levelEntry.level === item.last_level ? "primary" : "secondary",
        ),
      );
    });
    okStack.append(card);
  });

  if (!selectedModules.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "Після вибору модулів тут з’являться кнопки старту за рівнями.";
    okStack.append(note);
  }
}

function showLawParts(group) {
  const partCount = Math.ceil(group.count / 50);
  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>${escapeHtml(group.title)}</h2>
        <p>${group.count} питань у розділі. Оберіть частину або випадкову добірку.</p>
      </div>
      <div class="button-row" id="law-parts-actions"></div>
    </div>
    <div class="cards-grid" id="law-parts-grid"></div>
  `;

  document.querySelector("#law-parts-actions").append(
    actionButton("Назад", () => renderLearning()),
    actionButton("Рандом 50", () => startLearning({ kind: "lawrand", group_key: group.key }), "primary"),
  );

  const grid = document.querySelector("#law-parts-grid");
  for (let part = 1; part <= partCount; part += 1) {
    const start = (part - 1) * 50 + 1;
    const end = Math.min(part * 50, group.count);
    const card = document.createElement("article");
    card.className = "grid-card";
    card.innerHTML = `
      <span class="eyebrow">Частина ${part}</span>
      <strong>${start}-${end}</strong>
      <p class="muted">Питання ${start}–${end} цього розділу.</p>
    `;
    const actions = document.createElement("div");
    actions.className = "button-row";
    actions.append(actionButton("Відкрити", () => startLearning({ kind: "law", group_key: group.key, part }), "primary"));
    card.append(actions);
    grid.append(card);
  }
}

function renderTesting() {
  const { catalog } = state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);

  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Тестування</h2>
        <p>Зберіть набір із законодавства та модулів ОК. Для кожного модуля можна лишити один або кілька рівнів.</p>
      </div>
    </div>
    <div class="inline-form">
      <div class="check-row">
        <input id="include-law" type="checkbox" checked />
        <label for="include-law">Додати 50 випадкових питань із законодавства</label>
      </div>
      <div id="test-module-config" class="list-stack"></div>
      <div class="button-row" id="test-actions"></div>
    </div>
  `;

  const configNode = document.querySelector("#test-module-config");
  const selections = {};

  selectedModules.forEach((item) => {
    const initialLevel = item.last_level || item.levels[0]?.level;
    selections[item.name] = initialLevel ? [initialLevel] : [];

    const block = document.createElement("article");
    block.className = "list-item";
    block.innerHTML = `
      <div class="list-item__main">
        <strong>${escapeHtml(item.label)}</strong>
        <span class="muted">Зніміть усі рівні, якщо хочете виключити модуль із тесту.</span>
      </div>
      <div class="button-row"></div>
    `;

    const actions = block.querySelector(".button-row");
    item.levels.forEach((levelEntry) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `pill-button ${selections[item.name].includes(levelEntry.level) ? "is-selected" : ""}`;
      button.textContent = `Рівень ${levelEntry.level}`;
      button.addEventListener("click", () => {
        const set = new Set(selections[item.name]);
        if (set.has(levelEntry.level)) {
          set.delete(levelEntry.level);
          button.classList.remove("is-selected");
        } else {
          set.add(levelEntry.level);
          button.classList.add("is-selected");
        }
        selections[item.name] = Array.from(set).sort((a, b) => a - b);
      });
      actions.append(button);
    });
    configNode.append(block);
  });

  if (!selectedModules.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "Спочатку виберіть хоча б один модуль ОК у розділі «Навчання».";
    configNode.append(note);
  }

  document.querySelector("#test-actions").append(
    actionButton("Почати тест", async () => {
      try {
        state.currentView = await api("/api/test/start", {
          method: "POST",
          body: {
            include_law: document.querySelector("#include-law").checked,
            module_levels: selections,
          },
        });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
  );
}

function renderStats() {
  const { stats } = state.bootstrap;
  const last = stats.last;
  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Статистика</h2>
        <p>Останні тести та середній результат збережені в базі PostgreSQL.</p>
      </div>
    </div>
    <div class="stats-grid">
      ${metricCard("Тестів", String(stats.count), "Останні 50 спроб")}
      ${metricCard("Середній результат", `${stats.avg.toFixed(1)}%`, "За завершеними тестами")}
      ${metricCard("Останній тест", last ? `${last.correct}/${last.total}` : "—", last ? last.finished_at_label || "Без дати" : "Ще не було")}
    </div>
  `;
}

function renderHelp() {
  const { links } = state.bootstrap;
  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Допомога</h2>
        <p>Основні зовнішні переходи живуть тут, щоб не втрачати їх після переносу в Mini App.</p>
      </div>
    </div>
    <div class="cards-grid">
      <article class="link-card">
        <span class="eyebrow">Telegram-група</span>
        ${links.group_url ? `<a href="${escapeHtml(links.group_url)}" target="_blank" rel="noreferrer">Відкрити групу</a>` : `<p class="muted">Посилання не налаштовано.</p>`}
      </article>
      <article class="link-card">
        <span class="eyebrow">Адміністратор</span>
        ${links.admin_url ? `<a href="${escapeHtml(links.admin_url)}" target="_blank" rel="noreferrer">Написати адміну</a>` : `<p class="muted">Контакт не налаштовано.</p>`}
      </article>
      <article class="link-card">
        <span class="eyebrow">Railway / Web URL</span>
        <a href="${escapeHtml(links.webapp_url)}" target="_blank" rel="noreferrer">${escapeHtml(links.webapp_url)}</a>
      </article>
    </div>
  `;
}

function renderCurrentView() {
  const view = state.currentView;
  if (!view) {
    render();
    return;
  }
  if (view.mode === "pretest") {
    renderPretest(view);
    return;
  }
  if (view.screen === "question") {
    renderQuestionView(view);
    return;
  }
  if (view.screen === "feedback") {
    renderFeedbackView(view);
    return;
  }
  if (view.screen === "result") {
    renderResultView(view);
    return;
  }
  if (view.screen === "review") {
    renderReviewView(view);
    return;
  }

  mainPanel.innerHTML = `
    <div class="empty-state">
      <h2>Активний стан не знайдено</h2>
      <p>${escapeHtml(view.message || "Спробуйте повернутися на головну.")}</p>
    </div>
  `;
}

function renderPreviewQuestion(question, index, total) {
  const options = question.choices
    .map(
      (choice) => `
        <div class="choice-review ${choice.is_correct ? "choice-review--correct" : ""}">
          <span class="choice-review__index">${choice.index}</span>
          <div>
            <div>${escapeHtml(choice.text)}</div>
            ${choice.is_correct ? `<div class="muted">Правильна відповідь</div>` : ""}
          </div>
        </div>
      `,
    )
    .join("");

  return `
    <span class="eyebrow">Питання ${index}/${total}</span>
    <h3>${escapeHtml(question.question)}</h3>
    <p class="muted">${escapeHtml(question.ok_label || question.topic || question.section || "")}</p>
    <div class="review-grid">${options}</div>
  `;
}

function renderPretest(view) {
  const total = view.total || 0;
  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Підготовка до запуску</h2>
        <p>${escapeHtml(view.header || "Оберіть стартову точку для сесії.")}</p>
      </div>
      <div class="button-row" id="pretest-actions"></div>
    </div>
    <div class="two-column">
      <section class="inline-form">
        <div class="section-heading">
          <div>
            <h3>Сітка питань</h3>
            <p>Натисніть номер, щоб переглянути питання й правильні відповіді перед стартом.</p>
          </div>
        </div>
        <div class="numbers-grid" id="pretest-grid"></div>
      </section>
      <section class="question-card" id="pretest-preview"></section>
    </div>
  `;

  document.querySelector("#pretest-actions").append(
    actionButton("Почати навчання", async () => {
      try {
        state.currentView = await api("/api/pretest/start", { method: "POST" });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
    actionButton("Закрити", async () => {
      await leaveCurrentView();
    }),
  );

  const grid = document.querySelector("#pretest-grid");
  for (let index = 0; index < total; index += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === view.selected_index ? "is-active" : "";
    button.textContent = String(index + 1);
    button.addEventListener("click", async () => {
      try {
        state.currentView = await api("/api/pretest/select", { method: "POST", body: { index } });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    });
    grid.append(button);
  }

  document.querySelector("#pretest-preview").innerHTML = renderPreviewQuestion(view.question, view.selected_index + 1, total);
}

function renderQuestionView(view) {
  const question = view.question;
  const choices = question.choices
    .map(
      (choice, index) => `
        <button class="choice-button" type="button" data-choice="${index}">
          <span class="choice-button__index">${choice.index}</span>
          <span>${escapeHtml(choice.text)}</span>
        </button>
      `,
    )
    .join("");

  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>${escapeHtml(view.header || "Активна сесія")}</h2>
        <p>Питання ${view.progress.current}/${view.progress.total}${view.progress.phase === "skipped" ? " • повтор пропущених" : ""}</p>
      </div>
      <div class="button-row" id="question-actions"></div>
    </div>
    <section class="question-card">
      <h3>${escapeHtml(question.question)}</h3>
      <div class="choice-grid">${choices}</div>
    </section>
  `;

  mainPanel.querySelectorAll("[data-choice]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        state.currentView = await api("/api/session/answer", {
          method: "POST",
          body: { choice: Number(button.dataset.choice) },
        });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    });
  });

  const actions = document.querySelector("#question-actions");
  if (view.actions.allow_skip) {
    actions.append(
      actionButton("Пропустити", async () => {
        try {
          state.currentView = await api("/api/session/skip", { method: "POST" });
          render();
        } catch (error) {
          setMessage("error", error.message);
        }
      }),
    );
  }
  actions.append(actionButton("Вийти", async () => leaveCurrentView(), "danger"));
}

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

function renderFeedbackView(view) {
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice-review ${statusToClass(option.status)}">
          <span class="choice-review__index">${option.index}</span>
          <div>
            <div>${escapeHtml(option.text)}</div>
            <div class="muted">${statusToLabel(option.status)}</div>
          </div>
        </div>
      `,
    )
    .join("");

  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Розбір помилки</h2>
        <p>${escapeHtml(view.header || "Пояснення перед переходом до наступного питання.")}</p>
      </div>
      <div class="button-row" id="feedback-actions"></div>
    </div>
    <section class="question-card">
      <h3>${escapeHtml(view.question.question)}</h3>
      <div class="review-grid">${options}</div>
    </section>
  `;

  document.querySelector("#feedback-actions").append(
    actionButton("Продовжити", async () => {
      try {
        state.currentView = await api("/api/session/next", { method: "POST" });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
    actionButton("Вийти", async () => leaveCurrentView(), "danger"),
  );
}

function renderResultView(view) {
  const summary = view.summary || {};
  const blocks = summary.blocks || [];
  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>${escapeHtml(summary.title || "Результат")}</h2>
        <p>Підсумок завершеної сесії.</p>
      </div>
      <div class="button-row" id="result-actions"></div>
    </div>
    <div class="stats-grid">
      ${typeof summary.correct === "number" ? metricCard("Правильно", `${summary.correct}/${summary.total}`) : ""}
      ${typeof summary.percent === "number" ? metricCard("Відсоток", `${summary.percent.toFixed(1)}%`) : ""}
      ${typeof summary.remaining === "number" ? metricCard("Залишилось у помилках", String(summary.remaining)) : ""}
      ${typeof summary.passed === "boolean" ? metricCard("Поріг 60%", summary.passed ? "Складено" : "Не складено") : ""}
    </div>
    ${
      blocks.length
        ? `
          <section class="inline-form" style="margin-top:18px">
            <div class="section-heading"><div><h3>По блоках</h3></div></div>
            <div class="list-stack">
              ${blocks
                .map(
                  (block) => `
                    <article class="list-item">
                      <div class="list-item__main">
                        <strong>${escapeHtml(block.name)}</strong>
                        <span class="muted">${block.correct} із ${block.total}</span>
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

  const actions = document.querySelector("#result-actions");
  if (view.mode === "test_result" && view.wrong_count > 0) {
    actions.append(
      actionButton("Показати помилки", async () => {
        try {
          state.currentView = await api("/api/test/review/open", { method: "POST" });
          render();
        } catch (error) {
          setMessage("error", error.message);
        }
      }, "primary"),
    );
  }
  actions.append(
    actionButton("На головну", async () => {
      state.currentView = null;
      await loadBootstrap();
    }),
  );
}

function renderReviewView(view) {
  const options = view.question.options
    .map(
      (option) => `
        <div class="choice-review ${statusToClass(option.status)}">
          <span class="choice-review__index">${option.index}</span>
          <div>
            <div>${escapeHtml(option.text)}</div>
            <div class="muted">${statusToLabel(option.status)}</div>
          </div>
        </div>
      `,
    )
    .join("");

  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Помилки тесту</h2>
        <p>Питання ${view.index + 1}/${view.total}</p>
      </div>
      <div class="button-row" id="review-actions"></div>
    </div>
    <section class="question-card">
      <h3>${escapeHtml(view.question.question)}</h3>
      <div class="review-grid">${options}</div>
      ${view.question.selected_missing ? `<p class="muted">Ваш вибір не зберігся для цього питання.</p>` : ""}
    </section>
  `;

  const actions = document.querySelector("#review-actions");
  if (view.actions.has_prev) {
    actions.append(
      actionButton("Попереднє", async () => {
        state.currentView = await api("/api/test/review/index", { method: "POST", body: { index: view.index - 1 } });
        render();
      }),
    );
  }
  if (view.actions.has_next) {
    actions.append(
      actionButton("Наступне", async () => {
        state.currentView = await api("/api/test/review/index", { method: "POST", body: { index: view.index + 1 } });
        render();
      }, "primary"),
    );
  }
  actions.append(
    actionButton("До результату", async () => {
      state.currentView = await api("/api/test/review/back", { method: "POST" });
      render();
    }),
  );
}

async function leaveCurrentView() {
  try {
    await api("/api/session/leave", { method: "POST" });
  } catch (error) {
    setMessage("error", error.message);
  }
  state.currentView = null;
  await loadBootstrap();
}

async function startLearning(payload) {
  try {
    state.currentView = await api("/api/learning/start", { method: "POST", body: payload });
    render();
  } catch (error) {
    setMessage("error", error.message);
  }
}

function renderAdminUsers() {
  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Користувачі</h2>
        <p>Керуйте безстроковим доступом і швидко перевіряйте поточний статус користувачів.</p>
      </div>
    </div>
    <div class="two-column">
      <section class="inline-form" id="admin-users-list"></section>
      <section class="inline-form" id="admin-user-detail">
        <div class="empty-state">
          <h2>Оберіть користувача</h2>
          <p>Деталі і кнопки керування з’являться тут.</p>
        </div>
      </section>
    </div>
  `;
}

async function loadAdminUsers(offset = 0) {
  try {
    const payload = await api(`/api/admin/users?offset=${offset}&limit=10`);
    state.adminUsersOffset = payload.offset;
    renderAdminUsers();
    const listNode = document.querySelector("#admin-users-list");
    listNode.innerHTML = `
      <div class="section-heading">
        <div>
          <h3>Список</h3>
          <p>У цій сторінці: активні ${payload.counts.active}, тріал ${payload.counts.trial}, без доступу ${payload.counts.expired}.</p>
        </div>
      </div>
      <div class="list-stack" id="admin-users-items"></div>
      <div class="button-row" id="admin-users-pagination"></div>
    `;

    const items = document.querySelector("#admin-users-items");
    payload.items.forEach((item) => {
      const row = document.createElement("article");
      row.className = "list-item";
      row.innerHTML = `
        <div class="list-item__main">
          <strong>${escapeHtml(item.display_name)}</strong>
          <span class="muted">ID ${item.user_id} • ${escapeHtml(item.access.label)}</span>
        </div>
      `;
      row.append(actionButton("Відкрити", () => loadAdminUserDetail(item.user_id), "primary"));
      items.append(row);
    });

    const pagination = document.querySelector("#admin-users-pagination");
    if (payload.has_prev) {
      pagination.append(actionButton("Попередні", () => loadAdminUsers(Math.max(0, payload.offset - payload.limit))));
    }
    if (payload.has_next) {
      pagination.append(actionButton("Наступні", () => loadAdminUsers(payload.offset + payload.limit)));
    }
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function loadAdminUserDetail(userId) {
  try {
    state.selectedAdminUserId = userId;
    const payload = await api(`/api/admin/users/${userId}`);
    const detailNode = document.querySelector("#admin-user-detail");
    detailNode.innerHTML = `
      <div class="section-heading">
        <div>
          <h3>Користувач ${payload.user_id}</h3>
          <p>${escapeHtml(payload.access.label)}</p>
        </div>
      </div>
      <div class="list-stack">
        <article class="list-item">
          <div class="list-item__main">
            <strong>${escapeHtml([payload.first_name, payload.last_name].filter(Boolean).join(" ") || "—")}</strong>
            <span class="muted">Створено: ${escapeHtml(payload.created_at || "—")}</span>
          </div>
        </article>
      </div>
      <div class="button-row" id="admin-user-detail-actions"></div>
    `;

    const actions = document.querySelector("#admin-user-detail-actions");
    actions.append(
      actionButton(
        payload.access.state === "sub_infinite" ? "Скасувати безстроковий доступ" : "Дати безстроковий доступ",
        async () => {
          try {
            await api(`/api/admin/users/${userId}/subscription`, {
              method: "POST",
              body: { infinite: payload.access.state !== "sub_infinite" },
            });
            setMessage("success", "Доступ користувача оновлено.");
            await loadAdminUsers(state.adminUsersOffset);
            await loadAdminUserDetail(userId);
          } catch (error) {
            setMessage("error", error.message);
          }
        },
        "primary",
      ),
    );
    actions.append(
      actionButton(
        "Забрати доступ",
        async () => {
          try {
            await api(`/api/admin/users/${userId}/subscription`, {
              method: "POST",
              body: { infinite: false },
            });
            setMessage("success", "Доступ користувача оновлено.");
            await loadAdminUsers(state.adminUsersOffset);
            await loadAdminUserDetail(userId);
          } catch (error) {
            setMessage("error", error.message);
          }
        },
        "danger",
      ),
    );
  } catch (error) {
    setMessage("error", error.message);
  }
}

function renderAdminQuestions() {
  mainPanel.innerHTML = `
    <div class="section-heading">
      <div>
        <h2>Питання</h2>
        <p>Список, пошук і редагування в одному місці: текст питання, варіанти та правильні відповіді.</p>
      </div>
    </div>
    <div class="two-column">
      <section class="inline-form" id="admin-question-browser"></section>
      <section class="inline-form" id="admin-question-editor">
        <div class="empty-state">
          <h2>Оберіть питання</h2>
          <p>Форма редагування з’явиться після вибору зі списку або пошуку.</p>
        </div>
      </section>
    </div>
  `;
}

async function loadAdminQuestions(page = 0) {
  try {
    const payload = await api(`/api/admin/questions?page=${page}&page_size=10`);
    state.adminQuestionsPage = payload.page;
    state.searchResults = null;
    renderAdminQuestions();

    const browser = document.querySelector("#admin-question-browser");
    browser.innerHTML = `
      <div class="field">
        <label for="question-search-input">Пошук за текстом питання</label>
        <div class="button-row">
          <input id="question-search-input" type="text" placeholder="Введіть щонайменше 3 символи" />
          <button id="question-search-button" class="primary-button" type="button">Шукати</button>
        </div>
      </div>
      <div class="list-stack" id="question-list"></div>
      <div class="button-row" id="question-pagination"></div>
    `;

    document.querySelector("#question-search-button").addEventListener("click", async () => {
      const query = document.querySelector("#question-search-input").value.trim();
      if (!query) {
        return;
      }
      try {
        const result = await api(`/api/admin/questions/search?q=${encodeURIComponent(query)}`);
        state.searchResults = result.items;
        renderQuestionList(result.items);
        document.querySelector("#question-pagination").innerHTML = "";
      } catch (error) {
        setMessage("error", error.message);
      }
    });

    renderQuestionList(payload.items);

    const pagination = document.querySelector("#question-pagination");
    if (payload.page > 0) {
      pagination.append(actionButton("Попередні", () => loadAdminQuestions(payload.page - 1)));
    }
    if (payload.page + 1 < payload.pages) {
      pagination.append(actionButton("Наступні", () => loadAdminQuestions(payload.page + 1)));
    }
  } catch (error) {
    setMessage("error", error.message);
  }
}

function renderQuestionList(items) {
  const list = document.querySelector("#question-list");
  list.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = "list-item";
    row.innerHTML = `
      <div class="list-item__main">
        <strong>#${item.id}</strong>
        <span>${escapeHtml(item.question)}</span>
        <span class="muted">${escapeHtml(item.ok || item.topic || "Без модуля")}</span>
      </div>
    `;
    row.append(actionButton("Редагувати", () => loadQuestionDetail(item.id), "primary"));
    list.append(row);
  });
}

async function loadQuestionDetail(questionId) {
  try {
    state.selectedQuestionId = questionId;
    const payload = await api(`/api/admin/questions/${questionId}`);
    const question = payload.question;
    const editor = document.querySelector("#admin-question-editor");
    editor.innerHTML = `
      <div class="section-heading">
        <div>
          <h3>Питання #${question.id}</h3>
          <p>${escapeHtml(question.ok_label || question.topic || question.section || "Без групи")}</p>
        </div>
      </div>
      <form id="question-edit-form" class="list-stack">
        <div class="field">
          <label for="question-text">Текст питання</label>
          <textarea id="question-text">${escapeHtml(question.question)}</textarea>
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
          <textarea id="choice-${choice.index}">${escapeHtml(choice.text)}</textarea>
        </div>
        <div class="check-row">
          <input id="correct-${choice.index}" type="checkbox" ${choice.is_correct ? "checked" : ""} />
          <label for="correct-${choice.index}">Правильна відповідь</label>
        </div>
      `;
      choicesEditor.append(block);
    });

    document.querySelector("#reload-question").addEventListener("click", () => loadQuestionDetail(questionId));
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
        const updated = await api(`/api/admin/questions/${questionId}`, {
          method: "PATCH",
          body: {
            question: document.querySelector("#question-text").value.trim(),
            choices: updatedChoices,
            correct,
          },
        });
        setMessage("success", "Питання збережено.");
        await loadQuestionDetail(updated.question.id);
        if (state.searchResults) {
          renderQuestionList(state.searchResults);
        } else {
          await loadAdminQuestions(state.adminQuestionsPage);
        }
      } catch (error) {
        setMessage("error", error.message);
      }
    });
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function loadBootstrap(showSuccess = false) {
  try {
    const payload = await api("/api/bootstrap");
    state.bootstrap = payload;
    state.currentView = payload.saved_view || null;
    if (!state.currentTab || (state.currentTab.startsWith("admin") && !payload.user.is_admin)) {
      state.currentTab = "home";
    }
    if (showSuccess) {
      setMessage("success", "Дані оновлено.");
    }
    render();

    if (state.currentTab === "admin-users") {
      loadAdminUsers(state.adminUsersOffset);
    }
    if (state.currentTab === "admin-questions") {
      loadAdminQuestions(state.adminQuestionsPage);
    }
  } catch (error) {
    setMessage("error", error.message);
    mainPanel.innerHTML = `
      <div class="empty-state">
        <h2>Mini App не підключився</h2>
        <p>${escapeHtml(error.message)}</p>
        <p class="muted">Для локальної перевірки можна додати в URL параметр <code>?dev_user_id=123</code>, якщо на сервері дозволено debug-auth.</p>
      </div>
    `;
  }
}

loadBootstrap();
