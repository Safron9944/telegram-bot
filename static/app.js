const state = {
  bootstrap: null,
  currentScreen: "home",
  screenHistory: [],
  currentView: null,
  selectedLawGroup: null,
  adminUsersOffset: 0,
  adminQuestionsPage: 0,
  selectedAdminUserId: null,
  selectedQuestionId: null,
  searchResults: null,
  questionSearchQuery: "",
};

const tg = window.Telegram?.WebApp;
const mainPanel = document.querySelector("#main-panel");
const messagesPanel = document.querySelector("#messages-panel");
const refreshButton = document.querySelector("#refresh-button");
const backButton = document.querySelector("#back-button");
const titleNode = document.querySelector("#app-title");
const subtitleNode = document.querySelector("#app-subtitle");
const eyebrowNode = document.querySelector("#app-eyebrow");

let telegramBackAttached = false;
let closingConfirmationEnabled = false;

initializeTelegram();
refreshButton.addEventListener("click", () => loadBootstrap(true));
backButton.addEventListener("click", () => {
  void goBack();
});

function initializeTelegram() {
  if (!tg) {
    return;
  }

  tg.ready();
  tg.expand();
  applyTelegramTheme();

  if (!telegramBackAttached) {
    tg.BackButton?.onClick?.(() => {
      void goBack();
    });
    telegramBackAttached = true;
  }

  tg.onEvent?.("themeChanged", applyTelegramTheme);
}

function normalizeHex(value, fallback) {
  const raw = String(value || "").trim();
  if (/^#[0-9a-f]{6}$/i.test(raw)) {
    return raw;
  }
  if (/^#[0-9a-f]{3}$/i.test(raw)) {
    const [, r, g, b] = raw;
    return `#${r}${r}${g}${g}${b}${b}`;
  }
  return fallback;
}

function hexToRgba(value, alpha, fallback = "#000000") {
  const hex = normalizeHex(value, fallback).slice(1);
  const red = Number.parseInt(hex.slice(0, 2), 16);
  const green = Number.parseInt(hex.slice(2, 4), 16);
  const blue = Number.parseInt(hex.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function applyTelegramTheme() {
  if (!tg) {
    return;
  }

  const theme = tg.themeParams || {};
  const dark = tg.colorScheme === "dark";
  const bg = normalizeHex(theme.bg_color, dark ? "#0f1728" : "#e7ecf3");
  const secondary = normalizeHex(theme.secondary_bg_color || bg, dark ? "#182234" : "#f3f6fb");
  const text = normalizeHex(theme.text_color, dark ? "#f8fbff" : "#162033");
  const hint = normalizeHex(theme.hint_color, dark ? "#93a7c4" : "#66758c");
  const accent = normalizeHex(theme.button_color, dark ? "#4b8dff" : "#2168f5");
  const accentText = normalizeHex(theme.button_text_color, "#ffffff");

  document.documentElement.style.colorScheme = dark ? "dark" : "light";
  document.body.classList.toggle("is-dark", dark);
  document.documentElement.style.setProperty("--app-bg", bg);
  document.documentElement.style.setProperty("--app-bg-glow", hexToRgba(secondary, dark ? 0.82 : 0.95, secondary));
  document.documentElement.style.setProperty("--surface", hexToRgba(secondary, dark ? 0.72 : 0.84, secondary));
  document.documentElement.style.setProperty("--surface-strong", hexToRgba(secondary, dark ? 0.9 : 0.95, secondary));
  document.documentElement.style.setProperty("--surface-soft", hexToRgba(secondary, dark ? 0.82 : 0.78, secondary));
  document.documentElement.style.setProperty(
    "--surface-tint",
    `linear-gradient(180deg, ${hexToRgba(secondary, dark ? 0.96 : 0.98, secondary)}, ${hexToRgba(bg, dark ? 0.78 : 0.92, bg)})`,
  );
  document.documentElement.style.setProperty("--text", text);
  document.documentElement.style.setProperty("--muted", hint);
  document.documentElement.style.setProperty("--accent", accent);
  document.documentElement.style.setProperty("--accent-strong", accent);
  document.documentElement.style.setProperty("--accent-contrast", accentText);
  document.documentElement.style.setProperty("--accent-soft", hexToRgba(accent, dark ? 0.22 : 0.12, accent));
  document.documentElement.style.setProperty("--line", hexToRgba(text, dark ? 0.18 : 0.1, text));
  document.documentElement.style.setProperty("--line-strong", hexToRgba(text, dark ? 0.28 : 0.16, text));

  tg.setHeaderColor?.(secondary);
  tg.setBackgroundColor?.(bg);
}

function impact(style = "light") {
  tg?.HapticFeedback?.impactOccurred?.(style);
}

function setChrome({ eyebrow, title, subtitle, showBack = false, showRefresh = true }) {
  eyebrowNode.textContent = eyebrow;
  titleNode.textContent = title;
  subtitleNode.textContent = subtitle;

  backButton.hidden = !showBack;
  refreshButton.hidden = !showRefresh;

  if (showBack) {
    tg?.BackButton?.show?.();
  } else {
    tg?.BackButton?.hide?.();
  }
}

function syncClosingConfirmation() {
  const activeView = state.currentView;
  const shouldProtect = Boolean(
    activeView &&
      (activeView.mode === "pretest" || activeView.screen === "question" || activeView.screen === "feedback"),
  );

  if (shouldProtect && !closingConfirmationEnabled) {
    tg?.enableClosingConfirmation?.();
    closingConfirmationEnabled = true;
    return;
  }

  if (!shouldProtect && closingConfirmationEnabled) {
    tg?.disableClosingConfirmation?.();
    closingConfirmationEnabled = false;
  }
}

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
  button.addEventListener("click", async () => {
    impact(kind === "primary" ? "medium" : "light");
    await handler();
  });
  return button;
}

function actionCard({ code, title, body, meta, screen, link }) {
  const isDisabled = !screen && !link;
  const attr = screen
    ? `data-screen-target="${escapeHtml(screen)}"`
    : `data-link-target="${escapeHtml(link || "")}"`;

  return `
    <button class="nav-card" type="button" ${attr} ${isDisabled ? "disabled" : ""}>
      <div class="nav-card__top">
        <span class="monogram">${escapeHtml(code)}</span>
        <span class="nav-card__meta">${escapeHtml(meta)}</span>
      </div>
      <div class="nav-card__body">
        <h3 class="nav-card__title">${escapeHtml(title)}</h3>
        <p>${escapeHtml(body)}</p>
      </div>
    </button>
  `;
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
  document.querySelector("#dismiss-message")?.addEventListener("click", () => setMessage("", ""));
}

function openExternalLink(url) {
  if (!url) {
    return;
  }

  try {
    const parsed = new URL(url);
    if (tg?.openTelegramLink && (parsed.hostname.endsWith("t.me") || parsed.protocol === "tg:")) {
      tg.openTelegramLink(url);
      return;
    }
    if (tg?.openLink) {
      tg.openLink(url);
      return;
    }
  } catch {
    // Fall back to window.open below.
  }

  window.open(url, "_blank", "noopener,noreferrer");
}

function bindInlineTargets(root = mainPanel) {
  root.querySelectorAll("[data-screen-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.screenTarget;
      if (target) {
        navigate(target);
      }
    });
  });

  root.querySelectorAll("[data-link-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.linkTarget;
      if (target) {
        impact("light");
        openExternalLink(target);
      }
    });
  });
}

function navigate(screen, options = {}) {
  if (!screen) {
    return;
  }

  if (screen.startsWith("admin") && !state.bootstrap?.user?.is_admin) {
    setMessage("error", "Режим адміністратора недоступний.");
    return;
  }

  if (options.replace) {
    state.currentScreen = screen;
  } else if (screen !== state.currentScreen) {
    state.screenHistory.push(state.currentScreen);
    state.currentScreen = screen;
  }

  impact("light");
  render();
  ensureScreenData(screen);
}

function goHome() {
  state.screenHistory = [];
  state.currentScreen = "home";
  render();
}

async function goBack() {
  if (state.currentView) {
    if (state.currentView.screen === "review") {
      try {
        state.currentView = await api("/api/test/review/back", { method: "POST" });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
      return;
    }

    if (state.currentView.screen === "result") {
      state.currentView = null;
      await loadBootstrap();
      return;
    }

    await leaveCurrentView();
    return;
  }

  if (state.screenHistory.length) {
    const previous = state.screenHistory.pop();
    state.currentScreen = previous || "home";
    if (previous !== "law-parts") {
      state.selectedLawGroup = null;
    }
    render();
    ensureScreenData();
    return;
  }

  if (state.currentScreen !== "home") {
    state.currentScreen = "home";
    render();
  }
}

function ensureScreenData(screen = state.currentScreen) {
  if (state.currentView) {
    return;
  }

  if (screen === "admin-users") {
    void loadAdminUsers(state.adminUsersOffset);
  }

  if (screen === "admin-questions") {
    void loadAdminQuestions(state.adminQuestionsPage);
  }
}

function render() {
  if (!state.bootstrap) {
    setChrome({
      eyebrow: "Підготовка",
      title: "Підключення…",
      subtitle: "Очікуємо дані від сервера.",
      showBack: false,
    });
    syncClosingConfirmation();
    mainPanel.innerHTML = document.querySelector("#empty-state-template").innerHTML;
    return;
  }

  if (state.currentScreen.startsWith("admin") && !state.bootstrap.user.is_admin) {
    state.currentScreen = "home";
    state.screenHistory = [];
  }

  if (state.currentView) {
    renderCurrentView();
    syncClosingConfirmation();
    return;
  }

  switch (state.currentScreen) {
    case "home":
      renderHome();
      break;
    case "learning":
      renderLearning();
      break;
    case "law-parts":
      renderLawParts();
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
    case "admin":
      renderAdminHub();
      break;
    case "admin-users":
      renderAdminUsers();
      break;
    case "admin-questions":
      renderAdminQuestions();
      break;
    default:
      state.currentScreen = "home";
      renderHome();
      break;
  }

  syncClosingConfirmation();
}

function renderHome() {
  const { user, catalog, stats } = state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);
  const accessClass = user.access.has_access ? "is-active" : "is-danger";
  const lastResult = stats.last ? `${stats.last.percent.toFixed(1)}%` : "Ще не було";

  setChrome({
    eyebrow: "Підготовка",
    title: "Підготовка",
    subtitle: user.access.label,
    showBack: false,
  });

  mainPanel.innerHTML = `
    <section class="summary-card">
      <div class="summary-card__row">
        <div class="summary-card__copy">
          <span class="eyebrow">Профіль</span>
          <strong class="summary-card__name">${escapeHtml(user.display_name)}</strong>
          <p>${escapeHtml(user.access.label)}</p>
        </div>
        <div class="chips-row">
          <span class="status-chip ${accessClass}">${user.access.has_access ? "Доступ активний" : "Доступ обмежений"}</span>
          <span class="status-chip">${selectedModules.length} модулів ОК</span>
          <span class="status-chip">${stats.count} тестів</span>
        </div>
      </div>
      <div class="button-row" id="home-quick-actions"></div>
    </section>

    <section class="dashboard-grid">
      ${actionCard({
        code: "LE",
        title: "Навчання",
        body: "Законодавство, модулі ОК і окремий запуск блоку помилок.",
        meta: `${catalog.law_groups.length} розділів`,
        screen: "learning",
      })}
      ${actionCard({
        code: "TE",
        title: "Тестування",
        body: "Зберіть власну конфігурацію тесту з законодавства та модулів.",
        meta: "Старт у 1 клік",
        screen: "testing",
      })}
      ${actionCard({
        code: "PR",
        title: "Прогрес",
        body: "Середній результат, останній тест і загальна динаміка.",
        meta: lastResult,
        screen: "stats",
      })}
      ${actionCard({
        code: "HE",
        title: "Підтримка",
        body: "Telegram-група, контакт адміністратора та сервісні переходи.",
        meta: "Допомога",
        screen: "help",
      })}
    </section>

    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Стан зараз</h2>
          <p>Усе важливе зібране на одному екрані, щоб Mini App відчувався як робочий інструмент, а не сайт.</p>
        </div>
      </div>
      <div class="metrics-grid">
        ${metricCard("Доступ", user.access.has_access ? "Активний" : "Обмежений", user.access.label)}
        ${metricCard("Законодавство", String(catalog.counts.law), "Питань у банку")}
        ${metricCard("Модулі ОК", String(selectedModules.length), "Обрані для підготовки")}
        ${metricCard("Останній тест", lastResult, stats.last ? stats.last.finished_at_label || "Без дати" : "Ще немає історії")}
      </div>
    </section>
  `;

  const quickActions = document.querySelector("#home-quick-actions");
  quickActions.append(
    actionButton("Відкрити навчання", async () => navigate("learning"), "primary"),
    actionButton("Почати тест", async () => navigate("testing")),
    actionButton("Перевірити помилки", startMistakesSession, "ghost"),
  );

  bindInlineTargets();
}

function renderLearning() {
  const { user, catalog } = state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);

  setChrome({
    eyebrow: "Навчання",
    title: "Навчання",
    subtitle: "Законодавство, модулі ОК і запуск повторення помилок.",
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Старт підготовки</h2>
          <p>Оберіть розділ, модуль або швидкий сценарій. Навіть без активного доступу тут видно поточну конфігурацію.</p>
        </div>
        <div class="chips-row">
          <span class="status-chip ${user.access.has_access ? "is-active" : "is-danger"}">${user.access.has_access ? "Можна стартувати" : "Лише перегляд"}</span>
          <span class="status-chip">${selectedModules.length} модулів</span>
        </div>
      </div>
      <div class="button-row" id="learning-actions"></div>
    </section>

    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Розділи законодавства</h2>
          <p>Великі розділи автоматично діляться на частини по 50 питань.</p>
        </div>
      </div>
      <div class="list-stack" id="law-groups-stack"></div>
    </section>

    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Модулі ОК</h2>
          <p>Збережіть потрібні модулі і запускайте рівні окремо.</p>
        </div>
      </div>
      <div class="section-stack" id="ok-modules-stack"></div>
    </section>
  `;

  document.querySelector("#learning-actions").append(
    actionButton("Перевірити помилки", startMistakesSession, "primary"),
    actionButton("Налаштувати тест", async () => navigate("testing")),
  );

  const lawStack = document.querySelector("#law-groups-stack");
  catalog.law_groups.forEach((group) => {
    const node = document.createElement("article");
    node.className = "list-item";
    node.innerHTML = `
      <div class="list-item__main">
        <span class="list-item__eyebrow">Розділ</span>
        <strong>${escapeHtml(group.title)}</strong>
        <span class="list-item__meta">${group.count} питань у банку</span>
      </div>
      <div class="button-row"></div>
    `;

    const actions = node.querySelector(".button-row");
    const partCount = Math.ceil(group.count / 50);
    if (partCount <= 1) {
      actions.append(
        actionButton("Підготовка", async () => startLearning({ kind: "law", group_key: group.key, part: 1 }), "primary"),
      );
    } else {
      actions.append(actionButton(`Частини (${partCount})`, async () => showLawParts(group)));
    }
    actions.append(actionButton("Рандом 50", async () => startLearning({ kind: "lawrand", group_key: group.key })));
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
      impact("light");
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
        impact("medium");
        renderLearning();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
  );

  if (!selectedModules.length) {
    const note = document.createElement("div");
    note.className = "empty-state";
    note.innerHTML = `
      <h2>Поки що без модулів</h2>
      <p>Позначте потрібні модулі вище, і тут з’являться кнопки запуску за рівнями.</p>
    `;
    okStack.append(note);
    return;
  }

  selectedModules.forEach((item) => {
    const card = document.createElement("article");
    card.className = "list-item";
    card.innerHTML = `
      <div class="list-item__main">
        <span class="list-item__eyebrow">Модуль ОК</span>
        <strong>${escapeHtml(item.label)}</strong>
        <span class="list-item__meta">Доступні рівні: ${item.levels.map((entry) => entry.level).join(", ")}</span>
      </div>
      <div class="button-row"></div>
    `;

    const buttons = card.querySelector(".button-row");
    item.levels.forEach((levelEntry) => {
      buttons.append(
        actionButton(
          `Рівень ${levelEntry.level}`,
          async () => startLearning({ kind: "ok", module: item.name, level: levelEntry.level }),
          levelEntry.level === item.last_level ? "primary" : "secondary",
        ),
      );
    });
    okStack.append(card);
  });
}

function showLawParts(group) {
  state.selectedLawGroup = group;
  navigate("law-parts");
}

function renderLawParts() {
  const group = state.selectedLawGroup;
  if (!group) {
    state.currentScreen = "learning";
    renderLearning();
    return;
  }

  const partCount = Math.ceil(group.count / 50);
  setChrome({
    eyebrow: "Навчання",
    title: group.title,
    subtitle: `${group.count} питань у розділі.`,
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Частини розділу</h2>
          <p>Оберіть конкретну частину або випадкову добірку з цього блоку.</p>
        </div>
      </div>
      <div class="button-row" id="law-parts-actions"></div>
    </section>
    <section class="cards-grid" id="law-parts-grid"></section>
  `;

  document.querySelector("#law-parts-actions").append(
    actionButton("До навчання", async () => goBack()),
    actionButton("Рандом 50", async () => startLearning({ kind: "lawrand", group_key: group.key }), "primary"),
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
    actions.append(actionButton("Відкрити", async () => startLearning({ kind: "law", group_key: group.key, part }), "primary"));
    card.append(actions);
    grid.append(card);
  }
}

function renderTesting() {
  const { catalog } = state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);

  setChrome({
    eyebrow: "Тестування",
    title: "Тестування",
    subtitle: "Зберіть набір з законодавства та модулів ОК перед стартом.",
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Конфігурація тесту</h2>
          <p>Для кожного модуля можна лишити один або кілька рівнів. Якщо рівні не вибрані, модуль не потрапить у тест.</p>
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
    </section>
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
        <span class="list-item__eyebrow">Модуль ОК</span>
        <strong>${escapeHtml(item.label)}</strong>
        <span class="list-item__meta">Зніміть усі рівні, якщо хочете виключити модуль із цього тесту.</span>
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
        impact("light");
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
    const note = document.createElement("div");
    note.className = "empty-state";
    note.innerHTML = `
      <h2>Немає модулів для тесту</h2>
      <p>Спочатку оберіть хоча б один модуль ОК у розділі «Навчання».</p>
    `;
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
        impact("medium");
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
    actionButton("До навчання", async () => navigate("learning")),
  );
}

function renderStats() {
  const { stats } = state.bootstrap;
  const last = stats.last;

  setChrome({
    eyebrow: "Прогрес",
    title: "Прогрес",
    subtitle: "Коротка історія спроб і ваш середній результат.",
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Зведення</h2>
          <p>Останні тести і середній результат підтягуються із збереженої історії.</p>
        </div>
      </div>
      <div class="metrics-grid">
        ${metricCard("Тестів", String(stats.count), "Останні 50 спроб")}
        ${metricCard("Середній результат", `${stats.avg.toFixed(1)}%`, "За завершеними тестами")}
        ${metricCard("Останній тест", last ? `${last.correct}/${last.total}` : "—", last ? last.finished_at_label || "Без дати" : "Ще не було")}
        ${metricCard("Останній відсоток", last ? `${last.percent.toFixed(1)}%` : "—", last ? "Останній завершений тест" : "Немає даних")}
      </div>
    </section>

    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Що робити далі</h2>
          <p>Після короткої перевірки прогресу можна швидко повернутись у навчання або зібрати новий тест.</p>
        </div>
      </div>
      <div class="button-row" id="stats-actions"></div>
    </section>
  `;

  document.querySelector("#stats-actions").append(
    actionButton("До навчання", async () => navigate("learning"), "primary"),
    actionButton("Новий тест", async () => navigate("testing")),
  );
}

function renderHelp() {
  const { links, user } = state.bootstrap;

  setChrome({
    eyebrow: "Підтримка",
    title: "Підтримка",
    subtitle: "Контакти і зовнішні переходи без зайвих веб-елементів.",
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="support-grid">
      ${actionCard({
        code: "TG",
        title: "Telegram-група",
        body: links.group_url ? "Швидкий перехід до спільноти без виходу з контексту навчання." : "Посилання на групу поки не налаштовано.",
        meta: links.group_url ? "Відкрити" : "Недоступно",
        link: links.group_url || "",
      })}
      ${actionCard({
        code: "AD",
        title: "Адміністратор",
        body: links.admin_url ? "Написати адміну, якщо потрібен доступ або є питання по роботі бота." : "Контакт адміністратора поки не налаштовано.",
        meta: links.admin_url ? "Написати" : "Недоступно",
        link: links.admin_url || "",
      })}
    </section>

    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Сервісні дії</h2>
          <p>Тут зібрані допоміжні переходи, які не повинні заважати основному сценарію використання.</p>
        </div>
      </div>
      <div class="button-row" id="help-actions"></div>
    </section>

    ${
      user.is_admin
        ? `
          <section class="surface">
            <div class="section-header">
              <div class="section-copy">
                <h2>Режим адміністратора</h2>
                <p>Адмінські інструменти винесені в окремий режим, щоб не змішувати їх з користувацьким інтерфейсом.</p>
              </div>
            </div>
            <div class="button-row" id="admin-entry-actions"></div>
          </section>
        `
        : ""
    }
  `;

  const helpActions = document.querySelector("#help-actions");
  helpActions.append(
    actionButton("Оновити дані", async () => loadBootstrap(true), "primary"),
    actionButton("На головну", async () => goHome()),
  );

  if (user.is_admin) {
    document.querySelector("#admin-entry-actions")?.append(
      actionButton("Відкрити режим адміністратора", async () => navigate("admin"), "primary"),
    );
  }

  bindInlineTargets();
}

function renderAdminHub() {
  setChrome({
    eyebrow: "Адмін режим",
    title: "Адміністрування",
    subtitle: "Сервісні інструменти відокремлені від основного користувацького потоку.",
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Інструменти адміністратора</h2>
          <p>Оберіть потрібний напрям: користувачі або банк питань.</p>
        </div>
      </div>
      <div class="dashboard-grid">
        ${actionCard({
          code: "US",
          title: "Користувачі",
          body: "Безстроковий доступ, статус і перевірка поточних підписок.",
          meta: "Відкрити",
          screen: "admin-users",
        })}
        ${actionCard({
          code: "QA",
          title: "Питання",
          body: "Пошук і редагування текстів, варіантів та правильних відповідей.",
          meta: "Відкрити",
          screen: "admin-questions",
        })}
      </div>
    </section>
  `;

  bindInlineTargets();
}

function renderAdminUsers() {
  setChrome({
    eyebrow: "Адмін режим",
    title: "Користувачі",
    subtitle: "Керуйте доступом і переглядайте поточний стан користувачів.",
    showBack: true,
  });

  mainPanel.innerHTML = `
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

function renderAdminQuestions() {
  setChrome({
    eyebrow: "Адмін режим",
    title: "Банк питань",
    subtitle: "Пошук і редагування в одному режимі без зайвої навігації.",
    showBack: true,
  });

  mainPanel.innerHTML = `
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

  setChrome({
    eyebrow: "Сесія",
    title: "Активний стан",
    subtitle: "Не вдалося визначити активний сценарій.",
    showBack: true,
  });
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
  setChrome({
    eyebrow: "Передстарт",
    title: "Перед стартом",
    subtitle: view.header || "Оберіть питання для перегляду перед запуском сесії.",
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Префлайт сесії</h2>
          <p>Натисніть номер питання, щоб переглянути його разом із правильною відповіддю.</p>
        </div>
      </div>
      <div class="button-row" id="pretest-actions"></div>
    </section>
    <section class="split-layout">
      <section class="surface">
        <div class="section-header">
          <div class="section-copy">
            <h2>Сітка питань</h2>
            <p>У цій сесії доступно ${total} питань для швидкої перевірки.</p>
          </div>
        </div>
        <div class="numbers-grid" id="pretest-grid"></div>
      </section>
      <section class="question-card" id="pretest-preview"></section>
    </section>
  `;

  document.querySelector("#pretest-actions").append(
    actionButton("Почати навчання", async () => {
      try {
        state.currentView = await api("/api/pretest/start", { method: "POST" });
        impact("medium");
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
    actionButton("Закрити", leaveCurrentView),
  );

  const grid = document.querySelector("#pretest-grid");
  for (let index = 0; index < total; index += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === view.selected_index ? "is-active" : "";
    button.textContent = String(index + 1);
    button.addEventListener("click", async () => {
      try {
        impact("light");
        state.currentView = await api("/api/pretest/select", { method: "POST", body: { index } });
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    });
    grid.append(button);
  }

  document.querySelector("#pretest-preview").innerHTML = renderPreviewQuestion(
    view.question,
    view.selected_index + 1,
    total,
  );
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

  setChrome({
    eyebrow: "Активна сесія",
    title: view.header || "Сесія",
    subtitle: `Питання ${view.progress.current}/${view.progress.total}${view.progress.phase === "skipped" ? " • повтор пропущених" : ""}`,
    showBack: true,
    showRefresh: false,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Поточне питання</h2>
          <p>Виберіть один із варіантів нижче. Після відповіді Mini App одразу переведе вас далі.</p>
        </div>
      </div>
      <div class="button-row" id="question-actions"></div>
    </section>
    <section class="question-card">
      <h3>${escapeHtml(question.question)}</h3>
      <div class="choice-grid">${choices}</div>
    </section>
  `;

  mainPanel.querySelectorAll("[data-choice]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        impact("medium");
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
  actions.append(actionButton("Вийти", leaveCurrentView, "danger"));
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

  setChrome({
    eyebrow: "Розбір",
    title: "Розбір відповіді",
    subtitle: view.header || "Пояснення перед переходом до наступного питання.",
    showBack: true,
    showRefresh: false,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Розбір помилки</h2>
          <p>Подивіться правильний варіант і перейдіть до наступного питання.</p>
        </div>
      </div>
      <div class="button-row" id="feedback-actions"></div>
    </section>
    <section class="question-card">
      <h3>${escapeHtml(view.question.question)}</h3>
      <div class="review-grid">${options}</div>
    </section>
  `;

  document.querySelector("#feedback-actions").append(
    actionButton("Продовжити", async () => {
      try {
        state.currentView = await api("/api/session/next", { method: "POST" });
        impact("medium");
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
    }, "primary"),
    actionButton("Вийти", leaveCurrentView, "danger"),
  );
}

function renderResultView(view) {
  const summary = view.summary || {};
  const blocks = summary.blocks || [];

  setChrome({
    eyebrow: "Результат",
    title: summary.title || "Результат",
    subtitle: "Підсумок завершеної сесії.",
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>${escapeHtml(summary.title || "Результат")}</h2>
          <p>Сесію завершено. Нижче коротке зведення та наступні дії.</p>
        </div>
      </div>
      <div class="button-row" id="result-actions"></div>
    </section>
    <section class="metrics-grid">
      ${typeof summary.correct === "number" ? metricCard("Правильно", `${summary.correct}/${summary.total}`) : ""}
      ${typeof summary.percent === "number" ? metricCard("Відсоток", `${summary.percent.toFixed(1)}%`) : ""}
      ${typeof summary.remaining === "number" ? metricCard("У помилках", String(summary.remaining)) : ""}
      ${typeof summary.passed === "boolean" ? metricCard("Поріг 60%", summary.passed ? "Складено" : "Не складено") : ""}
    </section>
    ${
      blocks.length
        ? `
          <section class="surface">
            <div class="section-header">
              <div class="section-copy">
                <h2>По блоках</h2>
                <p>Результат по кожній частині завершеної сесії.</p>
              </div>
            </div>
            <div class="list-stack">
              ${blocks
                .map(
                  (block) => `
                    <article class="list-item">
                      <div class="list-item__main">
                        <strong>${escapeHtml(block.name)}</strong>
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
      goHome();
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

  setChrome({
    eyebrow: "Помилки",
    title: "Помилки тесту",
    subtitle: `Питання ${view.index + 1}/${view.total}`,
    showBack: true,
  });

  mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Перегляд помилок</h2>
          <p>Перейдіть між питаннями і поверніться до результату, коли завершите перегляд.</p>
        </div>
      </div>
      <div class="button-row" id="review-actions"></div>
    </section>
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
    impact("medium");
    render();
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function startMistakesSession() {
  try {
    state.currentView = await api("/api/mistakes/start", { method: "POST" });
    impact("medium");
    render();
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function loadAdminUsers(offset = 0) {
  if (state.currentScreen !== "admin-users") {
    return;
  }

  try {
    const payload = await api(`/api/admin/users?offset=${offset}&limit=10`);
    if (state.currentScreen !== "admin-users") {
      return;
    }

    state.adminUsersOffset = payload.offset;
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
            <strong>${escapeHtml(item.display_name)}</strong>
            <span class="list-item__meta">ID ${item.user_id} • ${escapeHtml(item.access.label)}</span>
          </div>
        `;
        row.append(actionButton("Відкрити", async () => loadAdminUserDetail(item.user_id), "primary"));
        items.append(row);
      });
    }

    const pagination = document.querySelector("#admin-users-pagination");
    if (payload.has_prev) {
      pagination.append(actionButton("Попередні", async () => loadAdminUsers(Math.max(0, payload.offset - payload.limit))));
    }
    if (payload.has_next) {
      pagination.append(actionButton("Наступні", async () => loadAdminUsers(payload.offset + payload.limit)));
    }

    if (state.selectedAdminUserId) {
      void loadAdminUserDetail(state.selectedAdminUserId);
    }
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function loadAdminUserDetail(userId) {
  if (state.currentScreen !== "admin-users") {
    return;
  }

  try {
    state.selectedAdminUserId = userId;
    const payload = await api(`/api/admin/users/${userId}`);
    if (state.currentScreen !== "admin-users") {
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
          <p>${escapeHtml(payload.access.label)}</p>
        </div>
      </div>
      <div class="list-stack">
        <article class="list-item">
          <div class="list-item__main">
            <strong>${escapeHtml([payload.first_name, payload.last_name].filter(Boolean).join(" ") || "—")}</strong>
            <span class="list-item__meta">Створено: ${escapeHtml(payload.created_at || "—")}</span>
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
            impact("medium");
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

async function loadAdminQuestions(page = 0) {
  if (state.currentScreen !== "admin-questions") {
    return;
  }

  try {
    const payload = await api(`/api/admin/questions?page=${page}&page_size=10`);
    if (state.currentScreen !== "admin-questions") {
      return;
    }

    state.adminQuestionsPage = payload.page;
    if (!state.questionSearchQuery) {
      state.searchResults = null;
    }

    const browser = document.querySelector("#admin-question-browser");
    if (!browser) {
      return;
    }

    browser.innerHTML = `
      <div class="field">
        <label for="question-search-input">Пошук за текстом питання</label>
        <input id="question-search-input" type="text" value="${escapeHtml(state.questionSearchQuery)}" placeholder="Введіть щонайменше 3 символи" />
      </div>
      <div class="button-row" id="question-search-actions"></div>
      <div class="list-stack" id="question-list"></div>
      <div class="button-row" id="question-pagination"></div>
    `;

    const searchActions = document.querySelector("#question-search-actions");
    searchActions.append(
      actionButton("Шукати", async () => {
        const query = document.querySelector("#question-search-input").value.trim();
        await runQuestionSearch(query);
      }, "primary"),
    );

    if (state.questionSearchQuery) {
      searchActions.append(
        actionButton("Скинути пошук", async () => {
          state.questionSearchQuery = "";
          state.searchResults = null;
          await loadAdminQuestions(state.adminQuestionsPage);
        }),
      );
    }

    document.querySelector("#question-search-input").addEventListener("keydown", async (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        const query = event.currentTarget.value.trim();
        await runQuestionSearch(query);
      }
    });

    renderQuestionList(state.searchResults || payload.items);

    const pagination = document.querySelector("#question-pagination");
    if (!state.questionSearchQuery) {
      if (payload.page > 0) {
        pagination.append(actionButton("Попередні", async () => loadAdminQuestions(payload.page - 1)));
      }
      if (payload.page + 1 < payload.pages) {
        pagination.append(actionButton("Наступні", async () => loadAdminQuestions(payload.page + 1)));
      }
    }

    if (state.selectedQuestionId) {
      void loadQuestionDetail(state.selectedQuestionId);
    }
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function runQuestionSearch(query) {
  if (!query || query.length < 3) {
    setMessage("error", "Введіть щонайменше 3 символи для пошуку.");
    return;
  }

  try {
    const result = await api(`/api/admin/questions/search?q=${encodeURIComponent(query)}`);
    if (state.currentScreen !== "admin-questions") {
      return;
    }

    state.questionSearchQuery = query;
    state.searchResults = result.items;
    renderQuestionList(result.items);
    document.querySelector("#question-pagination").innerHTML = "";
    impact("light");
  } catch (error) {
    setMessage("error", error.message);
  }
}

function renderQuestionList(items) {
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
        <strong>${escapeHtml(item.question)}</strong>
        <span class="list-item__meta">${escapeHtml(item.ok || item.topic || "Без модуля")}</span>
      </div>
    `;
    row.append(actionButton("Редагувати", async () => loadQuestionDetail(item.id), "primary"));
    list.append(row);
  });
}

async function loadQuestionDetail(questionId) {
  if (state.currentScreen !== "admin-questions") {
    return;
  }

  try {
    state.selectedQuestionId = questionId;
    const payload = await api(`/api/admin/questions/${questionId}`);
    if (state.currentScreen !== "admin-questions") {
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
          <p>${escapeHtml(question.ok_label || question.topic || question.section || "Без групи")}</p>
        </div>
      </div>
      <form id="question-edit-form" class="section-stack">
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

    document.querySelector("#reload-question").addEventListener("click", async () => {
      await loadQuestionDetail(questionId);
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
        const updated = await api(`/api/admin/questions/${questionId}`, {
          method: "PATCH",
          body: {
            question: document.querySelector("#question-text").value.trim(),
            choices: updatedChoices,
            correct,
          },
        });
        setMessage("success", "Питання збережено.");
        impact("medium");
        await loadQuestionDetail(updated.question.id);

        if (state.questionSearchQuery) {
          await runQuestionSearch(state.questionSearchQuery);
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

    if (state.currentScreen.startsWith("admin") && !payload.user.is_admin) {
      state.currentScreen = "home";
      state.screenHistory = [];
    }

    if (showSuccess) {
      setMessage("success", "Дані оновлено.");
    }

    render();
    ensureScreenData();
  } catch (error) {
    setMessage("error", error.message);
    setChrome({
      eyebrow: "Підготовка",
      title: "Mini App не підключився",
      subtitle: "Не вдалося синхронізуватися з бекендом.",
      showBack: false,
    });
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
