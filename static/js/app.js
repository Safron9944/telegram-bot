import { refs } from "./core/dom.js";
import { state } from "./core/state.js";
import { api } from "./core/api.js";
import { initializeTelegram, impact, syncClosingConfirmation } from "./core/telegram.js";
import { initializeTheme } from "./core/theme.js";
import {
  actionButton,
  actionCard,
  bindInlineTargets,
  escapeHtml,
  metricCard,
  screenBar,
  setChrome,
  setMessage,
} from "./core/ui.js";
import {
  renderHelp,
  renderHome,
  renderLawLearning,
  renderLawParts,
  renderLearning,
  renderOkLearning,
  renderStats,
  renderTesting,
} from "./screens/user.js";
import {
  loadAdminQuestions,
  loadAdminUserDetail,
  loadAdminUsers,
  loadQuestionDetail,
  renderAdminHub,
  renderAdminQuestions,
  renderAdminUsers,
  runQuestionSearch,
} from "./screens/admin.js";
import { renderCurrentView } from "./screens/session.js";

initializeTelegram(() => {
  void goBack();
});
initializeTheme();

refs.refreshButton?.addEventListener("click", () => {
  void loadBootstrap(true);
});

refs.backButton?.addEventListener("click", () => {
  void goBack();
});

function createContext() {
  return {
    state,
    refs,
    api,
    impact,
    actionButton,
    actionCard,
    bindInlineTargets,
    escapeHtml,
    metricCard,
    setChrome,
    setMessage,
    screenBar: (activeScreen = state.currentScreen) => screenBar(activeScreen, { isAdmin: Boolean(state.bootstrap?.user?.is_admin) }),
    navigate,
    goHome,
    goBack,
    render,
    loadBootstrap,
    startLearning,
    startMistakesSession,
    leaveCurrentView,
    loadAdminUsers: (offset = state.adminUsersOffset) => loadAdminUsers(createContext(), offset),
    loadAdminUserDetail: (userId) => loadAdminUserDetail(createContext(), userId),
    loadAdminQuestions: (page = state.adminQuestionsPage) => loadAdminQuestions(createContext(), page),
    loadQuestionDetail: (questionId) => loadQuestionDetail(createContext(), questionId),
    runQuestionSearch: (query) => runQuestionSearch(createContext(), query),
  };
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
    void loadAdminUsers(createContext(), state.adminUsersOffset);
  }

  if (screen === "admin-questions") {
    void loadAdminQuestions(createContext(), state.adminQuestionsPage);
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
    syncClosingConfirmation(state.currentView);
    refs.mainPanel.innerHTML = refs.emptyStateTemplate.innerHTML;
    return;
  }

  if (state.currentScreen.startsWith("admin") && !state.bootstrap.user.is_admin) {
    state.currentScreen = "home";
    state.screenHistory = [];
  }

  const ctx = createContext();

  if (state.currentView) {
    renderCurrentView(ctx);
    syncClosingConfirmation(state.currentView);
    return;
  }

  switch (state.currentScreen) {
    case "home":
      renderHome(ctx);
      break;
    case "learning":
      renderLearning(ctx);
      break;
    case "law-learning":
      renderLawLearning(ctx);
      break;
    case "ok-learning":
      renderOkLearning(ctx);
      break;
    case "law-parts":
      renderLawParts(ctx);
      break;
    case "testing":
      renderTesting(ctx);
      break;
    case "stats":
      renderStats(ctx);
      break;
    case "help":
      renderHelp(ctx);
      break;
    case "admin":
      renderAdminHub(ctx);
      break;
    case "admin-users":
      renderAdminUsers(ctx);
      break;
    case "admin-questions":
      renderAdminQuestions(ctx);
      break;
    default:
      state.currentScreen = "home";
      renderHome(ctx);
      break;
  }

  syncClosingConfirmation(state.currentView);
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
    refs.mainPanel.innerHTML = `
      <div class="empty-state">
        <h2>Mini App не підключився</h2>
        <p>${escapeHtml(error.message)}</p>
        <p class="muted">Для локальної перевірки можна додати в URL параметр <code>?dev_user_id=123</code>, якщо на сервері дозволено debug-auth.</p>
      </div>
    `;
  }
}

loadBootstrap();
