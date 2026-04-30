import { refs } from "./core/dom.js";
import { state } from "./core/state.js";
import { api } from "./core/api.js";
import { initializeTelegram, impact, syncClosingConfirmation } from "./core/telegram.js";
import { initializeTheme } from "./core/theme.js";
import {
  actionButton,
  bindInlineTargets,
  cell,
  escapeHtml,
  group,
  setActiveTab,
  setChrome,
  setMessage,
  setTabbarVisible,
  statPill,
} from "./core/ui.js";
import {
  renderHelp,
  renderHome,
  renderLawParts,
  renderLearning,
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

// Wire up tab bar (bottom navigation)
refs.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    if (!target) return;
    if (state.currentView) {
      // refuse to navigate while in active session — protects user input
      impact("light");
      return;
    }
    if (target === state.currentScreen) return;
    navigate(target, { reset: true });
  });
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
    bindInlineTargets,
    cell,
    escapeHtml,
    group,
    statPill,
    setChrome,
    setMessage,
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
  if (!screen) return;

  if (screen.startsWith("admin") && !state.bootstrap?.user?.is_admin) {
    setMessage("error", "Режим адміністратора недоступний.");
    return;
  }

  if (options.reset) {
    state.screenHistory = [];
    state.currentScreen = screen;
  } else if (options.replace) {
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
  if (state.currentView) return;
  if (screen === "admin-users") void loadAdminUsers(createContext(), state.adminUsersOffset);
  if (screen === "admin-questions") void loadAdminQuestions(createContext(), state.adminQuestionsPage);
}

/**
 * Tab bar should be shown only on top-level screens with no active session.
 */
function shouldShowTabbar() {
  if (state.currentView) return false;
  const main = new Set(["home", "learning", "testing", "stats", "help"]);
  return main.has(state.currentScreen);
}

function render() {
  if (!state.bootstrap) {
    setChrome({ showBack: false });
    syncClosingConfirmation(state.currentView);
    refs.mainPanel.innerHTML = refs.emptyStateTemplate.innerHTML;
    setTabbarVisible(false);
    setActiveTab(null);
    return;
  }

  if (state.currentScreen.startsWith("admin") && !state.bootstrap.user.is_admin) {
    state.currentScreen = "home";
    state.screenHistory = [];
  }

  const ctx = createContext();

  if (state.currentView) {
    setTabbarVisible(false);
    setActiveTab(null);
    renderCurrentView(ctx);
    syncClosingConfirmation(state.currentView);
    return;
  }

  setTabbarVisible(shouldShowTabbar());
  setActiveTab(state.currentScreen);

  switch (state.currentScreen) {
    case "home":              renderHome(ctx); break;
    case "learning":          renderLearning(ctx); break;
    case "law-parts":         renderLawParts(ctx); break;
    case "testing":           renderTesting(ctx); break;
    case "stats":             renderStats(ctx); break;
    case "help":              renderHelp(ctx); break;
    case "admin":             renderAdminHub(ctx); break;
    case "admin-users":       renderAdminUsers(ctx); break;
    case "admin-questions":   renderAdminQuestions(ctx); break;
    default:
      state.currentScreen = "home";
      renderHome(ctx);
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

    if (showSuccess) setMessage("success", "Дані оновлено.");

    render();
    ensureScreenData();
  } catch (error) {
    setMessage("error", error.message);
    setChrome({ showBack: false });
    setTabbarVisible(false);
    refs.mainPanel.innerHTML = `
      <div class="screen-content">
        <div class="empty">
          <h2>Mini App не підключився</h2>
          <p>${escapeHtml(error.message)}</p>
        </div>
      </div>
    `;
  }
}

loadBootstrap();
