import { refs } from "./core/dom.js?v=20260523-cases-search-01";
import { state } from "./core/state.js?v=20260523-cases-search-01";
import { api } from "./core/api.js?v=20260523-cases-search-01";
import { tg, initializeTelegram, impact, syncClosingConfirmation } from "./core/telegram.js?v=20260523-cases-search-01";
import { initializeTheme } from "./core/theme.js?v=20260523-cases-search-01";
import {
  actionButton,
  bindInlineTargets,
  cell,
  escapeHtml,
  group,
  setChrome,
  setMessage,
  statPill,
} from "./core/ui.js?v=20260523-cases-search-01";
import {
  loadCaseDetail,
  loadCases,
  loadCustomsArticle,
  loadCustomsCode,
  loadCustomsSection,
  renderCaseDetail,
  renderCases,
  renderCustoms,
  renderCustomsArticle,
  renderCustomsCode,
  renderCustomsSection,
  renderHelp,
  renderHome,
  renderLawParts,
  renderLearning,
  renderPaywall,
  renderStats,
  renderTesting,
} from "./screens/user.js?v=20260523-cases-search-01";
import {
  loadAdminCases,
  loadAdminQuestions,
  loadAdminSettings,
  loadAdminUserDetail,
  loadAdminUsers,
  loadQuestionDetail,
  renderAdminCases,
  renderAdminHub,
  renderAdminQuestions,
  renderAdminSettings,
  renderAdminUsers,
  runQuestionSearch,
} from "./screens/admin.js?v=20260523-cases-search-01";
import { renderCurrentView } from "./screens/session.js?v=20260523-cases-search-01";

window.__APP_READY__ = false;

initializeTelegram(() => {
  void goBack();
});
initializeTheme();

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
    openPayment,
    loadAdminUsers: (offset = state.adminUsersOffset) => loadAdminUsers(createContext(), offset),
    loadAdminUserDetail: (userId) => loadAdminUserDetail(createContext(), userId),
    loadAdminQuestions: (page = state.adminQuestionsPage) => loadAdminQuestions(createContext(), page),
    loadAdminCases: () => loadAdminCases(createContext()),
    loadAdminSettings: () => loadAdminSettings(createContext()),
    loadQuestionDetail: (questionId) => loadQuestionDetail(createContext(), questionId),
    runQuestionSearch: (query) => runQuestionSearch(createContext(), query),
    loadCases: () => loadCases(createContext()),
    loadCaseDetail: (offset = state.caseOffset) => loadCaseDetail(createContext(), offset),
    loadCustomsCode: () => loadCustomsCode(createContext()),
    loadCustomsSection: () => loadCustomsSection(createContext()),
    loadCustomsArticle: () => loadCustomsArticle(createContext()),
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
    if (previous !== "customs-code-section") {
      state.customsSectionDetail = null;
    }
    if (previous !== "customs-code-article") {
      state.customsArticle = null;
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
  if (screen === "admin-cases") void loadAdminCases(createContext());
  if (screen === "admin-settings") void loadAdminSettings(createContext());
  if (screen === "cases") void loadCases(createContext());
  if (screen === "case-detail") void loadCaseDetail(createContext(), state.caseOffset);
  if (screen === "customs-code") void loadCustomsCode(createContext());
  if (screen === "customs-code-section") void loadCustomsSection(createContext());
  if (screen === "customs-code-article") void loadCustomsArticle(createContext());
}

function render() {
  if (!state.bootstrap) {
    setChrome({ showBack: false });
    syncClosingConfirmation(state.currentView);
    refs.mainPanel.innerHTML = refs.emptyStateTemplate.innerHTML;
    return;
  }

  if (state.currentScreen.startsWith("admin") && !state.bootstrap.user.is_admin) {
    state.currentScreen = "home";
    state.screenHistory = [];
  }

  const ctx = createContext();

  window.scrollTo({ top: 0, behavior: "instant" });

  if (state.currentView) {
    renderCurrentView(ctx);
    syncClosingConfirmation(state.currentView);
    return;
  }

  switch (state.currentScreen) {
    case "home":              renderHome(ctx); break;
    case "learning":          renderLearning(ctx); break;
    case "law-parts":         renderLawParts(ctx); break;
    case "customs":           renderCustoms(ctx); break;
    case "customs-code":      renderCustomsCode(ctx); break;
    case "customs-code-section": renderCustomsSection(ctx); break;
    case "customs-code-article": renderCustomsArticle(ctx); break;
    case "cases":             renderCases(ctx); break;
    case "case-detail":       renderCaseDetail(ctx); break;
    case "testing":           renderTesting(ctx); break;
    case "stats":             renderStats(ctx); break;
    case "help":              renderHelp(ctx); break;
    case "admin":             renderAdminHub(ctx); break;
    case "admin-users":       renderAdminUsers(ctx); break;
    case "admin-questions":   renderAdminQuestions(ctx); break;
    case "admin-cases":       renderAdminCases(ctx); break;
    case "admin-settings":    renderAdminSettings(ctx); break;
    default:
      state.currentScreen = "home";
      renderHome(ctx);
  }

  const content = refs.mainPanel.querySelector(".screen-content");
  if (content) {
    content.style.animation = "none";
    requestAnimationFrame(() => {
      content.style.animation = "";
    });

    if (state.currentScreen !== "home" && state.screenHistory.length > 0) {
      const nav = document.createElement("button");
      nav.className = "back-nav";
      nav.type = "button";
      nav.innerHTML = '<span aria-hidden="true">‹</span> Назад';
      nav.addEventListener("click", () => void goBack());
      content.prepend(nav);
    }
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
    if (error.code === "access_expired" || error.code === "cases_access_required") {
      renderPaywall(createContext(), error.code);
      return;
    }
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

async function openPayment(tier) {
  try {
    const { invoice_link } = await api("/api/payment/create-link", {
      method: "POST",
      body: { tier },
    });
    if (tg?.openInvoice) {
      tg.openInvoice(invoice_link, async (status) => {
        if (status === "paid") {
          await loadBootstrap();
          setMessage("success", "Оплата успішна! Доступ активовано.");
        }
      });
    } else {
      window.open(invoice_link, "_blank");
    }
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function loadBootstrap(showSuccess = false) {
  try {
    const payload = await api("/api/bootstrap", { timeoutMs: 12000 });
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
    refs.mainPanel.innerHTML = `
      <div class="screen-content">
        <div class="empty">
          <h2>Mini App не підключився</h2>
          <p>${escapeHtml(error.message)}</p>
        </div>
      </div>
    `;
  } finally {
    window.__APP_READY__ = true;
  }
}

loadBootstrap();
