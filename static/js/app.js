import { refs } from "./core/dom.js?v=20260617-question-search-04";
import { state } from "./core/state.js?v=20260820-admin-users-05";
import { api } from "./core/api.js?v=20260617-question-search-04";
import { tg, initializeTelegram, impact, syncClosingConfirmation } from "./core/telegram.js?v=20260815-language-browse-03";
import { initializeTheme } from "./core/theme.js?v=20260809-prototype-01";
import {
  actionButton,
  bindInlineTargets,
  cell,
  escapeHtml,
  group,
  lineIcon,
  setChrome,
  setMessage,
  statPill,
} from "./core/ui.js?v=20260820-admin-users-05";
import {
  loadCaseDetail,
  loadCases,
  loadCustomsArticle,
  loadCustomsCode,
  loadCustomsSection,
  loadUserTestExamQuestions,
  renderCaseDetail,
  renderCases,
  renderAttestationParts,
  renderAttestationStage1,
  renderCustoms,
  renderCustomsArticle,
  renderCustomsCode,
  renderCustomsSection,
  renderHelp,
  renderHome,
  renderLawParts,
  renderLearning,
  renderOkLevels,
  renderOkQuestions,
  renderPaywall,
  renderPurchaseOptions,
  renderQuestionSearch,
  renderStats,
  renderTesting,
  renderTestExamQuestions,
} from "./screens/user.js?v=20260820-admin-users-05";
import {
  loadAdminCases,
  loadAdminAttestationBanks,
  loadAdminMessages,
  loadAdminQuestions,
  loadAdminTestQuestions,
  loadAdminUserDetail,
  loadAdminUsers,
  loadQuestionDetail,
  renderAdminCases,
  renderAdminAttestationBanks,
  renderAdminGlobalSearch,
  renderAdminHub,
  renderAdminMessages,
  renderAdminQuestionDetail,
  renderAdminQuestionView,
  renderAdminQuestions,
  renderAdminTestQuestions,
  renderAdminUserAccess,
  renderAdminUserDetail,
  renderAdminUsers,
  runQuestionSearch,
} from "./screens/admin.js?v=20260820-admin-users-05";
import { renderCurrentView } from "./screens/session.js?v=20260815-language-browse-03";
import {
  cleanupAdminApkImport,
  renderAdminApkImport,
} from "./admin_apk_import.js?v=20260812-apk-100mb-01";
import {
  loadAdminAttestationBank,
  loadAdminAttestationQuestion,
  renderAdminAttestationBank,
  renderAdminAttestationQuestion,
} from "./admin_attestation_banks.js?v=20260812-global-search-01";
import {
  loadAdminSectionTopics,
  renderAdminSection,
  renderAdminSectionOrder,
  renderAdminSectionSettings,
  renderAdminSectionTopics,
  renderAdminSectionTopicEdit,
} from "./admin_sections.js?v=20260812-order-sync-01";

const PROTOTYPE_SCREENS = new Set([
  "home",
  "customs",
  "learning",
  "law-parts",
  "ok-levels",
  "attestation-stage-1",
  "attestation-bank",
  "attestation-parts",
  "testing",
  "stats",
  "cases",
  "case-detail",
  "customs-code",
  "customs-code-section",
  "customs-code-article",
  "ok-questions",
  "test-exam-questions",
  "question-search",
  "purchase-options",
  "help",
  "admin",
  "admin-global-search",
  "admin-apk-import",
  "admin-users",
  "admin-messages",
  "admin-user-detail",
  "admin-user-access",
  "admin-questions",
  "admin-question-detail",
  "admin-question-view",
  "admin-cases",
  "admin-attestation-banks",
  "admin-attestation-question",
  "admin-section",
  "admin-section-settings",
  "admin-section-order",
  "admin-section-topics",
  "admin-section-topic-edit",
  "admin-section-questions",
  "admin-test-questions",
]);

window.__APP_READY__ = false;

const SCREEN_TRANSITIONS = new Set(["initial", "forward", "back", "replace"]);
let pendingScreenTransition = "initial";

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
    lineIcon,
    statPill,
    setChrome,
    setMessage,
    queueTransition,
    applyTransition,
    navigate,
    goHome,
    goBack,
    render,
    loadBootstrap,
    startLearning,
    startCustomsPreview,
    startMistakesSession,
    leaveCurrentView,
    openPayment,
    loadAdminUsers: (offset = state.adminUsersOffset) => loadAdminUsers(createContext(), offset),
    loadAdminUserDetail: (userId) => loadAdminUserDetail(createContext(), userId),
    loadAdminQuestions: (page = state.adminQuestionsPage) => loadAdminQuestions(createContext(), page),
    loadAdminCases: () => loadAdminCases(createContext()),
    loadAdminTestQuestions: (offset = state.testQOffset || 0) => loadAdminTestQuestions(createContext(), offset),
    loadQuestionDetail: (questionId) => loadQuestionDetail(createContext(), questionId),
    runQuestionSearch: (query) => runQuestionSearch(createContext(), query),
    loadCases: () => loadCases(createContext()),
    loadCaseDetail: (offset = state.caseOffset) => loadCaseDetail(createContext(), offset),
    loadCustomsCode: () => loadCustomsCode(createContext()),
    loadCustomsSection: () => loadCustomsSection(createContext()),
    loadCustomsArticle: () => loadCustomsArticle(createContext()),
    loadUserTestExamQuestions: (offset = state.testExamOffset || 0) => loadUserTestExamQuestions(createContext(), offset),
  };
}

function queueTransition(type = "replace") {
  pendingScreenTransition = SCREEN_TRANSITIONS.has(type) ? type : "replace";
}

function applyTransition() {
  refs.mainPanel.dataset.transition = pendingScreenTransition;
  pendingScreenTransition = "replace";
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
    queueTransition(screen === "home" ? "back" : "forward");
  } else if (options.replace) {
    state.currentScreen = screen;
    queueTransition("replace");
  } else if (screen !== state.currentScreen) {
    state.screenHistory.push(state.currentScreen);
    state.currentScreen = screen;
    queueTransition("forward");
  } else {
    queueTransition("replace");
  }

  impact("light");
  render();
  ensureScreenData(screen);
}

function goHome() {
  state.screenHistory = [];
  state.currentScreen = "home";
  queueTransition("back");
  render();
}

async function goBack() {
  if (state.currentView) {
    if (state.currentView.screen === "open-practice-detail") {
      state.currentView = null;
      queueTransition("back");
      render();
      return;
    }

    if (state.currentView.screen === "review") {
      try {
        state.currentView = await api("/api/test/review/back", { method: "POST" });
        queueTransition("back");
        render();
      } catch (error) {
        setMessage("error", error.message);
      }
      return;
    }

    if (state.currentView.screen === "result") {
      try {
        await api("/api/session/leave", { method: "POST" });
      } catch (_) {}
      state.currentView = null;
      queueTransition("back");
      await loadBootstrap();
      return;
    }

    await leaveCurrentView();
    return;
  }

  if (state.currentScreen === "admin-apk-import") {
    await cleanupAdminApkImport();
  }

  if (state.screenHistory.length) {
    const previous = state.screenHistory.pop();
    state.currentScreen = previous || "home";
    if (previous !== "law-parts") {
      state.selectedLawGroup = null;
    }
    if (previous !== "attestation-parts") {
      state.selectedAttestationSection = null;
    }
    if (previous !== "customs-code-section") {
      state.customsSectionDetail = null;
    }
    if (previous !== "customs-code-article") {
      state.customsArticle = null;
    }
    queueTransition("back");
    render();
    ensureScreenData();
    return;
  }

  if (state.currentScreen !== "home") {
    state.currentScreen = "home";
    queueTransition("back");
    render();
  }
}

function ensureScreenData(screen = state.currentScreen) {
  if (state.currentView) return;
  if (screen === "admin-users") void loadAdminUsers(createContext(), state.adminUsersOffset);
  if (screen === "admin-messages") void loadAdminMessages(createContext(), state.adminUsersOffset);
  if (screen === "admin-user-detail") void loadAdminUserDetail(createContext(), state.selectedAdminUserId);
  if (screen === "admin-user-access") void loadAdminUserDetail(createContext(), state.selectedAdminUserId);
  if (screen === "admin-questions") void loadAdminQuestions(createContext(), state.adminQuestionsPage);
  if (screen === "admin-cases") void loadAdminCases(createContext());
  if (screen === "admin-attestation-banks") void loadAdminAttestationBanks(createContext());
  if (screen === "admin-attestation-question") void loadAdminAttestationQuestion(createContext());
  if (screen === "admin-section-topics") void loadAdminSectionTopics(createContext());
  if (screen === "admin-section-questions") void loadAdminAttestationBank(createContext(), state.attestationAdminOffset || 0);
  if (screen === "admin-test-questions") void loadAdminTestQuestions(createContext(), state.testQOffset || 0);
  if (screen === "test-exam-questions") void loadUserTestExamQuestions(createContext(), state.testExamOffset || 0);
  if (screen === "cases") void loadCases(createContext());
  if (screen === "case-detail") void loadCaseDetail(createContext(), state.caseOffset);
  if (screen === "customs-code") void loadCustomsCode(createContext());
  if (screen === "customs-code-section") void loadCustomsSection(createContext());
  if (screen === "customs-code-article") void loadCustomsArticle(createContext());
}

function decoratePrototypeScreen() {
  const enabled = Boolean(state.currentView) || PROTOTYPE_SCREENS.has(state.currentScreen);
  document.body.classList.toggle("ui-prototype", enabled);
  refs.mainPanel.dataset.screen = state.currentView ? `session-${state.currentView.screen || state.currentView.mode || "active"}` : state.currentScreen;

  if (!enabled) return;
  // Test sessions start directly with progress and the question. Telegram already
  // provides the native top bar, so an additional in-app hero is unnecessary.
  if (state.currentView) return;

  const content = refs.mainPanel.querySelector(".screen-content");
  if (!content || content.querySelector(":scope > .page-hero")) return;

  const caseHeader = content.querySelector(":scope > .case-header");
  if (caseHeader) {
    const appLabel = caseHeader.querySelector(".case-header__app");
    appLabel?.remove();
    caseHeader.classList.add("page-hero", "page-hero--case");
    return;
  }

  const title = content.querySelector(":scope > .page-title");
  const subtitle = content.querySelector(":scope > .page-subtitle");

  if (!title) return;

  const hero = document.createElement("header");
  hero.className = "page-hero";
  subtitle?.remove();
  hero.append(title);
  content.prepend(hero);
}

function render() {
  applyTransition();

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
    decoratePrototypeScreen();
    syncClosingConfirmation(state.currentView);
    return;
  }

  switch (state.currentScreen) {
    case "home":              renderHome(ctx); break;
    case "attestation-stage-1": renderAttestationStage1(ctx); break;
    case "attestation-bank":    renderAttestationStage1(ctx); break;
    case "attestation-parts": renderAttestationParts(ctx); break;
    case "learning":          renderLearning(ctx); break;
    case "law-parts":         renderLawParts(ctx); break;
    case "customs":           renderCustoms(ctx); break;
    case "customs-code":      renderCustomsCode(ctx); break;
    case "customs-code-section": renderCustomsSection(ctx); break;
    case "customs-code-article": renderCustomsArticle(ctx); break;
    case "cases":             renderCases(ctx); break;
    case "case-detail":       renderCaseDetail(ctx); break;
    case "ok-levels":              renderOkLevels(ctx); break;
    case "ok-questions":          renderOkQuestions(ctx); break;
    case "question-search":       renderQuestionSearch(ctx); break;
    case "test-exam-questions":   renderTestExamQuestions(ctx); break;
    case "testing":               renderTesting(ctx); break;
    case "stats":                 renderStats(ctx); break;
    case "help":                  renderHelp(ctx); break;
    case "purchase-options":      renderPurchaseOptions(ctx); break;
    case "admin":                  renderAdminHub(ctx); break;
    case "admin-global-search":    renderAdminGlobalSearch(ctx); break;
    case "admin-apk-import":       renderAdminApkImport(ctx); break;
    case "admin-users":            renderAdminUsers(ctx); break;
    case "admin-messages":         renderAdminMessages(ctx); break;
    case "admin-user-detail":      renderAdminUserDetail(ctx); break;
    case "admin-user-access":      renderAdminUserAccess(ctx); break;
    case "admin-questions":        renderAdminQuestions(ctx); break;
    case "admin-question-detail":  renderAdminQuestionDetail(ctx); break;
    case "admin-question-view":    renderAdminQuestionView(ctx); break;
    case "admin-cases":            renderAdminCases(ctx); break;
    case "admin-attestation-banks": renderAdminAttestationBanks(ctx); break;
    case "admin-attestation-question": renderAdminAttestationQuestion(ctx); break;
    case "admin-section": renderAdminSection(ctx); break;
    case "admin-section-settings": renderAdminSectionSettings(ctx); break;
    case "admin-section-order": renderAdminSectionOrder(ctx); break;
    case "admin-section-topics": renderAdminSectionTopics(ctx); break;
    case "admin-section-topic-edit": renderAdminSectionTopicEdit(ctx); break;
    case "admin-section-questions": renderAdminAttestationBank(ctx); break;
    case "admin-test-questions": renderAdminTestQuestions(ctx); break;
    default:
      state.currentScreen = "home";
      renderHome(ctx);
  }

  decoratePrototypeScreen();
  syncClosingConfirmation(state.currentView);
}

async function leaveCurrentView() {
  try {
    await api("/api/session/leave", { method: "POST" });
  } catch (error) {
    setMessage("error", error.message);
  }
  state.currentView = null;
  queueTransition("back");
  await loadBootstrap();
}

async function startLearning(payload) {
  try {
    state.currentView = await api("/api/learning/start", { method: "POST", body: payload });
    impact("medium");
    queueTransition("forward");
    render();
  } catch (error) {
    if (error.code === "access_expired" || error.code === "cases_access_required" || error.code === "protected_materials_required") {
      queueTransition("forward");
      renderPaywall(createContext(), error.code);
      return;
    }
    setMessage("error", error.message);
  }
}

async function startCustomsPreview() {
  try {
    state.currentView = await api("/api/learning/preview/start", { method: "POST" });
    impact("medium");
    queueTransition("forward");
    render();
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function startMistakesSession() {
  try {
    state.currentView = await api("/api/mistakes/start", { method: "POST" });
    impact("medium");
    queueTransition("forward");
    render();
  } catch (error) {
    setMessage("error", error.message);
  }
}

async function openPayment(target) {
  try {
    const body = typeof target === "string" ? { tier: target } : target;
    const { invoice_link } = await api("/api/payment/create-link", {
      method: "POST",
      body,
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

    const openAttestationBank = window.sessionStorage.getItem("openAttestationBank");
    if (!state.currentView && openAttestationBank
        && (payload.catalog.attestation_banks || []).some((bank) => bank.slug === openAttestationBank)) {
      window.sessionStorage.removeItem("openAttestationBank");
      state.selectedAttestationBankSlug = openAttestationBank;
      state.selectedAttestationSection = null;
      state.currentScreen = "attestation-bank";
      state.screenHistory = ["home"];
    }

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
