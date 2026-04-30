import { refs } from "./dom.js";
import { tg } from "./telegram.js";

const STORAGE_KEY = "prep-app-theme";
const THEMES = new Set(["light", "dark"]);

function normalizeTheme(value) {
  return THEMES.has(value) ? value : "light";
}

function readStoredTheme() {
  try {
    const value = window.localStorage?.getItem(STORAGE_KEY);
    return THEMES.has(value) ? value : null;
  } catch {
    return null;
  }
}

function writeStoredTheme(theme) {
  try {
    window.localStorage?.setItem(STORAGE_KEY, theme);
  } catch {
    // Storage can be disabled inside some embedded browsers.
  }
}

function getPreferredTheme() {
  if (tg?.colorScheme === "dark") {
    return "dark";
  }

  if (tg?.colorScheme === "light") {
    return "light";
  }

  if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return "dark";
  }

  return "light";
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function updateTelegramChrome() {
  if (!tg) {
    return;
  }

  const background = cssVar("--telegram-bg") || cssVar("--app-bg");
  const header = cssVar("--telegram-header") || cssVar("--surface-strong");

  tg.setBackgroundColor?.(background);
  tg.setHeaderColor?.(header);
}

function updateThemeButton(theme) {
  const button = refs.themeButton;
  if (!button) {
    return;
  }

  const isDark = theme === "dark";
  button.setAttribute("aria-label", isDark ? "Увімкнути світлу тему" : "Увімкнути чорну тему");
  button.title = isDark ? "Світла тема" : "Чорна тема";

  const label = button.querySelector(".theme-button__label");
  if (label) {
    label.textContent = isDark ? "Світла" : "Чорна";
  }

  const icon = button.querySelector("svg");
  if (icon) {
    icon.innerHTML = isDark
      ? '<path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.41M17.66 6.34l1.41-1.41" /><circle cx="12" cy="12" r="4" />'
      : '<path d="M12 3a6 6 0 0 0 9 7.5A9 9 0 1 1 12 3Z" />';
  }
}

export function applyTheme(value, { persist = false } = {}) {
  const theme = normalizeTheme(value);
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  document.body.classList.toggle("is-dark", theme === "dark");
  updateThemeButton(theme);
  updateTelegramChrome();

  if (persist) {
    writeStoredTheme(theme);
  }
}

export function initializeTheme() {
  const storedTheme = readStoredTheme();
  applyTheme(storedTheme || getPreferredTheme());

  refs.themeButton?.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(nextTheme, { persist: true });
  });

  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", (event) => {
    if (!readStoredTheme()) {
      applyTheme(event.matches ? "dark" : "light");
    }
  });

  tg?.onEvent?.("themeChanged", () => {
    if (!readStoredTheme()) {
      applyTheme(getPreferredTheme());
      return;
    }

    updateTelegramChrome();
  });
}
