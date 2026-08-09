import { tg } from "./telegram.js?v=20260523-cases-search-02";

const STORAGE_KEY = "prep-app-theme";
const THEMES = new Set(["light", "dark"]);

function normalizeTheme(value) {
  return "light";
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
    /* embedded browsers may block storage */
  }
}

function getPreferredTheme() {
  return "light";
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function updateTelegramChrome() {
  if (!tg) return;
  const background = cssVar("--bg") || "#ffffff";
  const header = cssVar("--navy") || "#10233f";
  tg.setBackgroundColor?.(background);
  tg.setHeaderColor?.(header);
}

export function applyTheme(value, { persist = false } = {}) {
  const theme = normalizeTheme(value);
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  updateTelegramChrome();
  if (persist) writeStoredTheme(theme);
}

export function getCurrentTheme() {
  return "light";
}

export function toggleTheme() {
  applyTheme("light", { persist: true });
  return "light";
}

export function initializeTheme() {
  applyTheme("light", { persist: true });

  tg?.onEvent?.("themeChanged", () => {
    updateTelegramChrome();
  });
}
