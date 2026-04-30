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
    /* embedded browsers may block storage */
  }
}

function getPreferredTheme() {
  if (tg?.colorScheme === "dark") return "dark";
  if (tg?.colorScheme === "light") return "light";
  if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) return "dark";
  return "light";
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function updateTelegramChrome() {
  if (!tg) return;
  const background = cssVar("--bg") || "#ffffff";
  tg.setBackgroundColor?.(background);
  tg.setHeaderColor?.(background);
}

export function applyTheme(value, { persist = false } = {}) {
  const theme = normalizeTheme(value);
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  updateTelegramChrome();
  if (persist) writeStoredTheme(theme);
}

export function getCurrentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export function toggleTheme() {
  const next = getCurrentTheme() === "dark" ? "light" : "dark";
  applyTheme(next, { persist: true });
  return next;
}

export function initializeTheme() {
  const stored = readStoredTheme();
  applyTheme(stored || getPreferredTheme());

  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", (event) => {
    if (!readStoredTheme()) applyTheme(event.matches ? "dark" : "light");
  });

  tg?.onEvent?.("themeChanged", () => {
    if (!readStoredTheme()) {
      applyTheme(getPreferredTheme());
    } else {
      updateTelegramChrome();
    }
  });
}
