export const tg = window.Telegram?.WebApp || null;

let telegramBackAttached = false;
let closingConfirmationEnabled = false;

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

export function initializeTelegram(onBack) {
  if (!tg) {
    return;
  }

  tg.ready();
  tg.expand();
  applyTelegramTheme();

  if (!telegramBackAttached) {
    tg.BackButton?.onClick?.(() => {
      onBack();
    });
    telegramBackAttached = true;
  }

  tg.onEvent?.("themeChanged", applyTelegramTheme);
}

export function impact(style = "light") {
  tg?.HapticFeedback?.impactOccurred?.(style);
}

export function setTelegramBackButton(showBack) {
  if (showBack) {
    tg?.BackButton?.show?.();
    return;
  }
  tg?.BackButton?.hide?.();
}

export function syncClosingConfirmation(view) {
  const shouldProtect = Boolean(
    view && (view.mode === "pretest" || view.screen === "question" || view.screen === "feedback"),
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
