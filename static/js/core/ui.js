import { refs } from "./dom.js?v=20260523-cases-search-02";
import { impact, setTelegramBackButton, tg } from "./telegram.js?v=20260815-language-browse-03";

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function emptyState(title, text, { inline = false } = {}) {
  return `
    <div class="empty${inline ? " empty--inline" : ""}">
      <h2>${escapeHtml(title)}</h2>
      ${text ? `<p>${escapeHtml(text)}</p>` : ""}
    </div>
  `;
}

/**
 * Compact stat pill (used inside .stat-strip).
 */
export function statPill(label, value) {
  return `
    <div class="stat-pill">
      <span class="stat-pill__value">${escapeHtml(value)}</span>
      <span class="stat-pill__label">${escapeHtml(label)}</span>
    </div>
  `;
}

const CELL_ICONS = {
  support: '<path d="M7 10.5h10M7 14h6"/><path d="M21 12a8 8 0 0 1-8 8H7l-4 3v-7.5A8 8 0 1 1 21 12Z"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4.5 21a7.5 7.5 0 0 1 15 0"/>',
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06-2.83 2.83-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21h-4v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06-2.83-2.83.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3v-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06 2.83-2.83.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3h4v.09A1.65 1.65 0 0 0 15 4.6a1.65 1.65 0 0 0 1.82-.33l.06-.06 2.83 2.83-.06.06A1.65 1.65 0 0 0 19.32 9a1.65 1.65 0 0 0 1.51 1H21v4h-.09A1.65 1.65 0 0 0 19.4 15Z"/>',
  home: '<path d="m3 11 9-8 9 8"/><path d="M5 10v11h14V10M9 21v-7h6v7"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z"/>',
  folder: '<path d="M3 5h6l2 2h10v12H3Z"/>',
  document: '<path d="M6 2h9l5 5v15H6Z"/><path d="M14 2v6h6M9 13h8M9 17h8"/>',
  graduation: '<path d="m2 9 10-5 10 5-10 5Z"/><path d="M6 11.5V16c3 3 9 3 12 0v-4.5M22 9v6"/>',
  scale: '<path d="M12 3v18M5 6h14M7 6l-4 7h8L7 6ZM17 6l-4 7h8l-4-7ZM7 21h10"/>',
  clipboard: '<path d="M9 4h6l1 3H8l1-3Z"/><path d="M8 5H5v17h14V5h-3M8 12h8M8 16h8"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
  calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>',
};

export function lineIcon(name) {
  const paths = CELL_ICONS[name];
  if (!paths) return "";
  return `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${paths}</svg>`;
}

/**
 * iOS Settings-style cell.
 * @param {Object} opts
 * @param {string} opts.title
 * @param {string} [opts.subtitle]
 * @param {string} [opts.detail]      right-aligned secondary text
 * @param {string} [opts.icon]        emoji or short text
 * @param {string} [opts.iconName]    name of a built-in line icon
 * @param {string} [opts.tint]        blue|purple|pink|orange|green|red|teal|indigo|gray
 * @param {string} [opts.screen]      navigation target screen name
 * @param {string} [opts.link]        external link target
 * @param {boolean}[opts.chevron]     whether to show chevron arrow (default true if screen/link)
 * @param {string} [opts.variant]     "default" | "destructive" | "accent" | "plain"
 */
export function cell(opts = {}) {
  const {
    title,
    subtitle,
    detail,
    icon,
    iconName,
    tint = "gray",
    screen,
    link,
    chevron,
    variant = "default",
  } = opts;

  const showChevron = chevron ?? Boolean(screen || link);
  const iconMarkup = iconName ? lineIcon(iconName) : escapeHtml(icon || "");
  const variantClass = variant === "default" ? "" : ` cell--${variant}`;
  const attr = screen
    ? `data-screen-target="${escapeHtml(screen)}"`
    : link
      ? `data-link-target="${escapeHtml(link)}"`
      : "";

  return `
    <button class="cell${variantClass}" type="button" ${attr}>
      ${iconMarkup ? `<span class="cell__icon cell__icon--${escapeHtml(tint)}">${iconMarkup}</span>` : ""}
      <span class="cell__body">
        <span class="cell__title">${escapeHtml(title)}</span>
        ${subtitle ? `<span class="cell__subtitle">${escapeHtml(subtitle)}</span>` : ""}
      </span>
      ${detail ? `<span class="cell__detail">${escapeHtml(detail)}</span>` : ""}
      ${showChevron ? `<span class="cell__chevron" aria-hidden="true"></span>` : ""}
    </button>
  `;
}

/**
 * Group of cells with optional header/footer (iOS Settings style).
 */
export function group({ header, footer, children, variant }) {
  const variantClass = variant ? ` group--${escapeHtml(variant)}` : "";
  return `
    <section class="group${variantClass}">
      ${header ? `<div class="group__label">${escapeHtml(header)}</div>` : ""}
      <div class="group__list">${children}</div>
      ${footer ? `<div class="group__footer">${escapeHtml(footer)}</div>` : ""}
    </section>
  `;
}

/**
 * Build a button element with handler attached.
 */
export function actionButton(label, handler, variant = "default") {
  const button = document.createElement("button");
  button.type = "button";
  const cls = ["btn"];
  if (variant === "primary") cls.push("btn--primary");
  if (variant === "danger" || variant === "destructive") cls.push("btn--destructive");
  if (variant === "ghost") cls.push("btn--ghost");
  if (variant === "block") cls.push("btn--primary", "btn--block");
  if (variant === "block-ghost") cls.push("btn--ghost", "btn--block");
  if (variant === "block-danger") cls.push("btn--destructive", "btn--block");
  if (variant === "lg") cls.push("btn--lg");
  if (variant === "sm") cls.push("btn--sm");

  button.className = cls.join(" ");
  button.textContent = label;
  button.addEventListener("click", async () => {
    impact(variant === "primary" || variant === "block" ? "medium" : "light");
    await handler();
  });
  return button;
}

/**
 * Sets the Telegram back-button visibility (we don't render our own top bar).
 */
export function setChrome({ showBack = false } = {}) {
  setTelegramBackButton(showBack);
  if (refs.titleNode) refs.titleNode.textContent = "Підготовка";
  if (refs.eyebrowNode) refs.eyebrowNode.textContent = "";
  if (refs.subtitleNode) refs.subtitleNode.textContent = "";
}

let messageDismissTimer = null;

/**
 * Show a status message inside the visible Mini App area.
 */
export function setMessage(kind, text) {
  if (messageDismissTimer) {
    clearTimeout(messageDismissTimer);
    messageDismissTimer = null;
  }

  if (!text) {
    refs.messagesPanel.hidden = true;
    refs.messagesPanel.innerHTML = "";
    return;
  }

  refs.messagesPanel.hidden = false;
  refs.messagesPanel.innerHTML = `
    <div class="message message--${kind}">
      <div class="message__body">${escapeHtml(text)}</div>
      <button class="message__close" type="button" id="dismiss-message" aria-label="Закрити">×</button>
    </div>
  `;
  document.querySelector("#dismiss-message")?.addEventListener("click", () => setMessage("", ""));
  // auto-dismiss success messages
  if (kind === "success") {
    messageDismissTimer = setTimeout(() => setMessage("", ""), 2400);
  }
}

export function openExternalLink(url) {
  if (!url) return;
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "tg:") {
      window.open(url, "_blank");
      return;
    }
    if (tg?.openTelegramLink && parsed.hostname.endsWith("t.me")) {
      tg.openTelegramLink(url);
      return;
    }
    if (tg?.openLink) {
      tg.openLink(url);
      return;
    }
  } catch {
    /* fall through */
  }
  window.open(url, "_blank", "noopener,noreferrer");
}

/**
 * Wire up [data-screen-target] / [data-link-target] click handlers.
 */
export function bindInlineTargets(root, { navigate }) {
  const scope = root || refs.mainPanel;
  scope.querySelectorAll("[data-screen-target]").forEach((node) => {
    node.addEventListener("click", () => {
      const target = node.dataset.screenTarget;
      if (target) navigate(target);
    });
  });

  scope.querySelectorAll("[data-link-target]").forEach((node) => {
    node.addEventListener("click", () => {
      const target = node.dataset.linkTarget;
      if (target) {
        impact("light");
        openExternalLink(target);
      }
    });
  });
}
