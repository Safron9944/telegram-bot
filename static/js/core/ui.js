import { refs } from "./dom.js";
import { impact, setTelegramBackButton, tg } from "./telegram.js";

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

/**
 * iOS Settings-style cell.
 * @param {Object} opts
 * @param {string} opts.title
 * @param {string} [opts.subtitle]
 * @param {string} [opts.detail]      right-aligned secondary text
 * @param {string} [opts.icon]        emoji or short text
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
    tint = "gray",
    screen,
    link,
    chevron,
    variant = "default",
  } = opts;

  const showChevron = chevron ?? Boolean(screen || link);
  const variantClass = variant === "default" ? "" : ` cell--${variant}`;
  const attr = screen
    ? `data-screen-target="${escapeHtml(screen)}"`
    : link
      ? `data-link-target="${escapeHtml(link)}"`
      : "";

  return `
    <button class="cell${variantClass}" type="button" ${attr}>
      ${icon ? `<span class="cell__icon cell__icon--${escapeHtml(tint)}">${escapeHtml(icon)}</span>` : ""}
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

/**
 * Show a status message at the top of the screen.
 */
export function setMessage(kind, text) {
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
    setTimeout(() => setMessage("", ""), 2400);
  }
}

export function openExternalLink(url) {
  if (!url) return;
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

/**
 * Set the active tab in the bottom tab bar.
 * Tabs are: home | learning | testing | stats | help.
 * Other screens (admin*, law-parts) keep no tab active.
 */
export function setActiveTab(screen) {
  const main = ["home", "learning", "testing", "stats", "help"];
  const active = main.includes(screen) ? screen : null;
  refs.tabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.tab === active);
  });
}

/**
 * Hide tab bar in active session / drill-down flows.
 */
export function setTabbarVisible(visible) {
  if (!refs.tabbar || !refs.app) return;
  refs.tabbar.hidden = !visible;
  refs.app.classList.toggle("tabbar-hidden", !visible);
}
