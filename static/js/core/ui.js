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

export function emptyStateMarkup(title, text) {
  return `
    <div class="empty-state">
      <h2>${escapeHtml(title)}</h2>
      <p>${escapeHtml(text)}</p>
    </div>
  `;
}

export function metricCard(label, value, note = "") {
  return `
    <article class="stat-card">
      <span class="eyebrow">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${note ? `<p class="muted">${escapeHtml(note)}</p>` : ""}
    </article>
  `;
}

export function actionButton(label, handler, kind = "secondary") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `${kind}-button`;
  button.textContent = label;
  button.addEventListener("click", async () => {
    impact(kind === "primary" ? "medium" : "light");
    await handler();
  });
  return button;
}

export function actionCard({ code, title, body, meta, screen, link, tone = "default" }) {
  const isDisabled = !screen && !link;
  const attr = screen
    ? `data-screen-target="${escapeHtml(screen)}"`
    : `data-link-target="${escapeHtml(link || "")}"`;
  const toneClass = tone ? ` nav-card--${escapeHtml(tone)}` : "";

  return `
    <button class="nav-card${toneClass}" type="button" ${attr} ${isDisabled ? "disabled" : ""}>
      <span class="monogram">${escapeHtml(code)}</span>
      <span class="nav-card__content">
        <span class="nav-card__meta">${escapeHtml(meta)}</span>
        <span class="nav-card__title">${escapeHtml(title)}</span>
        ${body ? `<span class="nav-card__text">${escapeHtml(body)}</span>` : ""}
      </span>
      <span class="nav-card__chevron" aria-hidden="true">›</span>
    </button>
  `;
}

export function setChrome({ eyebrow, title, subtitle, showBack = false, showRefresh = false }) {
  if (refs.eyebrowNode) {
    refs.eyebrowNode.textContent = eyebrow || "";
    refs.eyebrowNode.hidden = true;
  }

  if (refs.titleNode) {
    refs.titleNode.textContent = title || eyebrow || "Підготовка";
  }

  if (refs.subtitleNode) {
    refs.subtitleNode.textContent = subtitle || "";
    refs.subtitleNode.hidden = true;
  }

  if (refs.backButton) {
    refs.backButton.hidden = !showBack;
  }

  if (refs.refreshButton) {
    refs.refreshButton.hidden = !showRefresh;
  }

  setTelegramBackButton(showBack);
}

export function setMessage(kind, text) {
  if (!text) {
    refs.messagesPanel.hidden = true;
    refs.messagesPanel.innerHTML = "";
    return;
  }

  refs.messagesPanel.hidden = false;
  refs.messagesPanel.innerHTML = `
    <div class="message message--${kind}">
      <div>${escapeHtml(text)}</div>
      <button class="ghost-button" type="button" id="dismiss-message">Закрити</button>
    </div>
  `;
  document.querySelector("#dismiss-message")?.addEventListener("click", () => setMessage("", ""));
}

export function openExternalLink(url) {
  if (!url) {
    return;
  }

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
    // Fall back to window.open below.
  }

  window.open(url, "_blank", "noopener,noreferrer");
}

export function bindInlineTargets(root = refs.mainPanel, { navigate }) {
  root.querySelectorAll("[data-screen-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.screenTarget;
      if (target) {
        navigate(target);
      }
    });
  });

  root.querySelectorAll("[data-link-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.linkTarget;
      if (target) {
        impact("light");
        openExternalLink(target);
      }
    });
  });
}
