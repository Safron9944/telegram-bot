
let caseSearchTimer = null;
let caseDetailRequestId = 0;
let casesSearchRequestId = 0;
let customsSearchTimer = null;
let okSearchTimer = null;
let okSearchRequestId = 0;

/* ===================== HELPERS ===================== */
function selectedModules(catalog) {
  return catalog.ok_modules.filter((item) => item.selected);
}

function percentLabel(value) {
  return typeof value === "number" ? `${value.toFixed(0)}%` : "—";
}

function initialOf(name) {
  return (name || "U").trim().slice(0, 1).toUpperCase();
}

const MENU_ICONS = {
  graduation: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 9 12 4l9 5-9 5-9-5Z"/><path d="M7 11.2V16c0 1.7 2.2 3 5 3s5-1.3 5-3v-4.8"/><path d="M21 10v5"/></svg>`,
  document: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3h7l4 4v14H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z"/><path d="M14 3v5h5"/><path d="M9 12h6M9 16h4"/></svg>`,
  folder: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 7.5h6l2 2h10v8.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7.5Z"/><path d="M3 7.5V6a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v1.5"/></svg>`,
  scale: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3v18M5 7h14M7 7l-4 7h8L7 7ZM17 7l-4 7h8l-4-7Z"/><path d="M8 21h8"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/></svg>`,
  support: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 13v-1a8 8 0 0 1 16 0v1"/><path d="M4 13h2a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a1 1 0 0 1-1-1v-6ZM20 13h-2a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1"/><path d="M19 20c0 1-1 1-2 1h-3"/></svg>`,
  books: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 4h11a2 2 0 0 1 2 2v14H7a2 2 0 0 1-2-2V4Z"/><path d="M8 4v16M18 17H8"/></svg>`,
  flask: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 18l-5-9V3"/><path d="M7.5 15h9"/></svg>`,
  chart: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>`,
  clipboard: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M8 5H6a2 2 0 0 0-2 2v13h16V7a2 2 0 0 0-2-2h-2"/><rect x="8" y="3" width="8" height="4" rx="2"/><path d="M8 12h8M8 16h5"/></svg>`,
  settings: `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/></svg>`,
};

function menuIcon(name) {
  return MENU_ICONS[name] || MENU_ICONS.document;
}

function prototypeMenuCell(ctx, { title, icon, screen, detail = "", sectionKey = "" }) {
  return `
    <button class="cell menu-cell" type="button" data-screen-target="${ctx.escapeHtml(screen)}"${sectionKey ? ` data-section-key="${ctx.escapeHtml(sectionKey)}"` : ""}>
      <span class="cell__icon menu-cell__icon">${menuIcon(icon)}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(title)}</span>
      </span>
      ${detail ? `<span class="cell__detail">${ctx.escapeHtml(detail)}</span>` : ""}
      <span class="cell__chevron" aria-hidden="true"></span>
    </button>
  `;
}

function prototypeFeature(ctx, { title, icon, screen, action, bankSlug = "", sectionKey = "" }) {
  return `
    <button class="feature-card" type="button" data-screen-target="${ctx.escapeHtml(screen)}"${bankSlug ? ` data-attestation-bank="${ctx.escapeHtml(bankSlug)}"` : ""}${sectionKey ? ` data-section-key="${ctx.escapeHtml(sectionKey)}"` : ""}>
      <span class="feature-card__topline">
        <span class="feature-card__icon">${menuIcon(icon)}</span>
        <span class="feature-card__copy">
          <span class="feature-card__title">${ctx.escapeHtml(title)}</span>
        </span>
      </span>
      <span class="feature-card__go">
        <span class="feature-card__arrow">→</span>
        <span>${ctx.escapeHtml(action)}</span>
      </span>
    </button>
  `;
}

function renderCorrectAnswer(ctx, value, correctCount = 0) {
  const lines = String(value || "—")
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  const isList = correctCount > 1 || lines.length > 1;

  if (!isList) {
    return `<div class="case-answer__correct-text">${ctx.escapeHtml(lines[0] || "—")}</div>`;
  }

  return `
    <div class="case-answer__correct-list">
      ${lines.map((line, index) => {
        const match = line.match(/^(\d+[).])\s*(.*)$/);
        const marker = match ? match[1] : `${index + 1})`;
        const text = match ? match[2] : line;
        return `
          <div class="case-answer__correct-item">
            <span class="case-answer__correct-index">${ctx.escapeHtml(marker)}</span>
            <span>${ctx.escapeHtml(text || "—")}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

/* ===================== HOME ===================== */
export function renderHome(ctx) {
  const { user, catalog } = ctx.state.bootstrap;
  const sections = ctx.state.bootstrap.sections || [];
  const byKey = new Map(sections.map((item) => [item.key, item]));
  const primaryItems = [];
  const materialItems = [];
  const helpItems = [];

  const catalogBySlug = new Map((catalog.attestation_banks || []).map((bank) => [bank.slug, bank]));
  sections.forEach((section) => {
    if (!section.visible) return;
    if (section.manual_grant_only && !user.is_admin && !section.has_access) return;
    if (section.group === "primary") {
      const bank = section.bank_slug ? catalogBySlug.get(section.bank_slug) : null;
      if (section.kind === "attestation" && !bank) return;
      primaryItems.push(prototypeFeature(ctx, {
        title: section.title,
        icon: section.icon,
        screen: section.screen,
        action: section.has_access
          ? (section.kind === "attestation" ? "Перейти до тестування" : "Відкрити розділ")
          : (section.preview_count ? `Перші ${section.preview_count} питань безкоштовно` : "Відкрити розділ"),
        bankSlug: section.bank_slug || "",
        sectionKey: section.key,
      }));
      return;
    }
    const cell = prototypeMenuCell(ctx, {
      title: section.title,
      icon: section.icon,
      screen: section.screen,
      sectionKey: section.key,
    });
    if (section.group === "materials") materialItems.push(cell);
    if (section.group === "help") helpItems.push(cell);
  });
  if (user.is_admin) {
    helpItems.push(prototypeMenuCell(ctx, { title: "Адмін-панель", icon: "settings", screen: "admin" }));
  }

  ctx.setChrome({ showBack: false });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content screen-content--home">
      <h1 class="page-title">Головна</h1>
      <p class="page-subtitle">Митні компетенції, атестація та робота з матеріалами — в одному середовищі.</p>

      ${primaryItems.length ? `<div class="home-primary">${primaryItems.join("")}</div>` : ""}

      ${materialItems.length ? ctx.group({ header: "Матеріали", children: materialItems.join("") }) : ""}

      ${helpItems.length ? ctx.group({ header: "Допомога", children: helpItems.join("") }) : ""}
    </section>
  `;

  ctx.refs.mainPanel.querySelectorAll("[data-section-key]").forEach((button) => {
    button.addEventListener("click", (event) => {
      const section = byKey.get(button.dataset.sectionKey);
      if (!section.has_access && !section.preview_count) {
        event.preventDefault();
        event.stopImmediatePropagation();
        ctx.state.selectedPurchaseSectionKey = section.key;
        ctx.navigate("purchase-options");
      }
    });
  });
  ctx.refs.mainPanel.querySelectorAll("[data-attestation-bank]").forEach((button) => {
    button.addEventListener("click", () => {
      ctx.state.selectedAttestationBankSlug = button.dataset.attestationBank;
      ctx.state.selectedAttestationSection = null;
    });
  });
  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

/* ===================== CUSTOMS ===================== */
export function renderCustoms(ctx) {
  const { stats } = ctx.state.bootstrap;
  const customsSection = (ctx.state.bootstrap.sections || []).find((section) => section.key === "customs");
  const hasFullAccess = Boolean(ctx.state.bootstrap.user.is_admin || customsSection?.has_access);
  const last = stats.last;

  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Митні компетенції</h1>
      <p class="page-subtitle">${hasFullAccess ? "Навчання, тестування та ваші результати." : "Ознайомтеся з першими 50 питаннями безкоштовно."}</p>

      ${hasFullAccess ? ctx.group({
        children: [
          prototypeMenuCell(ctx, { title: "Навчання", icon: "books", screen: "learning" }),
          prototypeMenuCell(ctx, { title: "Тестування", icon: "flask", screen: "testing" }),
          prototypeMenuCell(ctx, { title: "Статистика", icon: "chart", screen: "stats", detail: last ? percentLabel(last.percent) : "" }),
        ].join(""),
      }) : `
        <div class="group">
          <div class="group__list">
            <div class="attestation-preview-card customs-preview-card">
              <div>
                <strong>Перші 50 питань безкоштовно</strong>
                <p>Пройдіть ознайомчий блок. Повне навчання і тестування відкриваються після покупки.</p>
              </div>
              <div id="customs-preview-start"></div>
              <div id="customs-preview-buy"></div>
            </div>
          </div>
        </div>
      `}
    </section>
  `;

  if (!hasFullAccess) {
    ctx.refs.mainPanel.querySelector("#customs-preview-start")?.append(
      ctx.actionButton("Розпочати перші 50 питань", ctx.startCustomsPreview, "block"),
    );
    ctx.refs.mainPanel.querySelector("#customs-preview-buy")?.append(
      ctx.actionButton(
        "Обрати варіант доступу",
        () => {
          ctx.state.selectedPurchaseSectionKey = "customs";
          ctx.navigate("purchase-options");
        },
        "block-ghost",
      ),
    );
  }
  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

/* ===================== CUSTOMS CODE REFERENCE ===================== */
function sectionRange(item) {
  const parts = [];
  if (item.first_chapter != null && item.last_chapter != null) {
    parts.push(
      item.first_chapter === item.last_chapter
        ? `Глава ${item.first_chapter}`
        : `Глави ${item.first_chapter}–${item.last_chapter}`
    );
  }
  if (item.first_article != null && item.last_article != null) {
    parts.push(
      item.first_article === item.last_article
        ? `Стаття ${item.first_article}`
        : `Статті ${item.first_article}–${item.last_article}`
    );
  }
  return parts.length ? parts.join(" · ") : `${item.chapters_count} глав · ${item.articles_count} статей`;
}

function customsMetaLine(ctx) {
  const meta = ctx.state.customsCodeMeta || {};
  const counts = ctx.state.customsCodeCounts || {};
  const parts = [];
  if (meta.edition_date) parts.push(`редакція ${ctx.escapeHtml(meta.edition_date)}`);
  if (counts.sections) parts.push(`${counts.sections} розділів`);
  if (counts.articles) parts.push(`${counts.articles} статей`);
  return parts.join(" · ");
}

function articleSubtitle(item) {
  const section = item.section_number ? `Розділ ${item.section_number}` : "";
  const chapter = item.chapter_number ? `Глава ${item.chapter_number}` : "";
  return [section, chapter, item.is_excluded ? "виключено" : ""].filter(Boolean).join(" · ");
}

function isTransitionalArticle(number) {
  return /^XXI-\d+(?:-\d+)?$/i.test(String(number || "").trim());
}

function articleDisplayPrefix(number) {
  return isTransitionalArticle(number) ? "Пункт" : "Стаття";
}

function articleDisplayNumber(number) {
  const value = String(number || "").trim();
  return isTransitionalArticle(value) ? value.replace(/^XXI-/i, "") : value;
}

function articleDisplayTitle(item) {
  const number = item?.number || "";
  return `${articleDisplayPrefix(number)} ${articleDisplayNumber(number)}. ${item?.title || ""}`.trim();
}

function isArticleNumberQuery(value) {
  const text = (value || "").trim();
  return /^(ст\.?|стаття)?\s*\d+(?:-\d+)?$/i.test(text)
    || /^(ст\.?|стаття)?\s*(xxi|ххі|хxi|xxі)[\s-]*(пункт\s*)?\d+(?:-\d+)?$/i.test(text)
    || /^п(ункт)?\.?\s*\d+(?:-\d+)?(\s+розділу\s*(xxi|ххі|хxi|xxі))?$/i.test(text);
}

function renderSearchSnippet(ctx, value) {
  return ctx.escapeHtml(value || "").replaceAll("‹", "<mark>").replaceAll("›", "</mark>");
}

function openCustomsArticle(ctx, number) {
  ctx.state.selectedCustomsArticleNumber = String(number || "");
  ctx.state.customsArticle = null;
  ctx.navigate("customs-code-article");
}

function drawCustomsCodeContent(ctx, root) {
  if (!root) return;
  const query = (ctx.state.customsSearchQuery || "").trim();

  if (query) {
    if (ctx.state.customsSearchLoading) {
      root.innerHTML = `
        <div class="group">
          <div class="group__list"><div class="empty empty--inline"><h2>Шукаємо…</h2></div></div>
        </div>
      `;
      return;
    }

    if (!ctx.state.customsSearchResults.length) {
      root.innerHTML = `
        <div class="group">
          <div class="group__list">
            <div class="empty empty--inline">
              <h2>Нічого не знайдено</h2>
              <p>Спробуйте номер статті або інші слова.</p>
            </div>
          </div>
        </div>
      `;
      return;
    }

    root.innerHTML = `
      <div class="group">
        <div class="group__label">Результати пошуку</div>
        <div class="group__list" id="customs-search-list"></div>
      </div>
    `;
    const list = root.querySelector("#customs-search-list");
    ctx.state.customsSearchResults.forEach((item) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "cell customs-result";
      row.innerHTML = `
        <span class="cell__icon cell__icon--indigo">${ctx.escapeHtml(String(articleDisplayNumber(item.number) || "§").slice(0, 4))}</span>
        <span class="cell__body">
          <span class="cell__title">${ctx.escapeHtml(articleDisplayTitle(item))}</span>
          <span class="cell__subtitle">${ctx.escapeHtml(articleSubtitle(item))}</span>
          ${item.snippet ? `<span class="customs-snippet">${renderSearchSnippet(ctx, item.snippet)}</span>` : ""}
        </span>
        <span class="cell__chevron" aria-hidden="true"></span>
      `;
      row.addEventListener("click", () => openCustomsArticle(ctx, item.number));
      list.append(row);
    });
    return;
  }

  if (!ctx.state.customsSections.length) {
    root.innerHTML = `
      <div class="group">
        <div class="group__list"><div class="empty empty--inline"><h2>Завантажуємо розділи…</h2></div></div>
      </div>
    `;
    return;
  }

  root.innerHTML = `
    <div class="group">
      <div class="group__label">Розділи кодексу</div>
      <div class="group__list" id="customs-sections-list"></div>
      <div class="group__footer">Текст збережено локально в проєкті, відкривається частинами без завантаження всього кодексу в Mini App.</div>
    </div>
  `;
  const list = root.querySelector("#customs-sections-list");
  ctx.state.customsSections.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell";
    row.innerHTML = `
      <span class="cell__icon cell__icon--indigo">${ctx.escapeHtml(item.number)}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(item.title)}</span>
        <span class="cell__subtitle">${sectionRange(item)}</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", () => {
      ctx.state.selectedCustomsSectionId = item.id;
      ctx.state.customsSectionDetail = null;
      ctx.navigate("customs-code-section");
    });
    list.append(row);
  });
}

async function runCustomsSearch(ctx) {
  const root = document.querySelector("#customs-code-body");
  const query = (ctx.state.customsSearchQuery || "").trim();
  if (!query || (query.length < 2 && !isArticleNumberQuery(query))) {
    ctx.state.customsSearchResults = [];
    ctx.state.customsSearchLoading = false;
    drawCustomsCodeContent(ctx, root);
    return;
  }

  try {
    ctx.state.customsSearchLoading = true;
    drawCustomsCodeContent(ctx, root);
    const payload = await ctx.api(`/api/customs-code/search?q=${encodeURIComponent(query)}&limit=30`);
    ctx.state.customsSearchResults = payload.items || [];
  } catch (error) {
    ctx.setMessage("error", error.message);
    ctx.state.customsSearchResults = [];
  } finally {
    ctx.state.customsSearchLoading = false;
    drawCustomsCodeContent(ctx, root);
  }
}

export function renderCustomsCode(ctx) {
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Митний кодекс України</h1>
      <p class="page-subtitle">${customsMetaLine(ctx) || "Розділи, глави, статті та пошук."}</p>

      <label class="customs-search">
        <span class="customs-search__icon" aria-hidden="true">⌕</span>
        <input class="customs-search__input" id="customs-search" type="search" value="${ctx.escapeHtml(ctx.state.customsSearchQuery || "")}" placeholder="Стаття 257, XXI-1 або митна вартість" />
      </label>

      <div id="customs-code-body"></div>
    </section>
  `;

  const root = ctx.refs.mainPanel.querySelector("#customs-code-body");
  drawCustomsCodeContent(ctx, root);

  const input = ctx.refs.mainPanel.querySelector("#customs-search");
  input?.addEventListener("input", () => {
    ctx.state.customsSearchQuery = input.value;
    window.clearTimeout(customsSearchTimer);
    customsSearchTimer = window.setTimeout(() => void runCustomsSearch(ctx), 320);
  });
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      window.clearTimeout(customsSearchTimer);
      void runCustomsSearch(ctx);
    }
  });
}

export async function loadCustomsCode(ctx) {
  if (ctx.state.currentScreen !== "customs-code") return;
  if (ctx.state.customsSections.length) return;
  try {
    const payload = await ctx.api("/api/customs-code/sections");
    ctx.state.customsCodeMeta = payload.meta || null;
    ctx.state.customsSections = payload.items || [];
    ctx.state.customsCodeCounts = {
      sections: ctx.state.customsSections.length,
      chapters: ctx.state.customsSections.reduce((sum, item) => sum + Number(item.chapters_count || 0), 0),
      articles: ctx.state.customsSections.reduce((sum, item) => sum + Number(item.articles_count || 0), 0),
    };
    if (ctx.state.currentScreen === "customs-code") ctx.render();
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export function renderCustomsSection(ctx) {
  const detail = ctx.state.customsSectionDetail;
  ctx.setChrome({ showBack: true });

  if (!detail) {
    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <h1 class="page-title">Митний кодекс</h1>
        <div class="group"><div class="group__list"><div class="empty empty--inline"><h2>Завантажуємо розділ…</h2></div></div></div>
      </section>
    `;
    return;
  }

  const section = detail.section;
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Розділ ${ctx.escapeHtml(section.number)}</h1>
      <p class="page-subtitle">${ctx.escapeHtml(section.title)}</p>
      <div id="customs-section-body"></div>
    </section>
  `;

  const body = ctx.refs.mainPanel.querySelector("#customs-section-body");
  body.innerHTML = detail.chapters.map((chapter) => `
    <div class="group">
      <div class="group__label">Глава ${ctx.escapeHtml(chapter.number)}${chapter.is_excluded ? " · виключено" : ""}</div>
      <div class="customs-chapter-title">${ctx.escapeHtml(chapter.title)}</div>
      <div class="group__list" data-chapter="${ctx.escapeHtml(String(chapter.id))}"></div>
    </div>
  `).join("");

  detail.chapters.forEach((chapter) => {
    const list = body.querySelector(`[data-chapter="${CSS.escape(String(chapter.id))}"]`);
    (chapter.articles || []).forEach((article) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "cell";
      row.innerHTML = `
        <span class="cell__icon cell__icon--blue">${ctx.escapeHtml(String(articleDisplayNumber(article.number) || "§").slice(0, 4))}</span>
        <span class="cell__body">
          <span class="cell__title">${ctx.escapeHtml(articleDisplayTitle(article))}</span>
          ${article.is_excluded ? `<span class="cell__subtitle">Виключено</span>` : ""}
        </span>
        <span class="cell__chevron" aria-hidden="true"></span>
      `;
      row.addEventListener("click", () => openCustomsArticle(ctx, article.number));
      list?.append(row);
    });
  });
}

export async function loadCustomsSection(ctx) {
  if (ctx.state.currentScreen !== "customs-code-section") return;
  const sectionId = ctx.state.selectedCustomsSectionId;
  if (!sectionId) return;
  if (ctx.state.customsSectionDetail?.section?.id === sectionId) return;
  try {
    const payload = await ctx.api(`/api/customs-code/sections/${sectionId}`);
    ctx.state.customsSectionDetail = payload;
    if (ctx.state.currentScreen === "customs-code-section") ctx.render();
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export function renderCustomsArticle(ctx) {
  const article = ctx.state.customsArticle;
  ctx.setChrome({ showBack: true });

  if (!article) {
    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <h1 class="page-title">${ctx.escapeHtml(articleDisplayPrefix(ctx.state.selectedCustomsArticleNumber))} ${ctx.escapeHtml(articleDisplayNumber(ctx.state.selectedCustomsArticleNumber || ""))}</h1>
        <div class="group"><div class="group__list"><div class="empty empty--inline"><h2>Завантажуємо статтю…</h2></div></div></div>
      </section>
    `;
    return;
  }

  const paragraphs = String(article.text || "")
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content screen-content--article">
      <h1 class="page-title">${ctx.escapeHtml(articleDisplayPrefix(article.number))} ${ctx.escapeHtml(articleDisplayNumber(article.number))}</h1>
      <p class="page-subtitle">${ctx.escapeHtml(article.title || "")}</p>

      <div class="customs-breadcrumb">
        Розділ ${ctx.escapeHtml(article.section_number || "")} · Глава ${ctx.escapeHtml(article.chapter_number || "")}
      </div>

      <article class="customs-article-text">
        ${paragraphs.length ? paragraphs.map((part) => `<p>${ctx.escapeHtml(part)}</p>`).join("") : `<p class="muted">Текст статті відсутній або статтю виключено.</p>`}
      </article>
    </section>
  `;
}

export async function loadCustomsArticle(ctx) {
  if (ctx.state.currentScreen !== "customs-code-article") return;
  const number = ctx.state.selectedCustomsArticleNumber;
  if (!number) return;
  if (ctx.state.customsArticle?.number === number) return;
  try {
    const payload = await ctx.api(`/api/customs-code/articles/${encodeURIComponent(number)}`);
    ctx.state.customsArticle = payload.article || null;
    if (ctx.state.currentScreen === "customs-code-article") ctx.render();
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

/* ===================== LEARNING ===================== */
export function renderLearning(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);
  const tab = ctx.state.learningTab || "law";

  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Навчання</h1>
      <p class="page-subtitle">Оберіть розділ і запустіть підготовку.</p>

      <div class="segmented">
        <button class="segmented__btn ${tab === "law" ? "is-active" : ""}" data-tab="law" type="button">Закон</button>
        <button class="segmented__btn ${tab === "ok" ? "is-active" : ""}" data-tab="ok" type="button">ОК-модулі</button>
        <button class="segmented__btn ${tab === "mistakes" ? "is-active" : ""}" data-tab="mistakes" type="button">Помилки</button>
      </div>

      <div id="learning-body"></div>
    </section>
  `;

  // Tab switching
  ctx.refs.mainPanel.querySelectorAll(".segmented__btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      ctx.impact("light");
      ctx.state.learningTab = btn.dataset.tab;
      ctx.render();
    });
  });

  const body = ctx.refs.mainPanel.querySelector("#learning-body");

  if (tab === "law") {
    renderLawTab(ctx, body);
  } else if (tab === "ok") {
    renderOkTab(ctx, body, modules);
  } else {
    renderMistakesTab(ctx, body);
  }
}

function renderLawTab(ctx, root) {
  const { catalog } = ctx.state.bootstrap;
  root.innerHTML = `
    <div class="group">
      <div class="group__label">Розділи закону</div>
      <div class="group__list" id="law-list"></div>
      <div class="group__footer">Кожна частина — 50 питань. «Рандом» — 50 випадкових з усього розділу.</div>
    </div>
  `;
  const list = root.querySelector("#law-list");
  catalog.law_groups.forEach((group) => {
    const partCount = Math.ceil(group.count / 50);
    const row = document.createElement("button");
    row.className = "cell";
    row.type = "button";
    row.innerHTML = `
      <span class="cell__icon cell__icon--blue">${menuIcon("books")}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(group.title)}</span>
        <span class="cell__subtitle">${group.count} питань · ${partCount} ${partCount === 1 ? "частина" : "частин"}</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", () => {
      ctx.impact("light");
      if (partCount <= 1) {
        ctx.startLearning({ kind: "law", group_key: group.key, part: 1 });
      } else {
        ctx.state.selectedLawGroup = group;
        ctx.navigate("law-parts");
      }
    });
    list.append(row);
  });
}

function openModulePicker(ctx, selected, onSave) {
  const overlay = document.createElement("div");
  overlay.className = "bottom-sheet-overlay";

  const sheet = document.createElement("div");
  sheet.className = "bottom-sheet";
  sheet.innerHTML = `
    <div class="bottom-sheet__handle"></div>
    <div class="bottom-sheet__header">
      <span class="bottom-sheet__title">Вибір модулів</span>
      <button class="bottom-sheet__close" type="button">✕</button>
    </div>
    <div class="bottom-sheet__body">
      <div class="group__list" id="sheet-picker-list"></div>
    </div>
    <div class="bottom-sheet__footer" id="sheet-save"></div>
  `;

  document.body.append(overlay, sheet);
  requestAnimationFrame(() => {
    overlay.classList.add("is-open");
    sheet.classList.add("is-open");
  });

  const close = () => {
    overlay.classList.remove("is-open");
    sheet.classList.remove("is-open");
    setTimeout(() => { overlay.remove(); sheet.remove(); }, 300);
  };

  overlay.addEventListener("click", close);
  sheet.querySelector(".bottom-sheet__close").addEventListener("click", close);

  const pickerList = sheet.querySelector("#sheet-picker-list");
  const { catalog } = ctx.state.bootstrap;

  catalog.ok_modules.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell";
    const isOn = selected.has(item.name);

    const iconEl = document.createElement("span");
    iconEl.className = `cell__icon cell__icon--${isOn ? "purple" : "gray"}`;
    iconEl.textContent = item.label.slice(0, 2).toUpperCase();

    const bodyEl = document.createElement("span");
    bodyEl.className = "cell__body";
    bodyEl.innerHTML = `<span class="cell__title">${ctx.escapeHtml(item.label)}</span>`;

    const checkEl = document.createElement("span");
    checkEl.style.cssText = "color:var(--accent);font-size:18px;flex-shrink:0;";
    checkEl.textContent = isOn ? "✓" : "";

    row.append(iconEl, bodyEl, checkEl);

    row.addEventListener("click", () => {
      ctx.impact("light");
      if (selected.has(item.name)) {
        selected.delete(item.name);
      } else {
        selected.add(item.name);
      }
      const on = selected.has(item.name);
      iconEl.className = `cell__icon cell__icon--${on ? "purple" : "gray"}`;
      checkEl.textContent = on ? "✓" : "";
    });

    pickerList.append(row);
  });

  const saveFooter = sheet.querySelector("#sheet-save");
  saveFooter.append(
    ctx.actionButton("Зберегти вибір", async () => {
      try {
        await onSave(Array.from(selected));
        close();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "block"),
  );
}

function renderOkTab(ctx, root, modules) {
  root.innerHTML = `
    <div class="group">
      <div class="group__label">Активні модулі</div>
      <div class="group__list" id="active-modules"></div>
      <div class="group__footer">Натисніть рівень, щоб розпочати навчання.</div>
    </div>
    <div style="padding: 0 4px 4px;" id="module-save"></div>
  `;

  // Active modules
  const activeRoot = root.querySelector("#active-modules");
  if (!modules.length) {
    activeRoot.innerHTML = `
      <div class="empty empty--inline">
        <h2>Немає активних модулів</h2>
        <p>Натисніть «Змінити вибір», щоб додати.</p>
      </div>
    `;
  } else {
    modules.forEach((item) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "cell";
      row.innerHTML = `
        <span class="cell__icon cell__icon--purple">${ctx.escapeHtml(item.label.slice(0, 2).toUpperCase())}</span>
        <span class="cell__body">
          <span class="cell__title">${ctx.escapeHtml(item.label)}</span>
          <span class="cell__subtitle">${item.levels.length} ${item.levels.length === 1 ? "рівень" : "рівні"}</span>
        </span>
        <span class="cell__chevron" aria-hidden="true"></span>
      `;
      row.addEventListener("click", () => {
        ctx.impact("light");
        ctx.state.selectedOkModule = item;
        ctx.navigate("ok-levels");
      });
      activeRoot.append(row);
    });
  }

  // "Змінити вибір" button
  const saveRoot = root.querySelector("#module-save");
  saveRoot.append(
    ctx.actionButton("Змінити вибір", () => {
      ctx.impact("light");
      const selected = new Set(modules.map((m) => m.name));
      openModulePicker(ctx, selected, async (modulesList) => {
        const response = await ctx.api("/api/preferences/ok-modules", {
          method: "POST",
          body: { modules: modulesList },
        });
        ctx.state.bootstrap.user = response.user;
        ctx.state.bootstrap.catalog = response.catalog;
        ctx.setMessage("success", "Збережено.");
        ctx.impact("medium");
        ctx.render();
      });
    }, "block-ghost"),
  );
}

function renderMistakesTab(ctx, root) {
  root.innerHTML = `
    <div class="group">
      <div class="group__list" style="padding: 18px 16px;">
        <p style="margin: 0 0 6px; font-size: 16px; font-weight: 600;">Тренування з помилок</p>
        <p class="muted" style="margin: 0 0 14px;">Повторіть лише ті питання, де ви помилилися. Ідеально, щоб закріпити слабкі місця.</p>
        <div id="mistakes-cta"></div>
      </div>
    </div>
  `;
  root.querySelector("#mistakes-cta").append(
    ctx.actionButton("Розпочати", ctx.startMistakesSession, "block"),
  );
}

/* ===================== LAW PARTS ===================== */
export function renderLawParts(ctx) {
  const group = ctx.state.selectedLawGroup;
  if (!group) {
    ctx.state.currentScreen = "learning";
    renderLearning(ctx);
    return;
  }

  const partCount = Math.ceil(group.count / 50);
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${ctx.escapeHtml(group.title)}</h1>
      <p class="page-subtitle">${group.count} питань · ${partCount} ${partCount === 1 ? "частина" : "частин"}</p>

      <div id="law-rand"></div>

      <div class="group">
        <div class="group__label">Частини</div>
        <div class="group__list" id="law-parts-list"></div>
      </div>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#law-rand").append(
    ctx.actionButton(
      "Випадкові 50 питань",
      async () => ctx.startLearning({ kind: "lawrand", group_key: group.key }),
      "block",
    ),
  );

  const list = ctx.refs.mainPanel.querySelector("#law-parts-list");
  for (let part = 1; part <= partCount; part += 1) {
    const start = (part - 1) * 50 + 1;
    const end = Math.min(part * 50, group.count);
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell";
    row.innerHTML = `
      <span class="cell__icon cell__icon--blue">${part}</span>
      <span class="cell__body">
        <span class="cell__title">Частина ${part}</span>
        <span class="cell__subtitle">Питання ${start}–${end}</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", () => {
      ctx.impact("light");
      ctx.startLearning({ kind: "law", group_key: group.key, part });
    });
    list.append(row);
  }
}

/* ===================== OK LEVELS ===================== */
export function renderOkLevels(ctx) {
  const item = ctx.state.selectedOkModule;
  if (!item) {
    ctx.state.currentScreen = "learning";
    renderLearning(ctx);
    return;
  }

  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${ctx.escapeHtml(item.label)}</h1>
      <p class="page-subtitle">Оберіть рівень, щоб розпочати навчання.</p>
      <div class="group">
        <div class="group__list" id="ok-levels-list"></div>
      </div>
    </section>
  `;

  const list = ctx.refs.mainPanel.querySelector("#ok-levels-list");
  item.levels.forEach((entry) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell" + (entry.level === item.last_level ? " cell--accent" : "");
    row.innerHTML = `
      <span class="cell__icon cell__icon--purple">${entry.level}</span>
      <span class="cell__body">
        <span class="cell__title">Рівень ${entry.level}</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", () => {
      ctx.impact("medium");
      ctx.startLearning({ kind: "ok", module: item.name, level: entry.level });
    });
    list.append(row);
  });
}

/* ===================== PUBLISHED ATTESTATION BANKS ===================== */
export function renderAttestationStage1(ctx) {
  const banks = ctx.state.bootstrap.catalog.attestation_banks || [];
  const fallbackSlug = ctx.state.currentScreen === "attestation-stage-1" ? "stage-1" : "";
  const slug = ctx.state.selectedAttestationBankSlug || fallbackSlug;
  const bank = banks.find((item) => item.slug === slug) || banks[0];
  const sections = bank?.sections || [];
  const managedSection = (ctx.state.bootstrap.sections || []).find((item) => item.bank_slug === bank?.slug);
  const hasFullBankAccess = bank?.system || Boolean(managedSection?.has_access);
  const previewCount = Number(bank?.preview_count || managedSection?.preview_count || 0);
  if (bank) ctx.state.selectedAttestationBankSlug = bank.slug;

  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${ctx.escapeHtml(bank?.title || "Тестування")}</h1>
      <p class="page-subtitle">${bank?.practice_count
        ? `${ctx.escapeHtml(bank.count - bank.practice_count)} тестових · ${ctx.escapeHtml(bank.practice_count)} практичних завдань`
        : `${ctx.escapeHtml(bank?.count || 0)} питань · повноцінний режим тестування.`}</p>

      ${!hasFullBankAccess && previewCount ? `
        <div class="group">
          <div class="group__label">Безкоштовне ознайомлення</div>
          <div class="group__list">
            <div class="attestation-preview-card">
              <p>Пройдіть перші ${ctx.escapeHtml(previewCount)} питань без покупки.</p>
              <button class="btn btn--primary btn--block" id="attestation-preview-start" type="button">Почати перші ${ctx.escapeHtml(previewCount)} питань</button>
            </div>
          </div>
        </div>
      ` : ""}

      ${ctx.group({
        header: hasFullBankAccess
          ? "Оберіть розділ"
          : (managedSection?.manual_grant_only ? "Доступ надає адміністратор" : `Повний доступ · ${ctx.escapeHtml(managedSection?.price || 0)} ⭐`),
        children: sections.map((item, index) => `
          <button class="cell" type="button" data-attestation-section="${index}">
            <span class="cell__icon cell__icon--purple">${index + 1}</span>
            <span class="cell__body">
              <span class="cell__title">${ctx.escapeHtml(item.title)}</span>
              <span class="cell__subtitle">${ctx.escapeHtml(item.count)} ${item.practice ? "завдань" : "питань"}</span>
            </span>
            <span class="cell__chevron" aria-hidden="true"></span>
          </button>
        `).join(""),
      })}
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#attestation-preview-start")?.addEventListener("click", () => {
    void startAttestationBlock(ctx, "", "preview");
  });

  ctx.refs.mainPanel.querySelectorAll("[data-attestation-section]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!hasFullBankAccess && managedSection) {
        if (managedSection.manual_grant_only) {
          ctx.setMessage("error", "Доступ до цього розділу надає адміністратор.");
          return;
        }
        ctx.state.selectedPurchaseSectionKey = managedSection.key;
        ctx.navigate("purchase-options");
        return;
      }
      const item = sections[Number(button.dataset.attestationSection)];
      if (!item) return;
      ctx.impact("light");
      ctx.state.selectedAttestationSection = item;
      ctx.navigate("attestation-parts");
    });
  });
}

async function startAttestationBlock(ctx, section, block) {
  const errorPanel = ctx.refs.mainPanel.querySelector("#attestation-start-error");
  if (errorPanel) {
    errorPanel.hidden = true;
    errorPanel.textContent = "";
  }
  ctx.setMessage("", "");
  const banks = ctx.state.bootstrap.catalog.attestation_banks || [];
  const bank = banks.find((item) => item.slug === ctx.state.selectedAttestationBankSlug);
  try {
    if (!bank) throw new Error("Розділ тестування не знайдено.");
    ctx.state.currentView = await ctx.api(`/api/attestation/${bank.slug}/start`, {
      method: "POST",
      body: { section, block },
    });
    ctx.queueTransition("forward");
    ctx.render();
  } catch (error) {
    if (error.code === "protected_materials_required") {
      ctx.setMessage("error", "Доступ до цього розділу надає адміністратор.");
      return;
    }
    if (error.code === "attestation_access_required" || error.code === "access_expired") {
      const managedSection = (ctx.state.bootstrap.sections || []).find((item) => item.bank_slug === bank?.slug);
      if (managedSection) {
        ctx.state.selectedPurchaseSectionKey = managedSection.key;
        ctx.navigate("purchase-options");
        return;
      }
      ctx.queueTransition("forward");
      renderPaywall(ctx, error.code);
      return;
    }
    if (errorPanel) {
      errorPanel.textContent = error.message || "Не вдалося розпочати тест. Спробуйте ще раз.";
      errorPanel.hidden = false;
      return;
    }
    ctx.setMessage("error", error.message);
  }
}

function renderAttestationPracticeTopics(ctx, section) {
  const banks = ctx.state.bootstrap.catalog.attestation_banks || [];
  const bank = banks.find((item) => item.slug === ctx.state.selectedAttestationBankSlug);
  const items = section.items || [];

  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${ctx.escapeHtml(section.title)}</h1>
      <p class="page-subtitle">${ctx.escapeHtml(items.length)} тем · оберіть тему для перегляду.</p>

      <div class="group">
        <div class="group__label">Теми</div>
        <div class="group__list" id="attestation-practice-topics"></div>
      </div>
    </section>
  `;

  const list = ctx.refs.mainPanel.querySelector("#attestation-practice-topics");
  items.forEach((item, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell";
    row.innerHTML = `
      <span class="cell__icon cell__icon--blue">${index + 1}</span>
      <span class="cell__body">
        <span class="cell__title">${ctx.escapeHtml(item.title)}</span>
        <span class="cell__subtitle">Переглянути завдання і зразок відповіді</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", async () => {
      try {
        if (!bank) throw new Error("Розділ не знайдено.");
        ctx.impact("light");
        ctx.state.currentView = await ctx.api(`/api/attestation/${bank.slug}/practice/${item.id}`);
        ctx.queueTransition("forward");
        ctx.render();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    });
    list.append(row);
  });
}

export function renderAttestationParts(ctx) {
  const section = ctx.state.selectedAttestationSection;
  if (!section) {
    ctx.state.currentScreen = "attestation-bank";
    renderAttestationStage1(ctx);
    return;
  }
  if (section.practice) {
    renderAttestationPracticeTopics(ctx, section);
    return;
  }

  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">${ctx.escapeHtml(section.title)}</h1>
      <p class="page-subtitle">${ctx.escapeHtml(section.count)} питань · частини до 50</p>

      <div id="attestation-random"></div>
      <div class="message message--error attestation-start-error" id="attestation-start-error" role="alert" hidden></div>

      <div class="group">
        <div class="group__label">Частини</div>
        <div class="group__list" id="attestation-parts-list"></div>
        <div class="group__footer">Після кожної відповіді показується правильний варіант і ваш вибір.</div>
      </div>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#attestation-random").append(
    ctx.actionButton(
      `Випадкові ${Math.min(50, Number(section.count || 0))} питань`,
      () => startAttestationBlock(ctx, section.key, "random"),
      "block",
    ),
  );

  const list = ctx.refs.mainPanel.querySelector("#attestation-parts-list");
  (section.blocks || []).forEach((entry, index) => {
    const block = entry.key;
    const row = document.createElement("button");
    row.type = "button";
    row.className = "cell cell--part-range";
    row.innerHTML = `
      <span class="cell__icon cell__icon--blue">${index + 1}</span>
      <span class="cell__body">
        <span class="cell__title">Частина ${index + 1}</span>
        <span class="cell__subtitle">Питання ${block.replace("-", "–")}</span>
      </span>
      <span class="cell__chevron" aria-hidden="true"></span>
    `;
    row.addEventListener("click", () => {
      ctx.impact("light");
      void startAttestationBlock(ctx, section.key, block);
    });
    list.append(row);
  });
}

/* ===================== TESTING ===================== */
export function renderTesting(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);

  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Тестування</h1>
      <p class="page-subtitle">Зберіть тест із закону та обраних модулів.</p>

      <div class="group">
        <div class="group__label">Що увімкнути</div>
        <div class="group__list">
          <div class="cell" style="cursor: default;">
            <span class="cell__icon cell__icon--blue">${menuIcon("scale")}</span>
            <span class="cell__body">
              <span class="cell__title">Законодавство</span>
              <span class="cell__subtitle">50 випадкових питань</span>
            </span>
            <label class="switch">
              <input type="checkbox" id="include-law" checked />
              <span class="switch__track"></span>
            </label>
          </div>
        </div>
        <div class="group__footer">Натисніть «Рівень», щоб увімкнути або вимкнути.</div>
      </div>

      <div class="group">
        <div class="group__label">Модулі та рівні</div>
        <div class="group__list" id="test-modules"></div>
      </div>

      <div class="sticky-cta" id="test-cta"></div>
    </section>
  `;

  const modulesNode = ctx.refs.mainPanel.querySelector("#test-modules");
  const selections = {};

  if (!modules.length) {
    modulesNode.innerHTML = `
      <div class="empty empty--inline">
        <h2>Немає активних модулів</h2>
        <p>Перейдіть у «Навчання» → «ОК-модулі» і виберіть.</p>
      </div>
    `;
  } else {
    modules.forEach((item) => {
      const initialLevel = item.last_level || item.levels[0]?.level;
      selections[item.name] = initialLevel ? [initialLevel] : [];

      const row = document.createElement("div");
      row.className = "cell";
      row.style.cursor = "default";
      row.innerHTML = `
        <span class="cell__icon cell__icon--purple">${ctx.escapeHtml(item.label.slice(0, 2).toUpperCase())}</span>
        <span class="cell__body">
          <span class="cell__title">${ctx.escapeHtml(item.label)}</span>
          <span class="cell__subtitle">Виберіть рівні</span>
        </span>
        <span class="row-actions"></span>
      `;
      const actions = row.querySelector(".row-actions");
      item.levels.forEach((levelEntry) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pill" + (selections[item.name].includes(levelEntry.level) ? " is-selected" : "");
        btn.textContent = `Рівень ${levelEntry.level}`;
        btn.addEventListener("click", () => {
          ctx.impact("light");
          const set = new Set(selections[item.name]);
          if (set.has(levelEntry.level)) {
            set.delete(levelEntry.level);
            btn.classList.remove("is-selected");
          } else {
            set.add(levelEntry.level);
            btn.classList.add("is-selected");
          }
          selections[item.name] = Array.from(set).sort((a, b) => a - b);
        });
        actions.append(btn);
      });
      modulesNode.append(row);
    });
  }

  ctx.refs.mainPanel.querySelector("#test-cta").append(
    ctx.actionButton(
      "Почати тест",
      async () => {
        try {
          ctx.state.currentView = await ctx.api("/api/test/start", {
            method: "POST",
            body: {
              include_law: ctx.refs.mainPanel.querySelector("#include-law").checked,
              module_levels: selections,
            },
          });
          ctx.impact("medium");
          ctx.queueTransition("forward");
          ctx.render();
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      },
      "block",
    ),
  );
}

/* ===================== STATS ===================== */
export function renderStats(ctx) {
  const { stats } = ctx.state.bootstrap;
  const last = stats.last;

  ctx.setChrome({ showBack: true });

  const pctClass = last
    ? last.percent >= 60
      ? "result-hero__pct--success"
      : "result-hero__pct--danger"
    : "";

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Статистика</h1>

      <div class="result-hero">
        <div class="result-hero__pct ${pctClass}">${last ? percentLabel(last.percent) : "—"}</div>
        <div class="result-hero__label">${last ? "Останній результат" : "Поки без тестів"}</div>
        ${last ? `<div class="result-hero__sub">${last.correct} / ${last.total} ${last.finished_at_label ? "· " + ctx.escapeHtml(last.finished_at_label) : ""}</div>` : ""}
      </div>

      <div class="stat-strip">
        ${ctx.statPill("Тестів", String(stats.count))}
        ${ctx.statPill("Середній", `${stats.avg.toFixed(0)}%`)}
        ${ctx.statPill("Останній", last ? `${last.correct}/${last.total}` : "—")}
      </div>

      <div class="stack" id="stats-actions"></div>
    </section>
  `;

  const actions = ctx.refs.mainPanel.querySelector("#stats-actions");
  actions.append(ctx.actionButton("Новий тест", async () => ctx.navigate("testing"), "block"));
  actions.append(ctx.actionButton("Перейти до навчання", async () => ctx.navigate("learning"), "block-ghost"));
}

/* ===================== HELP / MORE ===================== */
export function renderHelp(ctx) {
  const { links, user } = ctx.state.bootstrap;

  ctx.setChrome({ showBack: false });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Ще</h1>
      <p class="page-subtitle">Контакти підтримки та службові розділи.</p>

      ${ctx.group({
        header: "Контакти",
        children: [
          ctx.cell({
            title: "Telegram-група",
            subtitle: links.group_url ? "Спільнота користувачів" : "Посилання не налаштовано",
            iconName: "support",
            tint: "blue",
            link: links.group_url || "",
            chevron: Boolean(links.group_url),
          }),
          ctx.cell({
            title: "Адміністратор",
            subtitle: links.admin_url ? "Питання щодо доступу" : "Контакт не налаштовано",
            iconName: "user",
            tint: "orange",
            link: links.admin_url || "",
            chevron: Boolean(links.admin_url),
          }),
        ].join(""),
      })}

      ${
        user.is_admin
          ? ctx.group({
              header: "Адмін",
              children: [
                ctx.cell({
                  title: "Адмін-панель",
                  subtitle: "Користувачі та банк питань",
                  iconName: "settings",
                  tint: "indigo",
                  screen: "admin",
                }),
              ].join(""),
            })
          : ""
      }

      ${ctx.group({
        header: "Дії",
        children: [
          ctx.cell({
            title: "На головну",
            iconName: "home",
            tint: "teal",
            screen: "home",
          }),
        ].join(""),
      })}
    </section>
  `;

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}


/* ===================== PAYWALL ===================== */
export function renderPurchaseOptions(ctx) {
  ctx.setChrome({ showBack: true });
  const section = (ctx.state.bootstrap.sections || []).find(
    (item) => item.key === ctx.state.selectedPurchaseSectionKey,
  );
  if (!section || section.has_access || !section.visible) {
    ctx.navigate("home", { replace: true });
    return;
  }
  const fullPrice = Number(ctx.state.bootstrap.payment_prices?.full || 250);
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <header class="page-hero">
        <h1 class="page-title">Оберіть варіант доступу</h1>
        <p class="page-subtitle">${ctx.escapeHtml(section.title)}</p>
      </header>

      <div class="group">
        <div class="group__label">Лише цей розділ — ${ctx.escapeHtml(section.price)} ⭐</div>
        <div class="group__list" style="padding: 16px;">
          <p class="muted" style="margin: 0 0 12px; font-size: 15px;">Безстроковий доступ тільки до розділу «${ctx.escapeHtml(section.title)}».</p>
          <div id="pay-section-wrap"></div>
        </div>
      </div>

      <div class="group">
        <div class="group__label">Усі розділи — ${fullPrice} ⭐</div>
        <div class="group__list" style="padding: 16px;">
          <p class="muted" style="margin: 0 0 12px; font-size: 15px;">Відкриває всі платні розділи, які показані вам адміністратором.</p>
          <div id="pay-all-wrap"></div>
        </div>
      </div>
    </section>
  `;
  ctx.refs.mainPanel.querySelector("#pay-section-wrap")?.append(
    ctx.actionButton(
      `Купити лише цей розділ · ${section.price} ⭐`,
      () => void ctx.openPayment({ section_key: section.key }),
      "block",
    ),
  );
  ctx.refs.mainPanel.querySelector("#pay-all-wrap")?.append(
    ctx.actionButton(
      `Купити повний доступ до всіх розділів · ${fullPrice} ⭐`,
      () => void ctx.openPayment("full"),
      "block",
    ),
  );
}


export function renderPaywall(ctx, errorCode) {
  ctx.applyTransition();
  ctx.setChrome({ showBack: true });
  document.body.classList.add("ui-prototype");
  ctx.refs.mainPanel.dataset.screen = "paywall";

  const prices = ctx.state.bootstrap?.payment_prices || { cases: 100, full: 250 };
  const fullOnly = errorCode === "full_access_required" || errorCode === "ok_questions_access_required";
  const attestationOnly = errorCode === "attestation_access_required";
  const title = "Оберіть варіант доступу";

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <header class="page-hero">
        <h1 class="page-title">${title}</h1>
      </header>

      ${!fullOnly ? `
      <div class="group">
        <div class="group__label">Лише цей розділ — ${prices.cases} ⭐</div>
        <div class="group__list" style="padding: 16px;">
          <p class="muted" style="margin: 0 0 12px; font-size: 15px;">Безстроковий доступ тільки до розділу «Атестація посадових осіб митних органів 1 етап».</p>
          <div id="pay-cases-wrap"></div>
        </div>
      </div>
      ` : ""}

      <div class="group">
        <div class="group__label">Усі розділи — ${prices.full} ⭐</div>
        <div class="group__list" style="padding: 16px;">
          <p class="muted" style="margin: 0 0 12px; font-size: 15px;">Усі платні розділи, які показані вам адміністратором.</p>
          <div id="pay-full-wrap"></div>
        </div>
      </div>
    </section>
  `;

  if (!fullOnly) {
    ctx.refs.mainPanel.querySelector("#pay-cases-wrap")?.append(
      ctx.actionButton(`Купити лише цей розділ · ${prices.cases} ⭐`, () => void ctx.openPayment("cases"), "block"),
    );
  }
  ctx.refs.mainPanel.querySelector("#pay-full-wrap")?.append(
    ctx.actionButton(`Купити повний доступ до всіх розділів · ${prices.full} ⭐`, () => void ctx.openPayment("full"), "block"),
  );
}

/* ===================== CASES ===================== */
export function renderCases(ctx) {
  ctx.state.casesGlobalQuery = "";
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Кейси</h1>
      <p class="page-subtitle">Оберіть кейс, щоб переглянути питання й правильні відповіді.</p>

      <div class="case-search">
        <span class="case-search__icon" aria-hidden="true"></span>
        <input class="case-search__input" id="cases-global-search" type="search" value="${ctx.escapeHtml(ctx.state.casesGlobalQuery || "")}" placeholder="Пошук по всіх кейсах" />
      </div>

      <div class="group" id="cases-list-group">
        <div class="group__label">Доступні кейси</div>
        <div class="group__list" id="cases-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
      </div>

      <div id="cases-search-results" style="display:none;">
        <div class="group__label" id="cases-search-label">Результати пошуку</div>
        <div class="case-answer-list" id="cases-search-list"></div>
        <div class="row" id="cases-search-pagination" style="justify-content:center; gap:8px; margin-top:12px;"></div>
      </div>
    </section>
  `;

  const input = ctx.refs.mainPanel.querySelector("#cases-global-search");
  const run = () => {
    ctx.state.casesGlobalQuery = input.value.trim();
    if (ctx.state.casesGlobalQuery) {
      loadCasesSearch(ctx, 0);
    } else {
      showCasesList(ctx);
    }
  };
  const runLive = () => {
    window.clearTimeout(caseSearchTimer);
    caseSearchTimer = window.setTimeout(run, 350);
  };
  input?.addEventListener("input", runLive);
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      window.clearTimeout(caseSearchTimer);
      run();
    }
  });
}

function showCasesList(ctx) {
  const listGroup = document.querySelector("#cases-list-group");
  const searchResults = document.querySelector("#cases-search-results");
  if (listGroup) listGroup.style.display = "";
  if (searchResults) searchResults.style.display = "none";
}

function showSearchResults(ctx) {
  const listGroup = document.querySelector("#cases-list-group");
  const searchResults = document.querySelector("#cases-search-results");
  if (listGroup) listGroup.style.display = "none";
  if (searchResults) searchResults.style.display = "";
}

async function loadCasesSearch(ctx, offset = 0) {
  if (ctx.state.currentScreen !== "cases") return;
  const query = ctx.state.casesGlobalQuery || "";
  if (!query) { showCasesList(ctx); return; }
  const requestId = ++casesSearchRequestId;
  showSearchResults(ctx);
  const list = document.querySelector("#cases-search-list");
  const label = document.querySelector("#cases-search-label");
  const pagination = document.querySelector("#cases-search-pagination");
  if (list) list.innerHTML = `<div class="empty empty--inline"><h2>Шукаємо…</h2></div>`;
  try {
    const payload = await ctx.api(`/api/cases/search?q=${encodeURIComponent(query)}&offset=${offset}&limit=25`);
    if (requestId !== casesSearchRequestId || ctx.state.currentScreen !== "cases") return;
    if (label) label.textContent = `Результати пошуку: «${query}»`;
    if (!list) return;
    if (!payload.items?.length) {
      list.innerHTML = `<div class="empty empty--inline"><h2>Нічого не знайдено</h2><p>Спробуйте інший запит.</p></div>`;
    } else {
      list.innerHTML = "";
      payload.items.forEach((q) => {
        const block = document.createElement("article");
        block.className = "case-answer";
        block.style.cursor = "pointer";
        block.innerHTML = `
          <div class="case-answer__head">
            <span class="case-answer__number">Кейс ${ctx.escapeHtml(q.case_number)} · Питання ${ctx.escapeHtml(q.position)}</span>
            ${q.correct_count > 1 ? `<span class="case-answer__count">${q.correct_count} відповіді</span>` : ""}
          </div>
          <h2 class="case-answer__question">${ctx.escapeHtml(q.question)}</h2>
          <div class="case-answer__label">Правильна відповідь</div>
          <div class="case-answer__correct">
            <span class="case-answer__check" aria-hidden="true">✓</span>
            <div class="case-answer__correct-body">${renderCorrectAnswer(ctx, q.correct_answer, q.correct_count)}</div>
          </div>
        `;
        block.addEventListener("click", () => {
          const caseItem = (ctx.state.cases || []).find((c) => c.id === q.case_id) || { id: q.case_id, case_number: q.case_number };
          ctx.state.selectedCase = caseItem;
          ctx.state.caseOffset = 0;
          ctx.state.caseQuery = "";
          ctx.navigate("case-detail");
        });
        list.append(block);
      });
    }
    if (pagination) {
      pagination.innerHTML = "";
      if (payload.has_prev) {
        pagination.append(ctx.actionButton("← Назад", async () => loadCasesSearch(ctx, Math.max(0, offset - payload.limit)), "sm"));
      }
      if (payload.has_next) {
        pagination.append(ctx.actionButton("Далі →", async () => loadCasesSearch(ctx, offset + payload.limit), "sm"));
      }
    }
  } catch (error) {
    if (error.code === "cases_access_required" || error.code === "protected_materials_required" || error.code === "access_expired") {
      renderPaywall(ctx, error.code);
      return;
    }
    if (list) list.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
  }
}

export async function loadCases(ctx) {
  if (ctx.state.currentScreen !== "cases") return;
  try {
    const payload = await ctx.api("/api/cases");
    ctx.state.cases = payload.items || [];
    const list = document.querySelector("#cases-list");
    if (!list) return;
    if (!ctx.state.cases.length) {
      list.innerHTML = `
        <div class="empty empty--inline">
          <h2>Кейсів ще немає</h2>
          <p>Адмін може додати Keys.db в адмін-панелі.</p>
        </div>
      `;
      return;
    }
    list.innerHTML = "";
    ctx.state.cases.forEach((item) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "cell";
      row.innerHTML = `
        <span class="cell__icon cell__icon--green">${ctx.escapeHtml((item.case_number || "К").slice(0, 2))}</span>
        <span class="cell__body">
          <span class="cell__title">Кейс ${ctx.escapeHtml(item.case_number || "—")}</span>
          <span class="cell__subtitle">${ctx.escapeHtml(item.questions_count)} питань · ${ctx.escapeHtml(item.correct_count)} правильних</span>
        </span>
        <span class="cell__chevron" aria-hidden="true"></span>
      `;
      row.addEventListener("click", () => {
        ctx.state.selectedCase = item;
        ctx.state.caseOffset = 0;
        ctx.state.caseQuery = "";
        ctx.navigate("case-detail");
      });
      list.append(row);
    });
    if (ctx.state.casesGlobalQuery) {
      loadCasesSearch(ctx, 0);
    }
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}

export function renderCaseDetail(ctx) {
  const item = ctx.state.selectedCase;
  ctx.setChrome({ showBack: true });
  if (!item) {
    ctx.refs.mainPanel.innerHTML = `
      <section class="screen-content">
        <div class="empty"><h2>Кейс не вибрано</h2><p>Поверніться до списку кейсів.</p></div>
      </section>
    `;
    return;
  }
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content screen-content--case">
      <header class="case-header">
        <div class="case-header__app">Test_Customs</div>
        <h1 class="case-header__title">Кейс ${ctx.escapeHtml(item.case_number || "—")}</h1>
      </header>

      <div class="case-search">
        <span class="case-search__icon" aria-hidden="true"></span>
        <input class="case-search__input" id="case-search" type="search" value="${ctx.escapeHtml(ctx.state.caseQuery || "")}" placeholder="Пошук по питанню або відповіді" />
      </div>

      <section class="case-questions">
        <h2 class="case-questions__title">Питання та правильні відповіді</h2>
        <div class="case-answer-list" id="case-question-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
      </section>

      <div class="row" id="case-pagination" style="justify-content:center; gap:8px;"></div>
    </section>
  `;

  const input = ctx.refs.mainPanel.querySelector("#case-search");
  const run = () => {
    ctx.state.caseQuery = input.value.trim();
    ctx.state.caseOffset = 0;
    loadCaseDetail(ctx, 0);
  };
  const runLive = () => {
    window.clearTimeout(caseSearchTimer);
    caseSearchTimer = window.setTimeout(run, 350);
  };
  input?.addEventListener("input", runLive);
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      window.clearTimeout(caseSearchTimer);
      run();
    }
  });
}

/* ===================== OK QUESTIONS ===================== */
export function renderOkQuestions(ctx) {
  ctx.state.okSearchQuery = "";
  ctx.state.okSearchOffset = 0;
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Питання ОК</h1>
      <p class="page-subtitle">Питання операційних митних компетенцій з правильними відповідями.</p>

      <div class="case-search">
        <span class="case-search__icon" aria-hidden="true"></span>
        <input class="case-search__input" id="ok-search-input" type="search" value="${ctx.escapeHtml(ctx.state.okSearchQuery || "")}" placeholder="Пошук по питанню або відповіді" />
      </div>

      <div id="ok-search-results">
        <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
      </div>
      <div class="row" id="ok-search-pagination" style="justify-content:center; gap:8px; margin-top:12px;"></div>
    </section>
  `;

  const input = ctx.refs.mainPanel.querySelector("#ok-search-input");
  const run = () => {
    ctx.state.okSearchQuery = input.value.trim();
    ctx.state.okSearchOffset = 0;
    void loadOkSearch(ctx, 0);
  };
  const runLive = () => {
    window.clearTimeout(okSearchTimer);
    okSearchTimer = window.setTimeout(run, 350);
  };
  input?.addEventListener("input", runLive);
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      window.clearTimeout(okSearchTimer);
      run();
    }
  });

  void loadOkSearch(ctx, ctx.state.okSearchOffset || 0);
}

async function loadOkSearch(ctx, offset = 0) {
  if (ctx.state.currentScreen !== "ok-questions") return;
  const query = ctx.state.okSearchQuery || "";
  const requestId = ++okSearchRequestId;
  const results = document.querySelector("#ok-search-results");
  const pagination = document.querySelector("#ok-search-pagination");
  if (results) results.innerHTML = `<div class="empty empty--inline"><h2>Шукаємо…</h2></div>`;
  try {
    const payload = await ctx.api(`/api/ok-questions/search?q=${encodeURIComponent(query)}&offset=${offset}&limit=25`);
    if (requestId !== okSearchRequestId || ctx.state.currentScreen !== "ok-questions") return;
    ctx.state.okSearchOffset = offset;
    if (!results) return;
    if (!payload.items?.length) {
      results.innerHTML = query
        ? `<div class="empty empty--inline"><h2>Нічого не знайдено</h2><p>Спробуйте інший запит.</p></div>`
        : `<div class="empty empty--inline"><h2>Питань ОК ще немає</h2></div>`;
    } else {
      results.innerHTML = "";
      payload.items.forEach((q) => {
        const correctAnswer = (q.correct_texts || []).join("; ") || "—";
        const block = document.createElement("article");
        block.className = "case-answer";
        block.innerHTML = `
          <div class="case-answer__head">
            <span class="case-answer__number">${ctx.escapeHtml(q.ok || "ОК")}${q.level != null ? ` · Рівень ${ctx.escapeHtml(String(q.level))}` : ""}</span>
          </div>
          <h2 class="case-answer__question">${ctx.escapeHtml(q.question)}</h2>
          <div class="case-answer__label">Правильна відповідь</div>
          <div class="case-answer__correct">
            <span class="case-answer__check" aria-hidden="true">✓</span>
            <div class="case-answer__correct-body">${renderCorrectAnswer(ctx, correctAnswer, q.correct_texts?.length || 1)}</div>
          </div>
        `;
        results.append(block);
      });
    }
    if (pagination) {
      pagination.innerHTML = "";
      if (payload.has_prev) {
        pagination.append(ctx.actionButton("← Назад", () => void loadOkSearch(ctx, Math.max(0, offset - payload.limit)), "sm"));
      }
      if (payload.has_next) {
        pagination.append(ctx.actionButton("Далі →", () => void loadOkSearch(ctx, offset + payload.limit), "sm"));
      }
    }
  } catch (error) {
    if (error.code === "ok_questions_access_required" || error.code === "access_expired") {
      ctx.navigate("home", { replace: true });
      renderPaywall(ctx, "access_expired");
      return;
    }
    if (results) results.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
  }
}

export async function loadCaseDetail(ctx, offset = ctx.state.caseOffset || 0) {
  if (ctx.state.currentScreen !== "case-detail") return;
  const item = ctx.state.selectedCase;
  if (!item?.id) return;
  const requestId = ++caseDetailRequestId;
  try {
    const query = encodeURIComponent(ctx.state.caseQuery || "");
    const payload = await ctx.api(`/api/cases/${item.id}?offset=${offset}&limit=25&q=${query}`);
    if (requestId !== caseDetailRequestId || ctx.state.currentScreen !== "case-detail") return;
    ctx.state.selectedCase = payload.case;
    ctx.state.caseQuestions = payload.items || [];
    ctx.state.caseOffset = payload.offset || 0;

    const list = document.querySelector("#case-question-list");
    if (!list) return;
    if (!ctx.state.caseQuestions.length) {
      list.innerHTML = `
        <div class="empty empty--inline"><h2>Нічого не знайдено</h2><p>Спробуйте інший пошук.</p></div>
      `;
    } else {
      list.innerHTML = "";
      ctx.state.caseQuestions.forEach((q) => {
        const block = document.createElement("article");
        block.className = "case-answer";
        block.innerHTML = `
          <div class="case-answer__head">
            <span class="case-answer__number">Питання ${ctx.escapeHtml(q.position)}</span>
            ${q.correct_count > 1 ? `<span class="case-answer__count">${q.correct_count} відповіді</span>` : ""}
          </div>
          <h2 class="case-answer__question">${ctx.escapeHtml(q.question)}</h2>
          <div class="case-answer__label">Правильна відповідь</div>
          <div class="case-answer__correct">
            <span class="case-answer__check" aria-hidden="true">✓</span>
            <div class="case-answer__correct-body">${renderCorrectAnswer(ctx, q.correct_answer, q.correct_count)}</div>
          </div>
        `;
        list.append(block);
      });
    }

    const pagination = document.querySelector("#case-pagination");
    if (pagination) {
      pagination.innerHTML = "";
      if (payload.has_prev) {
        pagination.append(ctx.actionButton("← Назад", async () => loadCaseDetail(ctx, Math.max(0, payload.offset - payload.limit)), "sm"));
      }
      if (payload.has_next) {
        pagination.append(ctx.actionButton("Далі →", async () => loadCaseDetail(ctx, payload.offset + payload.limit), "sm"));
      }
    }
  } catch (error) {
    if (error.code === "cases_access_required" || error.code === "protected_materials_required" || error.code === "access_expired") {
      renderPaywall(ctx, error.code);
      return;
    }
    ctx.setMessage("error", error.message);
  }
}

/* ===================== TEST EXAM QUESTIONS (USER) ===================== */
let testExamSearchTimer = null;
let testExamRequestId = 0;

export function renderTestExamQuestions(ctx) {
  ctx.state.testExamSearchQuery = "";
  ctx.state.testExamOffset = 0;
  ctx.setChrome({ showBack: true });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Тестові питання</h1>
      <p class="page-subtitle">Питання та відповіді підсумкового тестування.</p>

      <div class="case-search">
        <span class="case-search__icon" aria-hidden="true"></span>
        <input class="case-search__input" id="test-exam-input" type="search"
               placeholder="Пошук по питанню або відповіді" />
      </div>

      <section class="case-questions">
        <h2 class="case-questions__title">Питання та правильні відповіді</h2>
        <div class="case-answer-list" id="test-exam-list">
          <div class="empty empty--inline"><h2>Завантажуємо…</h2></div>
        </div>
      </section>

      <div class="row" id="test-exam-pagination" style="justify-content:center; gap:8px; margin-top:12px;"></div>
    </section>
  `;

  const input = ctx.refs.mainPanel.querySelector("#test-exam-input");
  const run = () => {
    ctx.state.testExamSearchQuery = input.value.trim();
    ctx.state.testExamOffset = 0;
    void loadUserTestExamQuestions(ctx, 0);
  };
  const runLive = () => {
    window.clearTimeout(testExamSearchTimer);
    testExamSearchTimer = window.setTimeout(run, 350);
  };
  input?.addEventListener("input", runLive);
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      window.clearTimeout(testExamSearchTimer);
      run();
    }
  });
}

export async function loadUserTestExamQuestions(ctx, offset = ctx.state.testExamOffset || 0) {
  if (ctx.state.currentScreen !== "test-exam-questions") return;
  const query = ctx.state.testExamSearchQuery || "";
  const requestId = ++testExamRequestId;
  const list = document.querySelector("#test-exam-list");
  const pagination = document.querySelector("#test-exam-pagination");
  if (list) list.innerHTML = `<div class="empty empty--inline"><h2>Шукаємо…</h2></div>`;

  try {
    const payload = await ctx.api(
      `/api/test-exam-questions?q=${encodeURIComponent(query)}&offset=${offset}&limit=20`,
    );
    if (requestId !== testExamRequestId || ctx.state.currentScreen !== "test-exam-questions") return;
    ctx.state.testExamOffset = offset;
    if (!list) return;

    if (!payload.items?.length) {
      list.innerHTML = query
        ? `<div class="empty empty--inline"><h2>Нічого не знайдено</h2><p>Спробуйте інший запит.</p></div>`
        : `<div class="empty empty--inline"><h2>Питань поки немає</h2></div>`;
    } else {
      list.innerHTML = "";
      payload.items.forEach((item) => {
        const block = document.createElement("article");
        block.className = "case-answer";
        block.innerHTML = `
          <div class="case-answer__head">
            <span class="case-answer__number">${ctx.escapeHtml(item.num || "")}</span>
            ${item.module ? `<span class="case-answer__count">${ctx.escapeHtml(item.module)}</span>` : ""}
          </div>
          <h2 class="case-answer__question">${ctx.escapeHtml(item.question)}</h2>
          <div class="case-answer__label">Правильна відповідь</div>
          <div class="case-answer__correct">
            <span class="case-answer__check" aria-hidden="true">✓</span>
            <div class="case-answer__correct-body">
              <div class="case-answer__correct-text">${ctx.escapeHtml(item.correct_answer || "—")}</div>
            </div>
          </div>
        `;
        list.append(block);
      });
    }

    if (pagination) {
      pagination.innerHTML = "";
      if (payload.has_prev) {
        pagination.append(ctx.actionButton("← Назад", () => void loadUserTestExamQuestions(ctx, Math.max(0, offset - payload.limit)), "sm"));
      }
      if (payload.has_next) {
        pagination.append(ctx.actionButton("Далі →", () => void loadUserTestExamQuestions(ctx, offset + payload.limit), "sm"));
      }
    }
  } catch (error) {
    if (error.code === "full_access_required" || error.code === "protected_materials_required") {
      renderPaywall(ctx, "full_access_required");
      return;
    }
    if (list) list.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
  }
}

/* ===================== QUESTION GLOBAL SEARCH ===================== */
let questionSearchTimer = null;
let questionSearchRequestId = 0;

export function renderQuestionSearch(ctx) {
  ctx.state.questionGlobalQuery = ctx.state.questionGlobalQuery || "";
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Пошук питань</h1>
      <p class="page-subtitle">Пошук по законодавству, ОК-компетенціях та кейсах.</p>

      <div class="case-search">
        <span class="case-search__icon" aria-hidden="true"></span>
        <input class="case-search__input" id="qs-input" type="search"
               value="${ctx.escapeHtml(ctx.state.questionGlobalQuery || "")}"
               placeholder="Введіть текст питання…" autofocus />
      </div>

      <div id="qs-results">
        ${ctx.state.questionGlobalQuery
          ? `<div class="empty empty--inline"><h2>Шукаємо…</h2></div>`
          : `<div class="empty empty--inline"><h2>Введіть запит</h2><p>Мінімум 3 символи для початку пошуку.</p></div>`}
      </div>
    </section>
  `;

  const input = ctx.refs.mainPanel.querySelector("#qs-input");
  const run = () => {
    ctx.state.questionGlobalQuery = input.value.trim();
    void loadQuestionSearch(ctx);
  };
  input?.addEventListener("input", () => {
    window.clearTimeout(questionSearchTimer);
    questionSearchTimer = window.setTimeout(run, 350);
  });
  input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { window.clearTimeout(questionSearchTimer); run(); }
  });

  if (ctx.state.questionGlobalQuery) void loadQuestionSearch(ctx);
}

async function loadQuestionSearch(ctx) {
  if (ctx.state.currentScreen !== "question-search") return;
  const query = ctx.state.questionGlobalQuery || "";
  const results = document.querySelector("#qs-results");
  if (!query || query.length < 3) {
    if (results) results.innerHTML = `<div class="empty empty--inline"><h2>Введіть запит</h2><p>Мінімум 3 символи для початку пошуку.</p></div>`;
    return;
  }

  const requestId = ++questionSearchRequestId;
  if (results) results.innerHTML = `<div class="empty empty--inline"><h2>Шукаємо…</h2></div>`;

  try {
    const payload = await ctx.api(`/api/questions/search?q=${encodeURIComponent(query)}&limit=15`);
    if (requestId !== questionSearchRequestId || ctx.state.currentScreen !== "question-search") return;
    if (!results) return;

    const totalCount = (payload.law?.length || 0) + (payload.ok?.length || 0) + (payload.cases?.length || 0);
    if (!totalCount) {
      results.innerHTML = `<div class="empty empty--inline"><h2>Нічого не знайдено</h2><p>Спробуйте інший запит.</p></div>`;
      return;
    }

    results.innerHTML = "";

    if (payload.law?.length) {
      const header = document.createElement("div");
      header.className = "group__label";
      header.textContent = `Законодавство (${payload.law.length})`;
      results.append(header);
      payload.law.forEach((q) => results.append(buildQuestionCard(ctx, q, "law")));
    }

    if (payload.ok?.length) {
      const header = document.createElement("div");
      header.className = "group__label";
      header.textContent = `ОК-компетенції (${payload.ok.length})`;
      results.append(header);
      payload.ok.forEach((q) => results.append(buildQuestionCard(ctx, q, "ok")));
    }

    if (payload.cases?.length) {
      const header = document.createElement("div");
      header.className = "group__label";
      header.textContent = `Кейси (${payload.cases.length})`;
      results.append(header);
      payload.cases.forEach((q) => results.append(buildCaseCard(ctx, q)));
    }
  } catch (error) {
    if (error.code === "full_access_required" || error.code === "ok_questions_access_required" || error.code === "protected_materials_required") {
      ctx.navigate("home", { replace: true });
      renderPaywall(ctx, "access_expired");
      return;
    }
    if (results && error.code !== "short_query") {
      results.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`;
    }
  }
}

function buildQuestionCard(ctx, q, type) {
  const badge = type === "ok" && q.ok
    ? `${ctx.escapeHtml(q.ok)}${q.level != null ? ` · Рівень ${ctx.escapeHtml(String(q.level))}` : ""}`
    : ctx.escapeHtml(q.topic || "Законодавство");
  const correctAnswer = (q.correct_texts || []).join("; ") || "—";
  const block = document.createElement("article");
  block.className = "case-answer";
  block.innerHTML = `
    <div class="case-answer__head">
      <span class="case-answer__number">${badge}</span>
    </div>
    <h2 class="case-answer__question">${ctx.escapeHtml(q.question)}</h2>
    <div class="case-answer__label">Правильна відповідь</div>
    <div class="case-answer__correct">
      <span class="case-answer__check" aria-hidden="true">✓</span>
      <div class="case-answer__correct-body">${renderCorrectAnswer(ctx, correctAnswer, q.correct_texts?.length || 1)}</div>
    </div>
  `;
  return block;
}

function buildCaseCard(ctx, q) {
  const block = document.createElement("article");
  block.className = "case-answer";
  block.innerHTML = `
    <div class="case-answer__head">
      <span class="case-answer__number">Кейс ${ctx.escapeHtml(q.case_number || "")}</span>
    </div>
    <h2 class="case-answer__question">${ctx.escapeHtml(q.question)}</h2>
    <div class="case-answer__label">Правильна відповідь</div>
    <div class="case-answer__correct">
      <span class="case-answer__check" aria-hidden="true">✓</span>
      <div class="case-answer__correct-body">${renderCorrectAnswer(ctx, q.correct_answer || "—", q.correct_count || 1)}</div>
    </div>
  `;
  return block;
}
