import { toggleTheme, getCurrentTheme } from "../core/theme.js";

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

/* ===================== HOME ===================== */
export function renderHome(ctx) {
  const { user, catalog, stats } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);
  const last = stats.last;
  const access = user.access.has_access;

  ctx.setChrome({ showBack: false });

  const initial = initialOf(user.first_name);
  const name = ctx.escapeHtml(user.first_name || "Користувач");
  const accessChipClass = access ? "chip--success" : "chip--danger";
  const accessLabel = access ? "Активний доступ" : "Доступ обмежено";

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Головна</h1>

      <div class="greeting">
        <div class="greeting__avatar">${ctx.escapeHtml(initial)}</div>
        <div class="greeting__copy">
          <div class="greeting__hello">Вітаємо</div>
          <div class="greeting__name">${name}</div>
        </div>
        <span class="chip ${accessChipClass}">${accessLabel}</span>
      </div>

      <div class="stat-strip">
        ${ctx.statPill("Модулі", String(modules.length))}
        ${ctx.statPill("Тестів", String(stats.count))}
        ${ctx.statPill("Останній", last ? percentLabel(last.percent) : "—")}
      </div>

      <div class="stack" id="home-cta"></div>

      ${ctx.group({
        header: "Швидкий старт",
        children: [
          ctx.cell({
            title: "Навчання",
            subtitle: `${catalog.law_groups.length} розділів закону`,
            icon: "📚",
            tint: "blue",
            screen: "learning",
          }),
          ctx.cell({
            title: "Новий тест",
            subtitle: "Зібрати з модулів і закону",
            icon: "🧪",
            tint: "purple",
            screen: "testing",
          }),
          ctx.cell({
            title: "Статистика",
            subtitle: last ? `Останній: ${last.correct}/${last.total}` : "Запустіть перший тест",
            icon: "📊",
            tint: "green",
            screen: "stats",
            detail: last ? percentLabel(last.percent) : undefined,
          }),
        ].join(""),
      })}

      ${ctx.group({
        header: "Допомога",
        children: [
          ctx.cell({
            title: "Підтримка",
            subtitle: "Група, адміністратор",
            icon: "💬",
            tint: "orange",
            screen: "help",
          }),
          user.is_admin
            ? ctx.cell({
                title: "Адмін-панель",
                subtitle: "Користувачі та банк питань",
                icon: "⚙",
                tint: "indigo",
                screen: "admin",
              })
            : "",
        ].join(""),
      })}
    </section>
  `;

  // Primary CTA — start mistakes session if any
  const cta = ctx.refs.mainPanel.querySelector("#home-cta");
  cta.append(
    ctx.actionButton(
      "Почати тренування з помилок",
      ctx.startMistakesSession,
      "block",
    ),
  );

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

/* ===================== LEARNING ===================== */
export function renderLearning(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);
  const tab = ctx.state.learningTab || "law";

  ctx.setChrome({ showBack: false });

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
      renderLearning(ctx);
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
      <span class="cell__icon cell__icon--blue">📖</span>
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

function renderOkTab(ctx, root, modules) {
  const { catalog } = ctx.state.bootstrap;
  root.innerHTML = `
    <div class="group">
      <div class="group__label">Активні модулі</div>
      <div class="group__list" id="active-modules"></div>
      <div class="group__footer">Запускайте по рівнях. Зміна вибору модулів — нижче.</div>
    </div>

    <div style="height: 12px"></div>

    <div class="group">
      <div class="group__label">Вибір модулів</div>
      <div class="group__list" style="padding: 12px;">
        <div class="row" id="module-picker"></div>
        <div style="height: 10px"></div>
        <div id="module-save"></div>
      </div>
    </div>
  `;

  const activeRoot = root.querySelector("#active-modules");
  if (!modules.length) {
    activeRoot.innerHTML = `
      <div class="empty empty--inline">
        <h2>Немає активних модулів</h2>
        <p>Виберіть нижче — і вони з’являться тут.</p>
      </div>
    `;
  } else {
    modules.forEach((item) => {
      const row = document.createElement("div");
      row.className = "cell";
      row.style.cursor = "default";
      row.innerHTML = `
        <span class="cell__icon cell__icon--purple">${ctx.escapeHtml(item.label.slice(0, 2).toUpperCase())}</span>
        <span class="cell__body">
          <span class="cell__title">${ctx.escapeHtml(item.label)}</span>
          <span class="cell__subtitle">Рівні: ${item.levels.map((l) => "L" + l.level).join(" · ")}</span>
        </span>
        <span class="row-actions" style="gap:6px"></span>
      `;
      const actions = row.querySelector(".row-actions");
      item.levels.forEach((entry) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pill";
        btn.textContent = `L${entry.level}`;
        if (entry.level === item.last_level) btn.classList.add("is-selected");
        btn.addEventListener("click", () => {
          ctx.impact("light");
          ctx.startLearning({ kind: "ok", module: item.name, level: entry.level });
        });
        actions.append(btn);
      });
      activeRoot.append(row);
    });
  }

  // Module picker
  const selector = root.querySelector("#module-picker");
  const selected = new Set(modules.map((m) => m.name));
  catalog.ok_modules.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pill" + (selected.has(item.name) ? " is-selected" : "");
    btn.textContent = item.label;
    btn.addEventListener("click", () => {
      ctx.impact("light");
      if (selected.has(item.name)) {
        selected.delete(item.name);
        btn.classList.remove("is-selected");
      } else {
        selected.add(item.name);
        btn.classList.add("is-selected");
      }
    });
    selector.append(btn);
  });

  const saveRoot = root.querySelector("#module-save");
  saveRoot.append(
    ctx.actionButton(
      "Зберегти вибір",
      async () => {
        try {
          const response = await ctx.api("/api/preferences/ok-modules", {
            method: "POST",
            body: { modules: Array.from(selected) },
          });
          ctx.state.bootstrap.user = response.user;
          ctx.state.bootstrap.catalog = response.catalog;
          ctx.setMessage("success", "Збережено.");
          ctx.impact("medium");
          renderLearning(ctx);
        } catch (error) {
          ctx.setMessage("error", error.message);
        }
      },
      "block",
    ),
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

/* ===================== TESTING ===================== */
export function renderTesting(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);

  ctx.setChrome({ showBack: false });

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Тестування</h1>
      <p class="page-subtitle">Зберіть тест із закону та обраних модулів.</p>

      <div class="group">
        <div class="group__label">Що увімкнути</div>
        <div class="group__list">
          <div class="cell" style="cursor: default;">
            <span class="cell__icon cell__icon--blue">⚖</span>
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
        <div class="group__footer">Натисніть на пілюлю рівня, щоб увімкнути або вимкнути.</div>
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
        btn.textContent = `L${levelEntry.level}`;
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

  ctx.setChrome({ showBack: false });

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

      ${ctx.group({
        header: "Контакти",
        children: [
          ctx.cell({
            title: "Telegram-група",
            subtitle: links.group_url ? "Спільнота користувачів" : "Посилання не налаштовано",
            icon: "💬",
            tint: "blue",
            link: links.group_url || "",
            chevron: Boolean(links.group_url),
          }),
          ctx.cell({
            title: "Адміністратор",
            subtitle: links.admin_url ? "Питання щодо доступу" : "Контакт не налаштовано",
            icon: "👤",
            tint: "orange",
            link: links.admin_url || "",
            chevron: Boolean(links.admin_url),
          }),
        ].join(""),
      })}

      ${ctx.group({
        header: "Інтерфейс",
        children: [
          `<button class="cell" type="button" id="theme-toggle">
            <span class="cell__icon cell__icon--gray">${getCurrentTheme() === "dark" ? "🌙" : "☀"}</span>
            <span class="cell__body">
              <span class="cell__title">Тема</span>
              <span class="cell__subtitle" id="theme-subtitle">${getCurrentTheme() === "dark" ? "Темна" : "Світла"}</span>
            </span>
            <span class="cell__chevron" aria-hidden="true"></span>
          </button>`,
          `<button class="cell" type="button" id="refresh-data">
            <span class="cell__icon cell__icon--green">↻</span>
            <span class="cell__body">
              <span class="cell__title">Оновити дані</span>
              <span class="cell__subtitle">Синхронізувати з сервером</span>
            </span>
            <span class="cell__chevron" aria-hidden="true"></span>
          </button>`,
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
                  icon: "⚙",
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
            icon: "🏠",
            tint: "teal",
            screen: "home",
          }),
        ].join(""),
      })}
    </section>
  `;

  // Theme toggle
  ctx.refs.mainPanel.querySelector("#theme-toggle")?.addEventListener("click", () => {
    ctx.impact("light");
    const next = toggleTheme();
    ctx.refs.mainPanel.querySelector("#theme-subtitle").textContent = next === "dark" ? "Темна" : "Світла";
    const iconEl = ctx.refs.mainPanel.querySelector("#theme-toggle .cell__icon");
    if (iconEl) iconEl.textContent = next === "dark" ? "🌙" : "☀";
  });

  ctx.refs.mainPanel.querySelector("#refresh-data")?.addEventListener("click", async () => {
    ctx.impact("light");
    await ctx.loadBootstrap(true);
  });

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}
