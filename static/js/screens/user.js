function openLawParts(ctx, group) {
  ctx.state.selectedLawGroup = group;
  ctx.navigate("law-parts");
}

function compactAccessLabel(user) {
  return user.access.has_access ? "Активний" : "Обмежено";
}

function selectedModules(catalog) {
  return catalog.ok_modules.filter((item) => item.selected);
}

function percentLabel(value) {
  return typeof value === "number" ? `${value.toFixed(0)}%` : "—";
}

function bindSharedNav(ctx) {
  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

export function renderHome(ctx) {
  const { user, catalog, stats } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);
  const last = stats.last;

  ctx.setChrome({
    eyebrow: "Підготовка",
    title: "Головна",
    subtitle: "",
    showBack: false,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen ios-home">
      <section class="ios-hero-card home-hero-card">
        <div class="ios-hero-card__top">
          <span class="ios-avatar">${ctx.escapeHtml((user.first_name || "U").slice(0, 1).toUpperCase())}</span>
          <div>
            <span class="eyebrow">Exam Mini App</span>
            <h2>Підготовка без зайвого</h2>
            <p class="muted">Один головний сценарій: навчання → тест → статистика.</p>
          </div>
          <span class="status-chip ${user.access.has_access ? "is-active" : "is-danger"}">${compactAccessLabel(user)}</span>
        </div>
        <div class="mini-stats-grid">
          ${ctx.metricCard("ОК-модулі", String(modules.length), "вибрано")}
          ${ctx.metricCard("Тести", String(stats.count), "усього")}
          ${ctx.metricCard("Останній", last ? percentLabel(last.percent) : "—", "результат")}
        </div>
        <div class="button-row" id="home-primary-actions"></div>
      </section>

      <section class="ios-section screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>Розділи</h2>
            <p>Один список навігації без дублювання плиток.</p>
          </div>
        </div>
        ${ctx.screenBar("home")}
      </section>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#home-primary-actions").append(
    ctx.actionButton("Почати тест", async () => ctx.navigate("testing"), "primary"),
    ctx.actionButton("Відкрити навчання", async () => ctx.navigate("learning")),
  );

  bindSharedNav(ctx);
}

export function renderLearning(ctx) {
  const { catalog, stats } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);
  const lawCount = catalog.law_groups.reduce((sum, group) => sum + Number(group.count || 0), 0);

  ctx.setChrome({
    eyebrow: "Навчання",
    title: "Навчання",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen learning-screen compact-screen">
      <section class="ios-section ios-section--hero">
        <div class="section-header">
          <div class="section-copy">
            <span class="eyebrow">Study</span>
            <h2>Оберіть напрям</h2>
            <p>Законодавство, ОК-модулі та помилки тепер розділені.</p>
          </div>
        </div>
        <div class="mini-stats-grid">
          ${ctx.metricCard("Законодавство", String(lawCount), "питань")}
          ${ctx.metricCard("ОК-модулі", String(modules.length), "вибрано")}
          ${ctx.metricCard("Тести", String(stats.count), "історія")}
        </div>
      </section>

      <section class="ios-tile-grid learning-route-grid" aria-label="Напрями навчання">
        ${ctx.actionCard({
          code: "⚖️",
          title: "Законодавство",
          body: "Окремі групи та частини по 50 питань.",
          meta: `${catalog.law_groups.length} розд.`,
          screen: "law-learning",
          tone: "blue",
        })}
        ${ctx.actionCard({
          code: "🎯",
          title: "ОК модулі",
          body: "Вибір модулів і запуск потрібного рівня.",
          meta: `${modules.length} вибр.`,
          screen: "ok-learning",
          tone: "purple",
        })}
        <button class="nav-card nav-card--orange" type="button" id="mistakes-card">
          <span class="monogram">↻</span>
          <span class="nav-card__content">
            <span class="nav-card__meta">Повтор</span>
            <span class="nav-card__title">Помилки</span>
            <span class="nav-card__text">Окремий запуск для слабких місць.</span>
          </span>
          <span class="nav-card__chevron" aria-hidden="true">›</span>
        </button>
        ${ctx.actionCard({
          code: "🧪",
          title: "Тестування",
          body: "Перевірка після навчання.",
          meta: "Старт",
          screen: "testing",
          tone: "green",
        })}
      </section>

      ${ctx.screenBar("learning")}
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#mistakes-card")?.addEventListener("click", ctx.startMistakesSession);
  bindSharedNav(ctx);
}

export function renderLawLearning(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const lawCount = catalog.law_groups.reduce((sum, group) => sum + Number(group.count || 0), 0);

  ctx.setChrome({
    eyebrow: "Навчання",
    title: "Законодавство",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen learning-screen compact-screen">
      <section class="ios-section ios-section--hero">
        <div class="section-header">
          <div class="section-copy">
            <span class="eyebrow">Law</span>
            <h2>Законодавство</h2>
            <p>Окремий екран тільки для законодавства: група, частина або випадкові питання.</p>
          </div>
          <span class="status-chip">${lawCount} питань</span>
        </div>
      </section>

      <section class="ios-section screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>Групи</h2>
            <p>Якщо група велика — відкриється список частин по 50 питань.</p>
          </div>
        </div>
        <div class="list-stack compact-list" id="law-groups-stack"></div>
      </section>

      ${ctx.screenBar("law-learning")}
    </section>
  `;

  const lawStack = ctx.refs.mainPanel.querySelector("#law-groups-stack");
  catalog.law_groups.forEach((group) => {
    const node = document.createElement("article");
    node.className = "list-item ios-list-item";
    node.innerHTML = `
      <div class="list-item__main">
        <strong>${ctx.escapeHtml(group.title)}</strong>
        <span class="list-item__meta">${group.count} питань</span>
      </div>
      <div class="button-row"></div>
    `;

    const actions = node.querySelector(".button-row");
    const partCount = Math.ceil(group.count / 50);
    if (partCount <= 1) {
      actions.append(ctx.actionButton("Старт", async () => ctx.startLearning({ kind: "law", group_key: group.key, part: 1 }), "primary"));
    } else {
      actions.append(ctx.actionButton(`${partCount} част.`, async () => openLawParts(ctx, group), "primary"));
    }
    actions.append(ctx.actionButton("Рандом", async () => ctx.startLearning({ kind: "lawrand", group_key: group.key })));
    lawStack.append(node);
  });

  bindSharedNav(ctx);
}

export function renderOkLearning(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);

  ctx.setChrome({
    eyebrow: "Навчання",
    title: "ОК модулі",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen learning-screen compact-screen">
      <section class="ios-section ios-section--hero">
        <div class="section-header">
          <div class="section-copy">
            <span class="eyebrow">Modules</span>
            <h2>ОК модулі</h2>
            <p>Окремо налаштовуємо модулі й запускаємо навчання по рівнях.</p>
          </div>
          <span class="status-chip">${modules.length} вибр.</span>
        </div>
      </section>

      <section class="ios-section screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>Вибір модулів</h2>
            <p>Позначте потрібні ОК-модулі та збережіть список.</p>
          </div>
        </div>
        <div class="section-stack" id="ok-modules-stack"></div>
      </section>

      ${ctx.screenBar("ok-learning")}
    </section>
  `;

  const okStack = ctx.refs.mainPanel.querySelector("#ok-modules-stack");
  const editor = document.createElement("section");
  editor.className = "inline-form module-picker ios-form";
  editor.innerHTML = `
    <div class="field">
      <label>Модулі</label>
      <div class="chips-row" id="module-selector"></div>
    </div>
    <div class="button-row" id="module-selector-actions"></div>
  `;
  okStack.append(editor);

  const selectedNames = new Set(modules.map((item) => item.name));
  const selector = editor.querySelector("#module-selector");
  catalog.ok_modules.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `pill-button ${selectedNames.has(item.name) ? "is-selected" : ""}`;
    button.textContent = item.label;
    button.addEventListener("click", () => {
      ctx.impact("light");
      if (selectedNames.has(item.name)) {
        selectedNames.delete(item.name);
        button.classList.remove("is-selected");
      } else {
        selectedNames.add(item.name);
        button.classList.add("is-selected");
      }
    });
    selector.append(button);
  });

  editor.querySelector("#module-selector-actions").append(
    ctx.actionButton("Зберегти", async () => {
      try {
        const response = await ctx.api("/api/preferences/ok-modules", {
          method: "POST",
          body: { modules: Array.from(selectedNames) },
        });
        ctx.state.bootstrap.user = response.user;
        ctx.state.bootstrap.catalog = response.catalog;
        ctx.setMessage("success", "Збережено.");
        ctx.impact("medium");
        renderOkLearning(ctx);
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "primary"),
  );

  if (!modules.length) {
    const note = document.createElement("div");
    note.className = "empty-state compact-empty";
    note.innerHTML = `
      <h2>Оберіть модулі</h2>
      <p>Після збереження тут з’являться рівні.</p>
    `;
    okStack.append(note);
    bindSharedNav(ctx);
    return;
  }

  const levelsSection = document.createElement("section");
  levelsSection.className = "section-stack";
  levelsSection.innerHTML = `
    <div class="section-header">
      <div class="section-copy">
        <h2>Рівні</h2>
        <p>Запускайте навчання по конкретному рівню вибраного модуля.</p>
      </div>
    </div>
  `;
  okStack.append(levelsSection);

  modules.forEach((item) => {
    const card = document.createElement("article");
    card.className = "list-item ios-list-item";
    card.innerHTML = `
      <div class="list-item__main">
        <strong>${ctx.escapeHtml(item.label)}</strong>
        <span class="list-item__meta">Рівні: ${item.levels.map((entry) => entry.level).join(", ")}</span>
      </div>
      <div class="button-row"></div>
    `;

    const buttons = card.querySelector(".button-row");
    item.levels.forEach((levelEntry) => {
      buttons.append(
        ctx.actionButton(
          `L${levelEntry.level}`,
          async () => ctx.startLearning({ kind: "ok", module: item.name, level: levelEntry.level }),
          levelEntry.level === item.last_level ? "primary" : "secondary",
        ),
      );
    });
    levelsSection.append(card);
  });

  bindSharedNav(ctx);
}

export function renderLawParts(ctx) {
  const group = ctx.state.selectedLawGroup;
  if (!group) {
    ctx.state.currentScreen = "law-learning";
    renderLawLearning(ctx);
    return;
  }

  const partCount = Math.ceil(group.count / 50);
  ctx.setChrome({
    eyebrow: "Навчання",
    title: group.title,
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen compact-screen">
      <section class="ios-section ios-section--hero">
        <div class="section-header">
          <div class="section-copy">
            <span class="eyebrow">${group.count} питань</span>
            <h2>Частини</h2>
            <p>Оберіть діапазон або випадкові 50.</p>
          </div>
        </div>
        <div class="button-row" id="law-parts-actions"></div>
      </section>
      <section class="cards-grid compact-card-grid" id="law-parts-grid"></section>
      ${ctx.screenBar("law-parts")}
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#law-parts-actions").append(
    ctx.actionButton("Назад", ctx.goBack),
    ctx.actionButton("Рандом 50", async () => ctx.startLearning({ kind: "lawrand", group_key: group.key }), "primary"),
  );

  const grid = ctx.refs.mainPanel.querySelector("#law-parts-grid");
  for (let part = 1; part <= partCount; part += 1) {
    const start = (part - 1) * 50 + 1;
    const end = Math.min(part * 50, group.count);
    const card = document.createElement("article");
    card.className = "grid-card ios-grid-card";
    card.innerHTML = `
      <span class="eyebrow">Частина ${part}</span>
      <strong>${start}-${end}</strong>
      <p class="muted">Питання ${start}–${end}</p>
    `;

    const actions = document.createElement("div");
    actions.className = "button-row";
    actions.append(ctx.actionButton("Старт", async () => ctx.startLearning({ kind: "law", group_key: group.key, part }), "primary"));
    card.append(actions);
    grid.append(card);
  }

  bindSharedNav(ctx);
}

export function renderTesting(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);

  ctx.setChrome({
    eyebrow: "Тестування",
    title: "Тестування",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen testing-screen testing-studio-screen">
      <section class="test-builder-hero">
        <div class="test-builder-hero__copy">
          <span class="test-mode-chip">Exam mode</span>
          <h2>Зберіть тест</h2>
          <p>Оберіть законодавство та рівні ОК. Нижче одразу видно, що саме піде в тест.</p>
        </div>
        <div class="test-builder-hero__stats">
          ${ctx.metricCard("Закон", "50", "випадкових")}
          ${ctx.metricCard("ОК", String(modules.length), "модулів")}
        </div>
      </section>

      <section class="test-builder-card">
        <div class="test-builder-card__header">
          <div class="section-copy">
            <h2>Склад тесту</h2>
            <p>Конфігуратор перед стартом замість випадкового списку кнопок.</p>
          </div>
          <span class="status-chip" id="test-total-chip">—</span>
        </div>

        <label class="test-law-toggle" for="include-law">
          <span class="test-law-toggle__icon">⚖️</span>
          <span class="test-law-toggle__copy">
            <strong>Законодавство</strong>
            <small>Додати 50 випадкових питань</small>
          </span>
          <input id="include-law" type="checkbox" checked />
        </label>

        <div class="test-builder-divider"></div>

        <div class="test-module-list" id="test-module-config"></div>
      </section>

      <section class="test-start-panel">
        <div>
          <strong id="test-summary-title">Готово до старту</strong>
          <p class="muted" id="test-summary-copy">Перевірте вибір і запускайте тест.</p>
        </div>
        <div class="button-row" id="test-actions"></div>
      </section>

      ${ctx.screenBar("testing")}
    </section>
  `;

  const configNode = ctx.refs.mainPanel.querySelector("#test-module-config");
  const totalChip = ctx.refs.mainPanel.querySelector("#test-total-chip");
  const summaryTitle = ctx.refs.mainPanel.querySelector("#test-summary-title");
  const summaryCopy = ctx.refs.mainPanel.querySelector("#test-summary-copy");
  const includeLawNode = ctx.refs.mainPanel.querySelector("#include-law");
  const selections = {};

  function selectedLevelCount() {
    return Object.values(selections).reduce((sum, levels) => sum + levels.length, 0);
  }

  function updateSummary() {
    const lawIncluded = includeLawNode.checked;
    const levels = selectedLevelCount();
    const modulesWithLevels = Object.values(selections).filter((entry) => entry.length).length;
    totalChip.textContent = `${lawIncluded ? 50 : 0}+${levels} рівн.`;
    summaryTitle.textContent = lawIncluded || levels ? "Готово до старту" : "Нічого не вибрано";
    summaryCopy.textContent = lawIncluded
      ? `Законодавство + ${modulesWithLevels} ОК-мод. / ${levels} рівн.`
      : `${modulesWithLevels} ОК-мод. / ${levels} рівн. без законодавства.`;
  }

  includeLawNode.addEventListener("change", () => {
    ctx.impact("light");
    updateSummary();
  });

  modules.forEach((item) => {
    const initialLevel = item.last_level || item.levels[0]?.level;
    selections[item.name] = initialLevel ? [initialLevel] : [];

    const block = document.createElement("article");
    block.className = "test-module-card";
    block.innerHTML = `
      <div class="test-module-card__top">
        <span class="test-module-card__icon">🎯</span>
        <div>
          <strong>${ctx.escapeHtml(item.label)}</strong>
          <small>Рівні для тестування</small>
        </div>
      </div>
      <div class="test-level-row"></div>
    `;

    const actions = block.querySelector(".test-level-row");
    item.levels.forEach((levelEntry) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `pill-button test-level-pill ${selections[item.name].includes(levelEntry.level) ? "is-selected" : ""}`;
      button.textContent = `L${levelEntry.level}`;
      button.addEventListener("click", () => {
        ctx.impact("light");
        const set = new Set(selections[item.name]);
        if (set.has(levelEntry.level)) {
          set.delete(levelEntry.level);
          button.classList.remove("is-selected");
        } else {
          set.add(levelEntry.level);
          button.classList.add("is-selected");
        }
        selections[item.name] = Array.from(set).sort((a, b) => a - b);
        updateSummary();
      });
      actions.append(button);
    });
    configNode.append(block);
  });

  if (!modules.length) {
    const note = document.createElement("div");
    note.className = "empty-state compact-empty test-empty-state";
    note.innerHTML = `
      <h2>ОК-модулів ще немає</h2>
      <p>Тест можна запустити із законодавством або спочатку додати ОК-модулі.</p>
    `;
    configNode.append(note);
  }

  ctx.refs.mainPanel.querySelector("#test-actions").append(
    ctx.actionButton("Почати тест", async () => {
      try {
        ctx.state.currentView = await ctx.api("/api/test/start", {
          method: "POST",
          body: {
            include_law: includeLawNode.checked,
            module_levels: selections,
          },
        });
        ctx.impact("medium");
        ctx.render();
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "primary"),
  );

  updateSummary();
  bindSharedNav(ctx);
}

export function renderStats(ctx) {
  const { stats } = ctx.state.bootstrap;
  const last = stats.last;

  ctx.setChrome({
    eyebrow: "Статистика",
    title: "Статистика",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen compact-screen stats-screen">
      <section class="ios-section ios-section--hero">
        <div class="section-header">
          <div class="section-copy">
            <span class="eyebrow">Progress</span>
            <h2>${last ? percentLabel(last.percent) : "Поки без тестів"}</h2>
            <p>${last ? `Останній тест: ${last.correct}/${last.total}` : "Запустіть перший тест."}</p>
          </div>
        </div>
        <div class="mini-stats-grid">
          ${ctx.metricCard("Тестів", String(stats.count), "усього")}
          ${ctx.metricCard("Середній", `${stats.avg.toFixed(0)}%`, "бал")}
          ${ctx.metricCard("Останній", last ? `${last.correct}/${last.total}` : "—", last ? last.finished_at_label || "без дати" : "—")}
        </div>
      </section>

      <section class="ios-section">
        <div class="section-header">
          <div class="section-copy">
            <h2>Далі</h2>
            <p>Повторіть слабкі місця або зберіть новий тест.</p>
          </div>
        </div>
        <div class="button-row" id="stats-actions"></div>
      </section>

      ${ctx.screenBar("stats")}
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#stats-actions").append(
    ctx.actionButton("Повторити помилки", ctx.startMistakesSession, "primary"),
    ctx.actionButton("Новий тест", async () => ctx.navigate("testing")),
  );

  bindSharedNav(ctx);
}

export function renderHelp(ctx) {
  const { links, user } = ctx.state.bootstrap;

  ctx.setChrome({
    eyebrow: "Підтримка",
    title: "Підтримка",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen compact-screen support-screen">
      <section class="ios-section ios-section--hero">
        <div class="section-header">
          <div class="section-copy">
            <span class="eyebrow">Support</span>
            <h2>Допомога</h2>
            <p>Контакти, сервісні дії та адмінка — без змішування з навчанням.</p>
          </div>
        </div>
      </section>

      <section class="support-grid">
        ${ctx.actionCard({
          code: "💬",
          title: "Telegram-група",
          body: links.group_url ? "Перейти до спільноти." : "Посилання не налаштовано.",
          meta: links.group_url ? "Відкрити" : "Немає",
          link: links.group_url || "",
          tone: "blue",
        })}
        ${ctx.actionCard({
          code: "👤",
          title: "Адміністратор",
          body: links.admin_url ? "Написати щодо доступу." : "Контакт не налаштовано.",
          meta: links.admin_url ? "Написати" : "Немає",
          link: links.admin_url || "",
          tone: "orange",
        })}
      </section>

      <section class="ios-section">
        <div class="section-header">
          <div class="section-copy">
            <h2>Сервіс</h2>
            <p>Швидкі дії без зайвих екранів.</p>
          </div>
        </div>
        <div class="button-row" id="help-actions"></div>
      </section>

      ${
        user.is_admin
          ? `
            <section class="ios-section">
              <div class="section-header">
                <div class="section-copy">
                  <h2>Адмін</h2>
                  <p>Користувачі та банк питань.</p>
                </div>
              </div>
              <div class="button-row" id="admin-entry-actions"></div>
            </section>
          `
          : ""
      }

      ${ctx.screenBar("help")}
    </section>
  `;

  const helpActions = ctx.refs.mainPanel.querySelector("#help-actions");
  helpActions.append(
    ctx.actionButton("Оновити", async () => ctx.loadBootstrap(true), "primary"),
    ctx.actionButton("Головна", ctx.goHome),
  );

  if (user.is_admin) {
    ctx.refs.mainPanel.querySelector("#admin-entry-actions")?.append(
      ctx.actionButton("Відкрити адмінку", async () => ctx.navigate("admin"), "primary"),
    );
  }

  bindSharedNav(ctx);
}
