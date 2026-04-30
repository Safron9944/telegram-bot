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
      <section class="ios-hero-card">
        <div class="ios-hero-card__top">
          <span class="ios-avatar">${ctx.escapeHtml((user.first_name || "U").slice(0, 1).toUpperCase())}</span>
          <div>
            <span class="eyebrow">Exam Mini App</span>
            <h2>Готові до старту</h2>
          </div>
          <span class="status-chip ${user.access.has_access ? "is-active" : "is-danger"}">${compactAccessLabel(user)}</span>
        </div>
        <div class="mini-stats-grid">
          ${ctx.metricCard("Модулі", String(modules.length), "ОК")}
          ${ctx.metricCard("Тести", String(stats.count), "історія")}
          ${ctx.metricCard("Останній", last ? percentLabel(last.percent) : "—", "результат")}
        </div>
      </section>

      <section class="quick-actions" aria-label="Швидкі дії">
        <button class="quick-action quick-action--primary" type="button" data-screen-target="testing">
          <span>🧪</span>
          <strong>Тестування</strong>
        </button>
        <button class="quick-action" type="button" data-screen-target="stats">
          <span>📊</span>
          <strong>Статистика</strong>
        </button>
        <button class="quick-action" type="button" data-screen-target="help">
          <span>💬</span>
          <strong>Підтримка</strong>
        </button>
      </section>

      <section class="ios-tile-grid" aria-label="Головне меню">
        ${ctx.actionCard({
          code: "📚",
          title: "Навчання",
          body: "Законодавство, ОК-модулі та помилки.",
          meta: `${catalog.law_groups.length} розд.` ,
          screen: "learning",
          tone: "blue",
        })}
        ${ctx.actionCard({
          code: "🧪",
          title: "Тестування",
          body: "Швидка збірка нового тесту.",
          meta: "Старт",
          screen: "testing",
          tone: "purple",
        })}
        ${ctx.actionCard({
          code: "📊",
          title: "Статистика",
          body: "Середній бал і останні спроби.",
          meta: last ? percentLabel(last.percent) : "—",
          screen: "stats",
          tone: "green",
        })}
        ${ctx.actionCard({
          code: "💬",
          title: "Підтримка",
          body: "Група, адміністратор і сервіс.",
          meta: "Help",
          screen: "help",
          tone: "orange",
        })}
      </section>
    </section>
  `;

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}

export function renderLearning(ctx) {
  const { user, catalog } = ctx.state.bootstrap;
  const modules = selectedModules(catalog);

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
            <h2>Навчання</h2>
            <p>Обери блок і запускай підготовку.</p>
          </div>
          <span class="status-chip ${user.access.has_access ? "is-active" : "is-danger"}">${user.access.has_access ? "Старт" : "Перегляд"}</span>
        </div>
        <div class="button-row" id="learning-actions"></div>
      </section>

      <section class="ios-section screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>Законодавство</h2>
            <p>Частини по 50 питань.</p>
          </div>
        </div>
        <div class="list-stack compact-list" id="law-groups-stack"></div>
      </section>

      <section class="ios-section screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>ОК модулі</h2>
            <p>Вибір модулів і рівнів.</p>
          </div>
          <span class="status-chip">${modules.length}</span>
        </div>
        <div class="section-stack" id="ok-modules-stack"></div>
      </section>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#learning-actions").append(
    ctx.actionButton("Помилки", ctx.startMistakesSession, "primary"),
    ctx.actionButton("Тест", async () => ctx.navigate("testing")),
  );

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
        renderLearning(ctx);
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
    return;
  }

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
    okStack.append(card);
  });
}

export function renderLawParts(ctx) {
  const group = ctx.state.selectedLawGroup;
  if (!group) {
    ctx.state.currentScreen = "learning";
    renderLearning(ctx);
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
    <section class="app-screen testing-screen compact-screen">
      <section class="ios-section ios-section--hero">
        <div class="section-header">
          <div class="section-copy">
            <span class="eyebrow">Test</span>
            <h2>Новий тест</h2>
            <p>Законодавство + вибрані рівні ОК.</p>
          </div>
          <span class="status-chip">${modules.length} мод.</span>
        </div>
      </section>

      <section class="ios-section screen-block">
        <div class="inline-form test-config-form ios-form">
          <label class="ios-switch-row" for="include-law">
            <span>
              <strong>Законодавство</strong>
              <small>50 випадкових питань</small>
            </span>
            <input id="include-law" type="checkbox" checked />
          </label>
          <div id="test-module-config" class="list-stack compact-list"></div>
          <div class="button-row sticky-actions" id="test-actions"></div>
        </div>
      </section>
    </section>
  `;

  const configNode = ctx.refs.mainPanel.querySelector("#test-module-config");
  const selections = {};

  modules.forEach((item) => {
    const initialLevel = item.last_level || item.levels[0]?.level;
    selections[item.name] = initialLevel ? [initialLevel] : [];

    const block = document.createElement("article");
    block.className = "list-item ios-list-item";
    block.innerHTML = `
      <div class="list-item__main">
        <strong>${ctx.escapeHtml(item.label)}</strong>
        <span class="list-item__meta">Виберіть рівні</span>
      </div>
      <div class="button-row"></div>
    `;

    const actions = block.querySelector(".button-row");
    item.levels.forEach((levelEntry) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `pill-button ${selections[item.name].includes(levelEntry.level) ? "is-selected" : ""}`;
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
      });
      actions.append(button);
    });
    configNode.append(block);
  });

  if (!modules.length) {
    const note = document.createElement("div");
    note.className = "empty-state compact-empty";
    note.innerHTML = `
      <h2>Немає модулів</h2>
      <p>Додайте ОК-модуль у «Навчанні».</p>
    `;
    configNode.append(note);
  }

  ctx.refs.mainPanel.querySelector("#test-actions").append(
    ctx.actionButton("Почати тест", async () => {
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
    }, "primary"),
    ctx.actionButton("Навчання", async () => ctx.navigate("learning")),
  );
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
          ${ctx.metricCard("Відсоток", last ? percentLabel(last.percent) : "—", "останній")}
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
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#stats-actions").append(
    ctx.actionButton("Навчання", async () => ctx.navigate("learning"), "primary"),
    ctx.actionButton("Новий тест", async () => ctx.navigate("testing")),
  );
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
            <p>Група, адміністратор та оновлення даних.</p>
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

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}
