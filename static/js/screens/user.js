function openLawParts(ctx, group) {
  ctx.state.selectedLawGroup = group;
  ctx.navigate("law-parts");
}

export function renderHome(ctx) {
  const { user, catalog, stats } = ctx.state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);
  const accessClass = user.access.has_access ? "is-active" : "is-danger";
  const lastResult = stats.last ? `${stats.last.percent.toFixed(1)}%` : "Ще не було";

  ctx.setChrome({
    eyebrow: "Підготовка",
    title: "Головна",
    subtitle: "",
    showBack: false,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen home-screen">
      <div class="screen-hero">
        <span class="eyebrow">Підготовка</span>
        <h2>Оберіть розділ</h2>
        <p>Навчання, тестування, прогрес і підтримка відкриваються як окремі екрани — без змішування блоків на одній сторінці.</p>
      </div>

      <section class="app-menu-grid" aria-label="Головне меню">
        ${ctx.actionCard({
          code: "LE",
          title: "Навчання",
          body: "Окремий екран для законодавства, модулів ОК і повторення помилок.",
          meta: `${catalog.law_groups.length} розділів`,
          screen: "learning",
        })}
        ${ctx.actionCard({
          code: "TE",
          title: "Тестування",
          body: "Окремий екран для конфігурації та запуску нового тесту.",
          meta: "Новий екран",
          screen: "testing",
        })}
        ${ctx.actionCard({
          code: "PR",
          title: "Прогрес",
          body: "Результати тестів, середній бал і остання спроба.",
          meta: lastResult,
          screen: "stats",
        })}
        ${ctx.actionCard({
          code: "HE",
          title: "Підтримка",
          body: "Контакти, Telegram-група, оновлення даних та адмінський режим.",
          meta: "Допомога",
          screen: "help",
        })}
      </section>

      <section class="surface compact-status">
        <div class="chips-row">
          <span class="status-chip ${accessClass}">${user.access.has_access ? "Доступ активний" : "Доступ обмежений"}</span>
          <span class="status-chip">${selectedModules.length} модулів ОК</span>
          <span class="status-chip">${stats.count} тестів</span>
        </div>
      </section>
    </section>
  `;

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}
export function renderLearning(ctx) {
  const { user, catalog } = ctx.state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);

  ctx.setChrome({
    eyebrow: "Навчання",
    title: "Навчання",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen learning-screen">
      <div class="screen-hero">
        <span class="eyebrow">Новий екран</span>
        <h2>Навчання</h2>
        <p>Оберіть законодавство, модуль ОК або повторення помилок. Це окремий робочий екран, не змішаний із головною.</p>
        <div class="screen-hero__actions" id="learning-actions"></div>
      </div>

      <section class="surface screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>Розділи законодавства</h2>
            <p>Великі розділи автоматично діляться на частини по 50 питань.</p>
          </div>
          <span class="status-chip ${user.access.has_access ? "is-active" : "is-danger"}">${user.access.has_access ? "Можна стартувати" : "Лише перегляд"}</span>
        </div>
        <div class="list-stack" id="law-groups-stack"></div>
      </section>

      <section class="surface screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>Модулі ОК</h2>
            <p>Збережіть потрібні модулі і запускайте рівні окремо.</p>
          </div>
          <span class="status-chip">${selectedModules.length} модулів</span>
        </div>
        <div class="section-stack" id="ok-modules-stack"></div>
      </section>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#learning-actions").append(
    ctx.actionButton("Перевірити помилки", ctx.startMistakesSession, "primary"),
    ctx.actionButton("Налаштувати тест", async () => ctx.navigate("testing")),
  );

  const lawStack = ctx.refs.mainPanel.querySelector("#law-groups-stack");
  catalog.law_groups.forEach((group) => {
    const node = document.createElement("article");
    node.className = "list-item";
    node.innerHTML = `
      <div class="list-item__main">
        <span class="list-item__eyebrow">Розділ</span>
        <strong>${ctx.escapeHtml(group.title)}</strong>
        <span class="list-item__meta">${group.count} питань у банку</span>
      </div>
      <div class="button-row"></div>
    `;

    const actions = node.querySelector(".button-row");
    const partCount = Math.ceil(group.count / 50);
    if (partCount <= 1) {
      actions.append(
        ctx.actionButton("Підготовка", async () => ctx.startLearning({ kind: "law", group_key: group.key, part: 1 }), "primary"),
      );
    } else {
      actions.append(ctx.actionButton(`Частини (${partCount})`, async () => openLawParts(ctx, group), "primary"));
    }
    actions.append(ctx.actionButton("Рандом 50", async () => ctx.startLearning({ kind: "lawrand", group_key: group.key })));
    lawStack.append(node);
  });

  const okStack = ctx.refs.mainPanel.querySelector("#ok-modules-stack");
  const editor = document.createElement("section");
  editor.className = "inline-form module-picker";
  editor.innerHTML = `
    <div class="field">
      <label>Оберіть модулі ОК</label>
      <div class="chips-row" id="module-selector"></div>
    </div>
    <div class="button-row" id="module-selector-actions"></div>
  `;
  okStack.append(editor);

  const selectedNames = new Set(selectedModules.map((item) => item.name));
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
    ctx.actionButton("Зберегти модулі", async () => {
      try {
        const response = await ctx.api("/api/preferences/ok-modules", {
          method: "POST",
          body: { modules: Array.from(selectedNames) },
        });
        ctx.state.bootstrap.user = response.user;
        ctx.state.bootstrap.catalog = response.catalog;
        ctx.setMessage("success", "Модулі збережено.");
        ctx.impact("medium");
        renderLearning(ctx);
      } catch (error) {
        ctx.setMessage("error", error.message);
      }
    }, "primary"),
  );

  if (!selectedModules.length) {
    const note = document.createElement("div");
    note.className = "empty-state";
    note.innerHTML = `
      <h2>Поки що без модулів</h2>
      <p>Позначте потрібні модулі вище, і тут з’являться кнопки запуску за рівнями.</p>
    `;
    okStack.append(note);
    return;
  }

  selectedModules.forEach((item) => {
    const card = document.createElement("article");
    card.className = "list-item";
    card.innerHTML = `
      <div class="list-item__main">
        <span class="list-item__eyebrow">Модуль ОК</span>
        <strong>${ctx.escapeHtml(item.label)}</strong>
        <span class="list-item__meta">Доступні рівні: ${item.levels.map((entry) => entry.level).join(", ")}</span>
      </div>
      <div class="button-row"></div>
    `;

    const buttons = card.querySelector(".button-row");
    item.levels.forEach((levelEntry) => {
      buttons.append(
        ctx.actionButton(
          `Рівень ${levelEntry.level}`,
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
    subtitle: `${group.count} питань у розділі.`,
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Частини розділу</h2>
          <p>Оберіть конкретну частину або випадкову добірку з цього блоку.</p>
        </div>
      </div>
      <div class="button-row" id="law-parts-actions"></div>
    </section>
    <section class="cards-grid" id="law-parts-grid"></section>
  `;

  ctx.refs.mainPanel.querySelector("#law-parts-actions").append(
    ctx.actionButton("До навчання", ctx.goBack),
    ctx.actionButton("Рандом 50", async () => ctx.startLearning({ kind: "lawrand", group_key: group.key }), "primary"),
  );

  const grid = ctx.refs.mainPanel.querySelector("#law-parts-grid");
  for (let part = 1; part <= partCount; part += 1) {
    const start = (part - 1) * 50 + 1;
    const end = Math.min(part * 50, group.count);
    const card = document.createElement("article");
    card.className = "grid-card";
    card.innerHTML = `
      <span class="eyebrow">Частина ${part}</span>
      <strong>${start}-${end}</strong>
      <p class="muted">Питання ${start}–${end} цього розділу.</p>
    `;

    const actions = document.createElement("div");
    actions.className = "button-row";
    actions.append(ctx.actionButton("Відкрити", async () => ctx.startLearning({ kind: "law", group_key: group.key, part }), "primary"));
    card.append(actions);
    grid.append(card);
  }
}

export function renderTesting(ctx) {
  const { catalog } = ctx.state.bootstrap;
  const selectedModules = catalog.ok_modules.filter((item) => item.selected);

  ctx.setChrome({
    eyebrow: "Тестування",
    title: "Тестування",
    subtitle: "",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="app-screen testing-screen">
      <div class="screen-hero">
        <span class="eyebrow">Новий екран</span>
        <h2>Тестування</h2>
        <p>Налаштуйте законодавство і рівні модулів ОК, а потім запустіть тест. Цей екран відокремлений від навчання.</p>
      </div>

      <section class="surface screen-block">
        <div class="section-header">
          <div class="section-copy">
            <h2>Конфігурація тесту</h2>
            <p>Для кожного модуля можна лишити один або кілька рівнів. Якщо рівні не вибрані, модуль не потрапить у тест.</p>
          </div>
        </div>
        <div class="inline-form test-config-form">
          <div class="check-row">
            <input id="include-law" type="checkbox" checked />
            <label for="include-law">Додати 50 випадкових питань із законодавства</label>
          </div>
          <div id="test-module-config" class="list-stack"></div>
          <div class="button-row" id="test-actions"></div>
        </div>
      </section>
    </section>
  `;

  const configNode = ctx.refs.mainPanel.querySelector("#test-module-config");
  const selections = {};

  selectedModules.forEach((item) => {
    const initialLevel = item.last_level || item.levels[0]?.level;
    selections[item.name] = initialLevel ? [initialLevel] : [];

    const block = document.createElement("article");
    block.className = "list-item";
    block.innerHTML = `
      <div class="list-item__main">
        <span class="list-item__eyebrow">Модуль ОК</span>
        <strong>${ctx.escapeHtml(item.label)}</strong>
        <span class="list-item__meta">Зніміть усі рівні, якщо хочете виключити модуль із цього тесту.</span>
      </div>
      <div class="button-row"></div>
    `;

    const actions = block.querySelector(".button-row");
    item.levels.forEach((levelEntry) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `pill-button ${selections[item.name].includes(levelEntry.level) ? "is-selected" : ""}`;
      button.textContent = `Рівень ${levelEntry.level}`;
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

  if (!selectedModules.length) {
    const note = document.createElement("div");
    note.className = "empty-state";
    note.innerHTML = `
      <h2>Немає модулів для тесту</h2>
      <p>Спочатку оберіть хоча б один модуль ОК у розділі «Навчання».</p>
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
    ctx.actionButton("До навчання", async () => ctx.navigate("learning")),
  );
}
export function renderStats(ctx) {
  const { stats } = ctx.state.bootstrap;
  const last = stats.last;

  ctx.setChrome({
    eyebrow: "Прогрес",
    title: "Прогрес",
    subtitle: "Коротка історія спроб і ваш середній результат.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Зведення</h2>
          <p>Останні тести і середній результат підтягуються із збереженої історії.</p>
        </div>
      </div>
      <div class="metrics-grid">
        ${ctx.metricCard("Тестів", String(stats.count), "Останні 50 спроб")}
        ${ctx.metricCard("Середній результат", `${stats.avg.toFixed(1)}%`, "За завершеними тестами")}
        ${ctx.metricCard("Останній тест", last ? `${last.correct}/${last.total}` : "—", last ? last.finished_at_label || "Без дати" : "Ще не було")}
        ${ctx.metricCard("Останній відсоток", last ? `${last.percent.toFixed(1)}%` : "—", last ? "Останній завершений тест" : "Немає даних")}
      </div>
    </section>

    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Що робити далі</h2>
          <p>Після короткої перевірки прогресу можна швидко повернутись у навчання або зібрати новий тест.</p>
        </div>
      </div>
      <div class="button-row" id="stats-actions"></div>
    </section>
  `;

  ctx.refs.mainPanel.querySelector("#stats-actions").append(
    ctx.actionButton("До навчання", async () => ctx.navigate("learning"), "primary"),
    ctx.actionButton("Новий тест", async () => ctx.navigate("testing")),
  );
}

export function renderHelp(ctx) {
  const { links, user } = ctx.state.bootstrap;

  ctx.setChrome({
    eyebrow: "Підтримка",
    title: "Підтримка",
    subtitle: "Контакти і зовнішні переходи без зайвих веб-елементів.",
    showBack: true,
  });

  ctx.refs.mainPanel.innerHTML = `
    <section class="support-grid">
      ${ctx.actionCard({
        code: "TG",
        title: "Telegram-група",
        body: links.group_url ? "Швидкий перехід до спільноти без виходу з контексту навчання." : "Посилання на групу поки не налаштовано.",
        meta: links.group_url ? "Відкрити" : "Недоступно",
        link: links.group_url || "",
      })}
      ${ctx.actionCard({
        code: "AD",
        title: "Адміністратор",
        body: links.admin_url ? "Написати адміну, якщо потрібен доступ або є питання по роботі бота." : "Контакт адміністратора поки не налаштовано.",
        meta: links.admin_url ? "Написати" : "Недоступно",
        link: links.admin_url || "",
      })}
    </section>

    <section class="surface">
      <div class="section-header">
        <div class="section-copy">
          <h2>Сервісні дії</h2>
          <p>Тут зібрані допоміжні переходи, які не повинні заважати основному сценарію використання.</p>
        </div>
      </div>
      <div class="button-row" id="help-actions"></div>
    </section>

    ${
      user.is_admin
        ? `
          <section class="surface">
            <div class="section-header">
              <div class="section-copy">
                <h2>Режим адміністратора</h2>
                <p>Адмінські інструменти винесені в окремий режим, щоб не змішувати їх з користувацьким інтерфейсом.</p>
              </div>
            </div>
            <div class="button-row" id="admin-entry-actions"></div>
          </section>
        `
        : ""
    }
  `;

  const helpActions = ctx.refs.mainPanel.querySelector("#help-actions");
  helpActions.append(
    ctx.actionButton("Оновити дані", async () => ctx.loadBootstrap(true), "primary"),
    ctx.actionButton("На головну", ctx.goHome),
  );

  if (user.is_admin) {
    ctx.refs.mainPanel.querySelector("#admin-entry-actions")?.append(
      ctx.actionButton("Відкрити режим адміністратора", async () => ctx.navigate("admin"), "primary"),
    );
  }

  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
}
