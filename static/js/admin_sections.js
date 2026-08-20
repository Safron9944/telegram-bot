function selected(ctx) {
  return ctx.state.selectedAdminSection || null;
}

function sectionUrl(ctx, suffix = "") {
  const key = selected(ctx)?.key || "";
  return key ? `/api/admin/sections/${encodeURIComponent(key)}${suffix}` : "";
}

export function renderAdminSection(ctx) {
  ctx.setChrome({ showBack: true });
  const section = selected(ctx);
  if (!section) return void ctx.goBack();
  const actions = [
    ctx.cell({ title: "Основне", subtitle: "Назва, показ і ціна доступу", iconName: "settings", tint: "blue", screen: "admin-section-settings" }),
  ];
  if (section.kind === "attestation") {
    actions.push(ctx.cell({ title: "Підрозділи", subtitle: "Перегляд і зміна назв", iconName: "folder", tint: "green", screen: "admin-section-topics" }));
    actions.push(ctx.cell({ title: "Питання", subtitle: `${section.questions_count || 0} питань`, iconName: "edit", tint: "purple", screen: "admin-section-questions" }));
  } else if (section.content_screen) {
    actions.push(ctx.cell({ title: section.content_label || "Вміст", subtitle: "Керування матеріалами та питаннями", iconName: "edit", tint: "purple", screen: section.content_screen }));
  }
  actions.push(ctx.cell({ title: "Порядок", subtitle: "Перемістити на головному екрані", iconName: "document", tint: "gray", screen: "admin-section-order" }));

  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content admin-section-menu-screen">
      <h1 class="page-title">${ctx.escapeHtml(section.title)}</h1>
      <p class="page-subtitle">Оберіть, що саме потрібно змінити.</p>
      ${ctx.group({ children: actions.join("") })}
      ${section.deletable ? `
        <div class="group">
          <div class="group__label">Небезпечна дія</div>
          <div class="group__list">
            <button class="cell" type="button" id="admin-section-delete">
              <span class="cell__body"><span class="cell__title">Видалити розділ</span><span class="cell__subtitle">Разом з усіма його питаннями</span></span>
              <span class="cell__chevron" aria-hidden="true"></span>
            </button>
          </div>
        </div>` : ""}
    </section>`;
  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
  ctx.refs.mainPanel.querySelector("#admin-section-delete")?.addEventListener("click", async () => {
    if (!window.confirm(`Видалити «${section.title}» разом із усіма питаннями?`)) return;
    try {
      await ctx.api(sectionUrl(ctx), { method: "DELETE" });
      ctx.state.selectedAdminSection = null;
      ctx.state.selectedAttestationAdminBankId = null;
      await ctx.goBack();
      ctx.setMessage("success", "Розділ видалено.");
    } catch (error) { ctx.setMessage("error", error.message); }
  });
}

export function renderAdminSectionSettings(ctx) {
  ctx.setChrome({ showBack: true });
  const section = selected(ctx);
  if (!section) return void ctx.goBack();
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Основне</h1>
      <p class="page-subtitle">${ctx.escapeHtml(section.title)}</p>
      <form class="admin-section-form" id="admin-section-settings-form">
        <label class="field"><span class="field__label">Назва розділу</span><input class="input" id="admin-section-title" maxlength="160" value="${ctx.escapeHtml(section.title)}"></label>
        <label class="field"><span class="field__label">Ціна доступу, ⭐</span><input class="input" id="admin-section-price" type="number" min="0" step="1" value="${ctx.escapeHtml(section.price)}"><span class="field__hint">0 — безкоштовно. Покупка діє назавжди.</span></label>
        <div class="group__list">
          <label class="cell admin-setting-row">
            <span class="cell__body"><span class="cell__title">Показувати користувачам</span><span class="cell__subtitle">Прихований розділ залишається в базі</span></span>
            <span class="switch"><input id="admin-section-visible" type="checkbox" ${section.visible ? "checked" : ""}><span class="switch__track"></span></span>
          </label>
        </div>
        ${section.key === "support" ? '<label class="field"><span class="field__label">Посилання для підтримки</span><input class="input" id="admin-section-contact" placeholder="https://t.me/username або @username"><span class="field__hint">Куди відкривати кнопку звернення до адміністратора.</span></label>' : ""}
        <button class="btn btn--primary btn--block" type="submit">Зберегти</button>
      </form>
    </section>`;
  ctx.refs.mainPanel.querySelector("#admin-section-settings-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const title = ctx.refs.mainPanel.querySelector("#admin-section-title").value.trim();
    const price = Number(ctx.refs.mainPanel.querySelector("#admin-section-price").value);
    const visible = ctx.refs.mainPanel.querySelector("#admin-section-visible").checked;
    if (!title || !Number.isInteger(price) || price < 0) return ctx.setMessage("error", "Перевірте назву та ціну.");
    try {
      const payload = await ctx.api(sectionUrl(ctx), { method: "PATCH", body: { title, price, visible } });
      const contact = ctx.refs.mainPanel.querySelector("#admin-section-contact")?.value.trim();
      if (contact !== undefined) await ctx.api("/api/admin/settings", { method: "POST", body: { admin_contact_url: contact } });
      ctx.state.selectedAdminSection = payload.section;
      ctx.state.selectedAttestationAdminBank = payload.section;
      await ctx.loadBootstrap();
      ctx.setMessage("success", "Налаштування збережено.");
    } catch (error) { ctx.setMessage("error", error.message); }
  });
  const contactInput = ctx.refs.mainPanel.querySelector("#admin-section-contact");
  if (contactInput) {
    void ctx.api("/api/admin/settings").then((payload) => { contactInput.value = payload.admin_contact_url || ""; }).catch(() => {});
  }
}

export function renderAdminSectionOrder(ctx) {
  ctx.setChrome({ showBack: true });
  const section = selected(ctx);
  if (!section) return void ctx.goBack();
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Порядок</h1>
      <p class="page-subtitle">${ctx.escapeHtml(section.title)}</p>
      <div class="group"><div class="group__list">
        <button class="cell" type="button" data-direction="up"><span class="cell__body"><span class="cell__title">↑ Перемістити вище</span></span></button>
        <button class="cell" type="button" data-direction="down"><span class="cell__body"><span class="cell__title">↓ Перемістити нижче</span></span></button>
      </div></div>
    </section>`;
  ctx.refs.mainPanel.querySelectorAll("[data-direction]").forEach((button) => button.addEventListener("click", async () => {
    try {
      await ctx.api(sectionUrl(ctx, "/move"), { method: "POST", body: { direction: button.dataset.direction } });
      await ctx.loadBootstrap();
      ctx.setMessage("success", button.dataset.direction === "up" ? "Розділ переміщено вище." : "Розділ переміщено нижче.");
    } catch (error) { ctx.setMessage("error", error.message); }
  }));
}

export function renderAdminSectionTopics(ctx) {
  ctx.setChrome({ showBack: true });
  const section = selected(ctx);
  if (!section?.bank_id) return void ctx.goBack();
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Підрозділи</h1>
      <p class="page-subtitle">Натисніть підрозділ, щоб змінити його назву.</p>
      <div class="group"><div class="group__list" id="admin-section-topic-list"><div class="empty empty--inline"><h2>Завантажуємо…</h2></div></div></div>
    </section>`;
}

export async function loadAdminSectionTopics(ctx) {
  if (ctx.state.currentScreen !== "admin-section-topics") return;
  const section = selected(ctx);
  const list = document.querySelector("#admin-section-topic-list");
  if (!section?.bank_id || !list) return;
  try {
    const payload = await ctx.api(`/api/admin/attestation-banks/${section.bank_id}/questions?offset=0&limit=1`);
    list.innerHTML = "";
    (payload.topics || []).forEach((topic) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "cell";
      row.innerHTML = `<span class="cell__body"><span class="cell__title">${ctx.escapeHtml(topic.topic)}</span><span class="cell__subtitle">${ctx.escapeHtml(topic.questions_count)} питань</span></span><span class="cell__chevron" aria-hidden="true"></span>`;
      row.addEventListener("click", () => {
        ctx.state.selectedAdminSectionTopic = topic;
        ctx.navigate("admin-section-topic-edit");
      });
      list.append(row);
    });
    if (!payload.topics?.length) list.innerHTML = '<div class="empty empty--inline"><h2>Підрозділів немає</h2></div>';
  } catch (error) { list.innerHTML = `<div class="empty empty--inline"><h2>Помилка</h2><p>${ctx.escapeHtml(error.message)}</p></div>`; }
}

export function renderAdminSectionTopicEdit(ctx) {
  ctx.setChrome({ showBack: true });
  const topic = ctx.state.selectedAdminSectionTopic;
  if (!selected(ctx)?.bank_id || !topic) return void ctx.goBack();
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content">
      <h1 class="page-title">Назва підрозділу</h1>
      <p class="page-subtitle">${ctx.escapeHtml(topic.questions_count)} питань залишаться на місці.</p>
      <form class="admin-section-form" id="admin-section-topic-form">
        <label class="field"><span class="field__label">Назва</span><input class="input" id="admin-section-topic-name" maxlength="240" value="${ctx.escapeHtml(topic.topic)}"></label>
        <button class="btn btn--primary btn--block" type="submit">Зберегти назву</button>
      </form>
    </section>`;
  ctx.refs.mainPanel.querySelector("#admin-section-topic-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = ctx.refs.mainPanel.querySelector("#admin-section-topic-name").value.trim();
    if (!name) return ctx.setMessage("error", "Вкажіть назву підрозділу.");
    try {
      await ctx.api(sectionUrl(ctx, "/topics"), { method: "PATCH", body: { old_topic: topic.topic, new_topic: name } });
      ctx.state.selectedAdminSectionTopic = { ...topic, topic: name };
      ctx.setMessage("success", "Назву підрозділу змінено.");
    } catch (error) { ctx.setMessage("error", error.message); }
  });
}
