function page(ctx, title, description, content) {
  ctx.setChrome({ showBack: true });
  ctx.refs.mainPanel.innerHTML = `
    <section class="screen-content admin-top-level-screen admin-management-screen">
      <header class="admin-mobile-header"><h1>${title}</h1><p>${description}</p></header>
      <div class="admin-prototype-body">${content}</div>
    </section>`;
}

function errorWithRetry(ctx, node, retry) {
  if (!node.isConnected) return;
  node.innerHTML = '<p class="admin-help">Не вдалося завантажити дані.</p><button class="btn btn--ghost" type="button">Спробувати ще раз</button>';
  node.querySelector('button')?.addEventListener('click', retry);
}

export function renderAdminTariffs(ctx) {
  page(ctx, 'Доступ і тарифи', 'Ціни доступу в Telegram Stars.', `
    <section class="admin-profile-block admin-form-block">
      <h2>Повний пакет</h2>
      <p class="admin-help">Ціна придбання повного доступу. Окремі покупки мають власну вартість.</p>
      <form id="admin-full-price-form">
        <label class="field"><span class="field__label">Вартість, ⭐</span>
          <input class="input" id="admin-price-full" type="number" min="1" step="1" required disabled>
        </label>
        <button class="btn btn--primary btn--block" type="submit" disabled>Зберегти ціну пакета</button>
      </form>
      <div id="admin-tariff-status" role="status" aria-live="polite">Завантажуємо ціну…</div>
    </section>
    <section class="admin-form-section"><h2>Окремі розділи</h2>
      <p class="admin-help">Відкрийте розділ, щоб змінити його ціну. Покупка діє назавжди.</p>
      <div id="admin-section-prices" class="admin-section-list" aria-live="polite">Завантажуємо розділи…</div>
    </section>`);
  const form = ctx.refs.mainPanel.querySelector('#admin-full-price-form');
  const input = form.querySelector('input');
  const button = form.querySelector('button');
  const status = ctx.refs.mainPanel.querySelector('#admin-tariff-status');
  const list = ctx.refs.mainPanel.querySelector('#admin-section-prices');
  let saving = false;
  const loadPrice = async () => {
    status.textContent = 'Завантажуємо ціну…';
    try {
      const data = await ctx.api('/api/admin/settings');
      if (!form.isConnected) return;
      input.value = data.price_full;
      input.disabled = button.disabled = false;
      status.textContent = '';
    } catch (error) {
      if (!form.isConnected) return;
      errorWithRetry(ctx, status, loadPrice);
    }
  };
  const loadSections = async () => {
    list.textContent = 'Завантажуємо розділи…';
    try {
      const data = await ctx.api('/api/admin/sections');
      if (!list.isConnected) return;
      const sections = data.items || [];
      list.innerHTML = '';
      sections.forEach(section => {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'admin-section-row';
        row.innerHTML = `<span class="admin-section-row__body"><strong>${ctx.escapeHtml(section.title)}</strong><small>${section.price > 0 ? `${Number(section.price)} ⭐ · назавжди` : 'Безкоштовно'}</small></span><span aria-hidden="true">›</span>`;
        row.addEventListener('click', () => {
          ctx.state.selectedAdminSection = section;
          ctx.navigate('admin-section-settings');
        });
        list.append(row);
      });
      if (!sections.length) list.textContent = 'Розділів ще немає. Додайте їх у «Матеріалах».';
    } catch (error) {
      errorWithRetry(ctx, list, loadSections);
    }
  };
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (saving || input.disabled) return;
    const price = Number(input.value);
    if (!Number.isInteger(price) || price < 1) return ctx.setMessage('error', 'Ціна має бути цілим числом від 1 ⭐.');
    saving = true;
    input.disabled = button.disabled = true;
    status.textContent = 'Зберігаємо…';
    try {
      await ctx.api('/api/admin/settings', { method: 'POST', body: { price_full: price } });
      ctx.state.bootstrap.payment_prices.full = price;
      if (form.isConnected) status.textContent = 'Ціну пакета збережено.';
    } catch (error) {
      if (form.isConnected) status.textContent = error.message;
    } finally {
      saving = false;
      input.disabled = button.disabled = false;
    }
  });
  void loadPrice();
  void loadSections();
}

export function renderAdminSettings(ctx) {
  page(ctx, 'Налаштування', 'Загальні налаштування застосунку.', `
    <section class="admin-profile-block admin-form-block">
      <h2>Підтримка</h2>
      <p class="admin-help">Куди звертаються користувачі з кнопки підтримки.</p>
      <form id="admin-contact-form">
        <label class="field"><span class="field__label">Контакт або посилання підтримки</span>
          <input class="input" id="admin-contact-url" type="text" placeholder="@username або https://t.me/username" maxlength="500" disabled>
        </label>
        <button class="btn btn--primary btn--block" type="submit" disabled>Зберегти контакт</button>
      </form>
      <div id="admin-settings-status" role="status" aria-live="polite">Завантажуємо налаштування…</div>
    </section>
    ${ctx.group({header:'Показ розділів', footer:'Назва, ціна й показ кожного розділу налаштовуються в його картці.', children:ctx.cell({title:'Налаштування розділів',subtitle:'Відкрити список матеріалів',iconName:'folder',screen:'admin-attestation-banks',tint:'gray'})})}`);
  ctx.bindInlineTargets(ctx.refs.mainPanel, { navigate: ctx.navigate });
  const form = ctx.refs.mainPanel.querySelector('#admin-contact-form');
  const input = form.querySelector('input');
  const button = form.querySelector('button');
  const status = ctx.refs.mainPanel.querySelector('#admin-settings-status');
  let saving = false;
  const load = async () => {
    status.textContent = 'Завантажуємо налаштування…';
    try {
      const data = await ctx.api('/api/admin/settings');
      if (!form.isConnected) return;
      input.value = data.admin_contact_url || '';
      input.disabled = button.disabled = false;
      status.textContent = '';
    } catch (error) {
      errorWithRetry(ctx, status, load);
    }
  };
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (saving || input.disabled) return;
    const contact = input.value.trim();
    if (contact) {
      let valid = /^@[a-zA-Z0-9_]{5,32}$/.test(contact);
      if (!valid) {
        try { valid = ['https:', 'http:', 'tg:'].includes(new URL(contact).protocol); } catch (_) {}
      }
      if (!valid) return ctx.setMessage('error', 'Вкажіть @username або коректне посилання підтримки.');
    }
    saving = true;
    input.disabled = button.disabled = true;
    status.textContent = 'Зберігаємо…';
    try {
      await ctx.api('/api/admin/settings', { method: 'POST', body: { admin_contact_url: contact } });
      await ctx.loadBootstrap();
      if (ctx.state.currentScreen === 'admin-settings') ctx.setMessage('success', 'Контакт підтримки збережено.');
    } catch (error) {
      if (form.isConnected) status.textContent = error.message;
    } finally {
      saving = false;
      input.disabled = button.disabled = false;
    }
  });
  void load();
}
