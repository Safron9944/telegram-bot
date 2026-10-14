export function paymentAmount(payment = {}) {
  if (payment.amount == null) return 'Сума невідома';
  const value = new Intl.NumberFormat('uk-UA', { maximumFractionDigits: 2 }).format(Number(payment.amount) / (payment.currency === 'UAH' ? 100 : 1));
  return `${value} ${payment.currency === 'XTR' ? '⭐' : payment.currency === 'UAH' ? '₴' : payment.currency || ''}`.trim();
}

export function paymentSource(source) {
  return ({ telegram: 'Telegram Stars', telegram_sync: 'Telegram Stars · звірено', legacy_telegram: 'Перенесена оплата Telegram', admin_external: 'Підтверджено адміністратором', admin: 'Підтверджено адміністратором', imported: 'Перенесена оплата Telegram' })[source] || 'Джерело потребує перевірки';
}

export function clientListSummary(user = {}) {
  const s=user.commerce_summary;
  if(!s)return {product:'Дані ще не отримано',evidence:'',amount:'',attention:false};
  const names=items=>(items||[]).filter(Boolean).join(', ');
  const manualProducts=(s.manual_titles||[]).filter(Boolean);
  const known=Object.entries(s.payment_totals||{}).filter(([,amount])=>amount>0).map(([currency,amount])=>paymentAmount({currency,amount})).join(' + ');
  if(s.paid_count){
    const sources=[...new Set((s.purchase_sources||[]).map(source=>({telegram:'Telegram Stars',telegram_sync:'Telegram Stars',legacy_telegram:'Telegram · старий запис',admin_external:'Переказ · підтвердив адміністратор'})[source]||'Джерело не перевірене'))].join(', ');
    return {product:names(s.purchase_titles)||'Підтверджена покупка',evidence:['Оплачено',sources,!manualProducts.length&&s.manual_count?`Ручних дозволів: ${s.manual_count}`:null].filter(Boolean).join(' · '),amount:known?(s.unknown_amount_count?`${known} + невідома сума`:known):'Сума невідома',attention:!!s.attention_count,manualProducts};
  }
  if(s.legacy_count)return {product:names(s.legacy_titles)||'Старий доступ',evidence:'Старий доступ · перевірити оплату',amount:'',attention:true,manualProducts};
  if(s.manual_count)return {product:names(s.manual_titles)||'Ручний доступ',evidence:'Доступ вручну · оплата не зафіксована',amount:'',attention:false};
  return {product:'Оплата не зафіксована',evidence:'',amount:'',attention:false};
}

export function adminDate(value) {
  if (!value) return 'Дата невідома';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Дата невідома' : new Intl.DateTimeFormat('uk-UA', { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

export function eventLabel(event) {
  return ({ grant_added:'Надано ручний дозвіл',grant_removed:'Скасовано ручний дозвіл',payment_confirmed:'Підтверджено оплату',payment_refunded:'Платіж повернено',purchases_restored:'Відновлено куплений доступ',section_shown:'Матеріал показано клієнту',section_hidden:'Матеріал приховано для клієнта',user_deleted:'Видалено профіль',grant: 'Надано ручний дозвіл', revoke: 'Скасовано ручний дозвіл', manual_grant: 'Надано ручний дозвіл', manual_revoke: 'Скасовано ручний дозвіл', restore: 'Відновлено куплений доступ', restore_purchases: 'Відновлено куплений доступ', payment: 'Записано оплату', external_payment: 'Підтверджено зовнішню оплату', payment_recorded: 'Записано оплату' })[event.action] || 'Зміна доступу';
}

export function renderEvents(ctx, events = [], products = []) {
  const e = ctx.escapeHtml;
  return events.length ? events.map(event => {const product=event.product_title||products.find(p=>p.product_key===event.product_key)?.product_title||({full:'Повний пакет',cases:'Пакет атестації','section:customs':'Митні компетенції','section:cases':'Кейси','section:test_questions':'Тестові питання','section:question_search':'Пошук питань'})[event.product_key]||event.product_key;return `<div class="aw-event"><strong>${e(eventLabel(event))}${product?` · ${e(product)}`:''}</strong><p>${e(event.reason || '')}</p><small>${e(adminDate(event.created_at))} · адміністратор ${e(event.actor_id ?? '—')}${event.user_id ? ` · клієнт ${e(event.user_id)}` : ''}</small></div>`}).join('') : '<p class="aw-muted">Записів ще немає.</p>';
}

export function renderPurchases(ctx, purchases = []) {
  const e = ctx.escapeHtml;
  return purchases.length ? purchases.map(p => `<article class="aw-purchase"><div class="aw-row-head"><strong>${e(p.product_title || 'Покупка')}</strong><span class="aw-status ${p.status === 'confirmed' ? 'aw-positive' : ''}">${p.status === 'confirmed' ? 'Оплачено' : p.status === 'refunded' ? 'Повернено' : 'Потребує перевірки'}</span></div><p>${e(paymentAmount(p))} · ${e(paymentSource(p.source))}</p><small>${e(adminDate(p.paid_at))}${p.reference ? ` · ${e(p.reference)}` : ''}</small></article>`).join('') : '<p class="aw-muted">Підтверджених покупок немає.</p>';
}

export function renderCustomer(ctx, detail, { draft, pending, notice, error }) {
  const e = ctx.escapeHtml;
  const name = [detail.first_name, detail.last_name].filter(Boolean).join(' ') || `Клієнт ${detail.user_id}`;
  const purchases = detail.purchases || [], grants = detail.manual_grants || [], legacy = detail.legacy_access || [];
  const paid = purchases.filter(p => p.status === 'confirmed');
  const unavailable = detail.unavailable_purchases || [];
  const missingWarning = unavailable.length ? `<div class="aw-warning" role="status"><strong>Оплачений матеріал відсутній</strong><p>${e(unavailable.map(p => p.product_title || p.product_key).join(', '))}</p><p>Запис оплати збережено. Відновіть матеріал у каталозі або зв’яжіться з клієнтом для вирішення питання.</p></div>` : '';
  const blocked = (detail.section_controls || []).filter(s => s.purchased && (!s.has_access || !s.visible));
  const products = [...(detail.grant_products || [])];
  const values = draft?.values || {};
  const missingProduct=detail.manual_grants?.find(p=>p.product_key===values.product_key);
  if(missingProduct&&!products.some(p=>p.product_key===values.product_key))products.push(missingProduct);
  const productOptions = products.map(p => `<option value="${e(p.product_key)}" ${values.product_key === p.product_key ? 'selected' : ''}>${e(p.product_title)}</option>`).join('');
  let form = '';
  if (draft?.phase === 'review') {
    const product = products.find(p => p.product_key === values.product_key)?.product_title || values.product_key || 'Куплений доступ';
    const description = draft.kind === 'payment' ? `Записати підтверджену оплату: ${product}, ${paymentAmount(values)}.` : draft.kind === 'revoke' ? `Скасувати ручний дозвіл «${product}».` : draft.kind === 'restore' ? 'Відновити доступ за підтвердженими покупками.' : `Надати ручний доступ: ${product}.`;
    form = `<section class="aw-review"><h3>Перевірте зміну</h3><p>${e(description)}</p><p class="aw-muted">${e(name)} · ID ${detail.user_id}</p><p>${e(values.reason)}</p>${paid.length ? '<p class="aw-positive">Оплачені покупки залишаються чинними.</p>' : ''}<div class="aw-actions"><button class="aw-button aw-primary" type="button" data-aw-confirm ${pending ? 'disabled' : ''}>${pending ? 'Зберігаємо…' : 'Підтвердити'}</button><button class="aw-button" type="button" data-aw-cancel ${pending ? 'disabled' : ''}>Скасувати</button></div></section>`;
  } else if (draft) {
    form = `<form class="aw-editor" data-aw-form="customer"><h3>${draft.kind === 'payment' ? 'Записати підтверджену оплату' : draft.kind === 'revoke' ? 'Скасувати ручний дозвіл' : draft.kind === 'restore' ? 'Відновити куплений доступ' : 'Надати доступ вручну'}</h3>${draft.kind !== 'restore' ? `<label>Продукт<select class="aw-input" name="product_key" ${draft.kind === 'revoke' ? 'disabled' : ''}>${productOptions}</select></label>` : ''}${draft.kind === 'payment' ? `<div class="aw-fields"><label>Сума<input class="aw-input" name="amount" type="number" min="1" step="1" required value="${e(values.amount || '')}"></label><label>Валюта<select class="aw-input" name="currency"><option value="UAH" ${values.currency === 'UAH' ? 'selected' : ''}>Гривні</option><option value="XTR" ${values.currency === 'XTR' ? 'selected' : ''}>Telegram Stars</option></select></label></div><label>Номер або опис переказу<input class="aw-input" name="reference" maxlength="200" value="${e(values.reference || '')}"></label>` : ''}<label>${draft.kind === 'payment' ? 'Як перевірено оплату' : 'Причина зміни'}<textarea class="aw-input" name="reason" rows="2" minlength="3" maxlength="500" required>${e(values.reason || '')}</textarea></label><div class="aw-actions"><button class="aw-button aw-primary" type="submit">Переглянути зміну</button><button class="aw-button" type="button" data-aw-cancel>Скасувати</button></div></form>`;
  }
  return `<div class="aw-customer" data-aw-customer="${detail.user_id}"><button class="aw-text-button aw-mobile-only" type="button" data-aw-close>До списку клієнтів</button><header class="aw-customer-head"><h2 tabindex="-1" data-aw-focus="customer-title">${e(name)}</h2><p class="aw-muted">Telegram ID ${detail.user_id}</p></header><section class="aw-section"><h3>Покупки</h3>${renderPurchases(ctx, purchases)}${missingWarning}${blocked.length ? `<div class="aw-warning"><strong>Оплата є, доступ потребує уваги</strong><p>${e(blocked.map(s => s.title).join(', '))}</p><button class="aw-button" type="button" data-aw-action="restore" ${pending ? 'disabled' : ''}>Відновити куплений доступ</button></div>` : paid.length && !unavailable.length ? '<p class="aw-positive">Куплений доступ захищений.</p>' : ''}</section><section class="aw-section"><h3>Ручні дозволи</h3>${grants.length ? grants.map(g => `<div class="aw-grant"><div><strong>${e(g.product_title || g.product_key)}</strong><small>${e(g.reason || '')}</small></div><button class="aw-text-button aw-danger" type="button" data-aw-revoke="${e(g.product_key)}" ${pending ? 'disabled' : ''}>Скасувати дозвіл</button></div>`).join('') : '<p class="aw-muted">Ручних дозволів немає.</p>'}<div class="aw-actions"><button class="aw-button" type="button" data-aw-action="grant" ${pending || !products.length ? 'disabled' : ''}>Надати доступ</button><button class="aw-text-button" type="button" data-aw-action="record-payment" ${pending || !products.length ? 'disabled' : ''}>Записати оплату</button></div>${form}${error ? `<p class="aw-error" role="alert">${e(error)}</p>` : ''}${notice ? `<p class="aw-positive" role="status">${e(notice)}</p>` : ''}</section>${legacy.length ? `<section class="aw-section aw-warning"><h3>Старий доступ: походження не перевірене</h3>${legacy.map(p => `<p>${e(p.product_title || p.product_key)}</p>`).join('')}<p class="aw-muted">Історії оплати в старій системі немає. Звірте Telegram або запишіть підтверджений переказ.</p></section>` : ''}<details class="aw-section"><summary>Доступні розділи</summary>${(detail.section_controls || []).map(s => `<div class="aw-access-fact"><span>${e(s.title)}</span><span>${s.has_access && s.visible ? 'Відкрито' : s.has_access ? 'Приховано' : 'Не надано'}<small>${e(({ paid:'За покупкою',manual:'Ручний дозвіл',legacy:'Старий доступ',free:'Безкоштовно',none:'Без доступу' })[s.access_source] || 'Потребує перевірки')}</small></span></div>`).join('')}</details><details class="aw-section"><summary>Історія покупок і змін</summary>${renderEvents(ctx, detail.access_events || [], products)}</details><details class="aw-section"><summary>Навчальні результати та профіль</summary><p>Тестів: ${e(detail.stats?.count || 0)} · середній результат: ${e(detail.stats?.avg || 0)}%</p><p class="aw-muted">Реєстрація: ${e(adminDate(detail.created_at))}</p>${!detail.is_admin && detail.user_id !== ctx.state.bootstrap.user.id ? `<button class="aw-text-button aw-danger" type="button" data-aw-delete ${pending ? 'disabled' : ''}>Видалити профіль</button>` : ''}</details></div>`;
}
