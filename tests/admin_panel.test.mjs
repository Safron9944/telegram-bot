import test from 'node:test';
import assert from 'node:assert/strict';
import { adminAccessSummary } from '../static/js/admin_access.js';
import { ADMIN_DESTINATIONS } from '../static/js/admin_navigation.js';
import { paymentAmount, paymentSource, eventLabel, renderEvents, renderCustomer } from '../static/js/admin_customer_view.js';

const section = (has_access, visible, control_mode = 'access') => ({ has_access, visible, control_mode });

test('a visible unpaid material does not count as available', () => {
  const summary = adminAccessSummary({ access: { tier: 'none' }, section_controls: [section(true, true), section(false, true, 'visibility')] });
  assert.equal(summary.available, 1);
  assert.equal(summary.shownLocked, 1);
  assert.equal(summary.label, 'Окремий доступ');
});

test('a paid but hidden material has rights without being available in the menu', () => {
  const summary = adminAccessSummary({ access: { tier: 'none' }, section_controls: [section(true, false, 'visibility')] });
  assert.equal(summary.available, 0);
  assert.equal(summary.hiddenGranted, 1);
  assert.equal(summary.label, 'Окремий доступ');
});

test('full package does not make an explicitly blocked section available', () => {
  const summary = adminAccessSummary({ access: { tier: 'full', sub_infinite: true }, section_controls: [section(false, true), section(true, true)] });
  assert.equal(summary.available, 1);
  assert.equal(summary.label, 'Повний пакет');
  assert.equal(summary.packageLabel, 'Повний пакет · безстроково');
});

test('only visibility without paid rights does not claim individual access', () => {
  const summary = adminAccessSummary({ access: { tier: 'none' }, section_controls: [section(false, true, 'visibility')] });
  assert.equal(summary.available, 0);
  assert.equal(summary.label, 'Немає платного доступу');
});

test('administrator status takes priority over package status', () => {
  assert.equal(adminAccessSummary({ is_admin: true, access: { tier: 'none' } }).label, 'Адміністратор');
});

test('missing or malformed section controls are safe', () => {
  assert.equal(adminAccessSummary().available, 0);
  assert.equal(adminAccessSummary({ section_controls: {} }).total, 0);
});

test('legacy attestation package is distinguished from a full package', () => {
  const summary = adminAccessSummary({ access: { tier: 'cases' }, section_controls: [section(true, true)] });
  assert.equal(summary.packageLabel, 'Пакет атестації');
  assert.equal(summary.label, 'Окремий доступ');
});

test('customer, content and system workspaces have distinct destinations', () => {
  assert.deepEqual(ADMIN_DESTINATIONS.map(item => item.title), ['Клієнти', 'Контент', 'Система']);
  assert.equal(new Set(ADMIN_DESTINATIONS.map(item => item.screen)).size, 3);
  assert.ok(ADMIN_DESTINATIONS.every(item => item.screen.startsWith('admin-')));
});

test('unknown historic payment amount is never represented as zero', () => {
  assert.equal(paymentAmount({amount:null,currency:'XTR'}), 'Сума невідома');
});

test('currency minor units are displayed correctly', () => {
  assert.equal(paymentAmount({amount:50000,currency:'UAH'}), '500 ₴');
  assert.equal(paymentAmount({amount:125,currency:'UAH'}), '1,25 ₴');
  assert.equal(paymentAmount({amount:250,currency:'XTR'}), '250 ⭐');
});

test('admin confirmed external payment is distinguished from Telegram', () => {
  assert.equal(paymentSource('admin_external'), 'Підтверджено адміністратором');
  assert.match(paymentSource('telegram_sync'), /Telegram Stars/);
  assert.equal(paymentSource('unknown'), 'Джерело потребує перевірки');
});

const escapeHtml=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
test('real backend audit actions include a safe product snapshot', () => {
  assert.equal(eventLabel({action:'grant_removed'}),'Скасовано ручний дозвіл');
  assert.equal(eventLabel({action:'payment_refunded'}),'Платіж повернено');
  const html=renderEvents({escapeHtml},[{action:'grant_added',product_title:'<Атестація>',product_key:'section:attestation:7',reason:'Причина',actor_id:1}]);
  assert.match(html,/Надано ручний дозвіл · &lt;Атестація&gt;/);
  assert.doesNotMatch(html,/<Атестація>/);
});
test('missing paid material is shown honestly and an orphan gift remains revocable', () => {
  const html=renderCustomer({escapeHtml,state:{bootstrap:{user:{id:1}}}},{user_id:42,purchases:[{status:'confirmed',product_title:'Старий банк',amount:100,currency:'XTR'}],unavailable_purchases:[{product_title:'Старий банк'}],manual_grants:[{product_key:'section:attestation:9',product_title:'Видалений розділ'}],grant_products:[]},{draft:{kind:'revoke',phase:'edit',values:{product_key:'section:attestation:9'}},pending:true});
  assert.match(html,/Оплачений матеріал відсутній/);
  assert.doesNotMatch(html,/Куплений доступ захищений/);
  assert.match(html,/value="section:attestation:9" selected/);
  assert.match(html,/data-aw-delete disabled/);
});
