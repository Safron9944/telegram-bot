import test from 'node:test';
import assert from 'node:assert/strict';
import { adminAccessSummary } from '../static/js/admin_access.js';
import { ADMIN_DESTINATIONS } from '../static/js/admin_navigation.js';

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

test('all five administrative tasks have distinct reachable destinations', () => {
  assert.deepEqual(ADMIN_DESTINATIONS.map(item => item.title), ['Користувачі', 'Доступ і тарифи', 'Матеріали', 'Повідомлення', 'Налаштування']);
  assert.equal(new Set(ADMIN_DESTINATIONS.map(item => item.screen)).size, 5);
  assert.ok(ADMIN_DESTINATIONS.every(item => item.screen.startsWith('admin-')));
});
