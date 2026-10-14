/** Describe rights separately from package status and menu visibility. */
export function adminAccessSummary(payload = {}) {
  const sections = Array.isArray(payload.section_controls) ? payload.section_controls : [];
  const access = payload.access || {};
  const granted = sections.filter(section => section.has_access);
  const available = granted.filter(section => section.visible).length;
  const hiddenGranted = granted.filter(section => !section.visible).length;
  const shownLocked = sections.filter(section => section.visible && !section.has_access).length;
  const full = access.tier === 'full';
  const label = payload.is_admin ? 'Адміністратор'
    : full ? 'Повний пакет'
    : granted.length ? 'Окремий доступ' : 'Немає платного доступу';
  return {
    available,
    total: sections.length,
    hiddenGranted,
    shownLocked,
    label,
    tone: payload.is_admin ? 'admin' : full ? 'full' : granted.length ? 'partial' : 'none',
    packageLabel: full
      ? `Повний пакет${access.sub_infinite ? ' · безстроково' : ''}`
      : access.tier === 'cases' ? 'Пакет атестації' : 'Повний пакет не активний',
  };
}
