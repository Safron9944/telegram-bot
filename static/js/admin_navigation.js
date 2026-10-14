export const ADMIN_DESTINATIONS = [
  { title: 'Користувачі', subtitle: 'Знайти людину й керувати її доступом', iconName: 'users', screen: 'admin-users' },
  { title: 'Доступ і тарифи', subtitle: 'Повний пакет і ціни окремих розділів', iconName: 'scale', screen: 'admin-tariffs' },
  { title: 'Матеріали', subtitle: 'Розділи, питання, кейси та імпорт', iconName: 'folder', screen: 'admin-attestation-banks' },
  { title: 'Повідомлення', subtitle: 'Текст, одержувачі й результат надсилання', iconName: 'message', screen: 'admin-messages' },
  { title: 'Налаштування', subtitle: 'Контакт підтримки й налаштування показу', iconName: 'settings', screen: 'admin-settings' },
];

/** Telegram owns BackButton; browsers receive one explicit fallback. */
export function mountAdminNavigation(ctx) {
  const content = ctx.refs.mainPanel.querySelector('.screen-content');
  if (!content || content.querySelector('.admin-navigation')) return;
  const root = ctx.state.currentScreen === 'admin';
  const nativeBack = Boolean(window.Telegram?.WebApp?.initData && window.Telegram.WebApp.BackButton);
  const nav = document.createElement('nav');
  nav.className = 'admin-navigation';
  nav.setAttribute('aria-label', 'Адміністративна навігація');
  nav.innerHTML = `
    ${nativeBack ? '<span class="admin-navigation__context">Адміністрування</span>' : '<button type="button" data-admin-back>← Назад</button>'}
    ${root ? '<span class="admin-navigation__context">Test_Customs</span>' : '<button type="button" data-admin-home>До меню адміністратора</button>'}
  `;
  nav.querySelector('[data-admin-back]')?.addEventListener('click', () => void ctx.goBack());
  nav.querySelector('[data-admin-home]')?.addEventListener('click', async () => {
    // Leaving an APK import must release its temporary session, as Back does.
    if (ctx.state.currentScreen === 'admin-apk-import') await ctx.goBack();
    ctx.state.screenHistory = ['home'];
    ctx.navigate('admin', { replace: true });
  });
  content.prepend(nav);
}
