export const ADMIN_DESTINATIONS = [
  { title: 'Клієнти', iconName: 'users', screen: 'admin-users', key: 'clients' },
  { title: 'Матеріали', iconName: 'folder', screen: 'admin-attestation-banks', key: 'content' },
  { title: 'Система', iconName: 'settings', screen: 'admin-settings', key: 'system' },
];

/** Telegram owns BackButton; browsers receive one explicit fallback. */
export function mountAdminNavigation(ctx) {
  const content = ctx.refs.mainPanel.querySelector('.screen-content');
  if (!content || content.closest('.aw-shell')) return;
  const screen = ctx.state.currentScreen;
  const active = ['admin','admin-users','admin-user-detail','admin-user-access','admin-payments'].includes(screen) ? 'clients' : ['admin-settings','admin-messages','admin-activity'].includes(screen) ? 'system' : 'content';
  const shell = document.createElement('div');
  shell.className = 'aw-shell';
  const nav = document.createElement('nav');
  nav.className = 'aw-navigation';
  nav.setAttribute('aria-label', 'Адміністративна навігація');
  nav.innerHTML = `
    <div class="aw-brand"><strong>Test_Customs</strong><span>Адмінпанель</span></div>
    <div class="aw-destinations">${ADMIN_DESTINATIONS.map(item => `<button type="button" data-aw-destination="${item.key}" aria-current="${active === item.key ? 'page' : 'false'}">${ctx.lineIcon(item.iconName)}<span>${item.title}</span></button>`).join('')}</div>
    <button class="aw-exit" type="button" data-aw-exit>${ctx.lineIcon('home')}<span>До застосунку</span></button>
  `;
  nav.querySelectorAll('[data-aw-destination]').forEach(button => button.addEventListener('click', async () => {
    if (ctx.state.currentScreen === 'admin-apk-import') await ctx.goBack();
    if(button.dataset.awDestination==='clients'&&window.innerWidth<=960&&ctx.state.adminWorkspace){
      const workspace=ctx.state.adminWorkspace,id=workspace.selected;workspace.selected=null;
      if(id)workspace.focusSelector=`[data-aw-user="${id}"]`;
    }
    ctx.state.screenHistory = ['home'];
    ctx.navigate(ADMIN_DESTINATIONS.find(item => item.key === button.dataset.awDestination).screen, { replace: true });
  }));
  nav.querySelector('[data-aw-exit]').addEventListener('click', async () => {
    if (ctx.state.currentScreen === 'admin-apk-import') await ctx.goBack();
    ctx.state.screenHistory = [];
    ctx.navigate('home', { replace: true });
  });
  const main = document.createElement('div'); main.className = 'aw-main';
  content.replaceWith(shell); main.append(content); shell.append(nav,main);
  document.body.classList.add('admin-mode');
}
