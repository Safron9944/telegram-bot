import { mountAdminNavigation } from './admin_navigation.js?v=20261014-admin-compact-01';
import { adminDate, paymentAmount, paymentSource, renderCustomer, renderEvents, clientListSummary } from './admin_customer_view.js?v=20261014-admin-compact-01';

const CLIENT_SCREENS = new Set(['admin','admin-users','admin-user-detail','admin-user-access','admin-payments']);
function model(ctx) {
  return ctx.state.adminWorkspace ||= { selected:null, view:'users', users:null, payments:null, loading:false, listError:'', query:'', filter:'all', offset:0, detail:new Map(), versions:new Map(), pending:new Set(), drafts:new Map(), errors:new Map(), notices:new Map(), sync:{busy:false,offset:0,imported:0,skipped:0,message:''} };
}
const alive = ctx => CLIENT_SCREENS.has(ctx.state.currentScreen);
const listKey = (m,view) => JSON.stringify([m.query,view==='users'?m.filter:null,m.offset]);
const uuid = () => globalThis.crypto?.randomUUID?.() || 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.random()*16|0;return(c==='x'?r:(r&3|8)).toString(16)});

export function renderAdminWorkspace(ctx) {
  ctx.setChrome({showBack:true}); const m=model(ctx);
  m.view=ctx.state.currentScreen==='admin-payments'?'payments':'users';
  if(['admin-user-detail','admin-user-access'].includes(ctx.state.currentScreen)&&ctx.state.selectedAdminUserId)m.selected=ctx.state.selectedAdminUserId;
  paint(ctx);
  if((!(m.view==='users'?m.users:m.payments)||m[`${m.view}Key`]!==listKey(m,m.view))&&(!m.loading||m.loadingView!==m.view))void loadList(ctx);
  if(m.selected&&!m.detail.has(m.selected)&&!m.pending.has(m.selected))void loadCustomer(ctx,m.selected);
}

function listRows(ctx, m, data) {
  const e=ctx.escapeHtml;
  if(!data?.items?.length)return '<div class="aw-empty"><h2>Нічого не знайдено</h2><p>Змініть пошук або фільтр.</p></div>';
  if(m.view==='payments')return data.items.map(p=>`<button class="aw-payment-row" type="button" data-aw-user="${Number(p.user_id)}"><span><strong>${e(p.product_title)}</strong><small>${e(p.display_name || `ID ${p.user_id}`)} · ${e(adminDate(p.paid_at))}</small><small>${e(paymentSource(p.source))} · ${p.status==='confirmed'?'Оплачено':p.status==='refunded'?'Повернено':'Потребує перевірки'}</small></span><strong>${e(paymentAmount(p))}</strong></button>`).join('');
  return data.items.map(user=>{const summary=clientListSummary(user),name=user.display_name||[user.first_name,user.last_name].filter(Boolean).join(' ')||'Без імені';return `<button type="button" class="aw-client-row" data-aw-user="${user.user_id}" aria-pressed="${m.selected===user.user_id}" aria-label="${e(`Відкрити клієнта ${name}, ${summary.product}`)}"><span class="aw-client-description"><strong>${e(name)}</strong><span class="aw-client-product">${e(summary.product)}</span><small class="${summary.attention?'aw-attention':''}">${e([summary.evidence,`ID ${user.user_id}`].filter(Boolean).join(' · '))}</small>${summary.manualProducts?.length?`<small class="aw-client-manual">Вручну: ${e(summary.manualProducts.join(', '))}</small>`:''}</span><span class="aw-client-amount">${e(summary.amount)}</span><svg class="aw-client-chevron" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m9 5 7 7-7 7"/></svg></button>`}).join('');
}

function paint(ctx) {
  if(!alive(ctx))return;
  const m=model(ctx),e=ctx.escapeHtml,data=m.view==='users'?m.users:m.payments,detail=m.detail.get(m.selected);
  const active=document.activeElement,focusName=active?.closest('[data-aw-search]')||active?.closest('[data-aw-customer]')?.dataset.awCustomer===String(m.selected)?active?.name:null;
  const focusKey=active?.dataset.awFocus,focusUser=active?.dataset.awUser;
  const scroll=window.scrollY;
  const inspector=detail?renderCustomer(ctx,detail,{draft:m.drafts.get(m.selected),pending:m.pending.has(m.selected),notice:m.notices.get(m.selected),error:m.errors.get(m.selected)}):m.selected?`<button class="aw-text-button aw-mobile-only" type="button" data-aw-close>До списку клієнтів</button><h2 tabindex="-1" data-aw-focus="customer-title">Картка клієнта</h2><p class="aw-muted" role="status">${m.errors.get(m.selected)?e(m.errors.get(m.selected)):'Завантажуємо клієнта…'}</p>${m.errors.has(m.selected)?'<button class="aw-button" type="button" data-aw-retry-customer>Спробувати ще раз</button>':''}`:'<div class="aw-empty"><h2>Покупки й доступ клієнта</h2><p>Оберіть людину зі списку. Тут будуть її покупки, ручні дозволи й історія змін.</p></div>';
  ctx.refs.mainPanel.innerHTML=`<section class="screen-content aw-workspace"><header class="aw-page-heading"><div><h1>Клієнти</h1></div><div class="aw-view-switch" aria-label="Перегляд клієнтів"><button type="button" data-aw-view="users" aria-pressed="${m.view==='users'}">Картотека</button><button type="button" data-aw-view="payments" aria-pressed="${m.view==='payments'}">Оплати</button></div></header><div class="aw-client-layout ${m.selected?'aw-has-selection':''}"><div class="aw-directory"><form class="aw-search" data-aw-search><label><span class="aw-sr-only">Ім'я або Telegram ID</span><input class="aw-input" name="query" type="search" placeholder="Ім’я або ID" value="${e(m.searchDraft ?? m.query)}" maxlength="100"></label><button class="aw-button aw-search-button" type="submit" aria-label="Знайти">${ctx.lineIcon('search')}<span>Знайти</span></button></form>${m.view==='users'?`<div class="aw-mobile-filter"><label for="aw-client-filter">Показати</label><select id="aw-client-filter" class="aw-input" data-aw-filter-select data-aw-focus="client-filter">${[['all','Усі клієнти'],['paid','Є оплати'],['manual','Доступ вручну'],['attention','Потребують перевірки']].map(([key,label])=>`<option value="${key}" ${m.filter===key?'selected':''}>${label}</option>`).join('')}</select></div><div class="aw-filters" aria-label="Стан клієнтів">${[['all','Усі'],['paid','З покупками'],['manual','Ручний доступ'],['attention','Потребують уваги']].map(([key,label])=>`<button type="button" data-aw-filter="${key}" aria-pressed="${m.filter===key}">${label}</button>`).join('')}</div>`:`<div class="aw-sync"><button class="aw-button" type="button" data-aw-sync ${m.sync.busy?'disabled':''}>${m.sync.busy?'Звіряємо Telegram…':m.sync.offset?'Продовжити звірку Telegram':'Звірити оплати Telegram'}</button><p class="aw-muted" role="status">${e(m.sync.message || 'Порівняти журнал із доступною історією Telegram Stars.')}</p></div>`}<div class="aw-list" aria-live="polite">${m.loading?'<div class="aw-loading">Завантажуємо записи…</div>':m.listError?`<p class="aw-error" role="alert">${e(m.listError)}</p><button class="aw-button" type="button" data-aw-retry-list>Спробувати ще раз</button>`:listRows(ctx,m,data)}</div><div class="aw-pagination">${data?.has_prev?'<button class="aw-button" type="button" data-aw-page="prev">Попередні</button>':''}${data?.has_next?'<button class="aw-button" type="button" data-aw-page="next">Наступні</button>':''}</div></div><aside class="aw-inspector" aria-label="Картка клієнта">${inspector}</aside></div></section>`;
  mountAdminNavigation(ctx); bind(ctx);
  if(focusName){const input=[...ctx.refs.mainPanel.querySelectorAll('[name]')].find(n=>n.name===focusName);input?.focus({preventScroll:true});}
  const selector=m.focusSelector||(focusKey?`[data-aw-focus="${focusKey}"]`:focusUser?`[data-aw-user="${focusUser}"]`:null);
  if(selector){const target=ctx.refs.mainPanel.querySelector(selector);if(target){target.focus({preventScroll:true});m.focusSelector=null;}}
  window.scrollTo({top:scroll,behavior:'instant'});
}

async function loadList(ctx) {
  const m=model(ctx),view=m.view,key=listKey(m,view),version=(m.listVersion||0)+1;m.listVersion=version;m.loading=true;m.loadingView=view;m.listError='';paint(ctx);
  try{const query=new URLSearchParams({offset:m.offset,limit:20,q:m.query});if(view==='users')query.set('status',m.filter);const data=await ctx.api(`/api/admin/${view==='users'?'users':'payments'}?${query}`);if(m.listVersion!==version)return;if(view==='users')m.users=data;else m.payments=data;m[`${view}Key`]=key;}
  catch(error){if(m.listVersion===version)m.listError=error.message;}
  finally{if(m.listVersion===version){m.loading=false;paint(ctx);}}
}

async function loadCustomer(ctx,id) {
  const m=model(ctx);if(m.pending.has(id))return;
  const version=(m.versions.get(id)||0)+1;m.versions.set(id,version);m.errors.delete(id);
  try{const detail=await ctx.api(`/api/admin/users/${id}`);if(m.versions.get(id)!==version||m.pending.has(id))return;m.detail.set(id,detail);}
  catch(error){if(m.versions.get(id)===version)m.errors.set(id,error.message);}
  finally{if(m.selected===id)paint(ctx);}
}

async function confirm(ctx) {
  const m=model(ctx),id=m.selected,draft=m.drafts.get(id);if(!draft||draft.phase!=='review'||m.pending.has(id))return;
  const values={...draft.values};m.pending.add(id);m.versions.set(id,(m.versions.get(id)||0)+1);m.errors.delete(id);m.notices.delete(id);m.focusSelector='[data-aw-focus="customer-title"]';paint(ctx);
  const route=draft.kind==='payment'?'payments':draft.kind==='restore'?'restore-purchases':'grants';
  const body=draft.kind==='payment'?{...values,request_id:draft.requestId}:draft.kind==='restore'?{reason:values.reason}:{product_key:values.product_key,enabled:draft.kind==='grant',reason:values.reason};
  try{const detail=await ctx.api(`/api/admin/users/${id}/${route}`,{method:'POST',body});m.detail.set(id,detail);m.drafts.delete(id);m.notices.set(id,draft.kind==='payment'?'Оплату записано.':draft.kind==='revoke'?'Ручний дозвіл скасовано.':draft.kind==='restore'?'Куплений доступ відновлено.':'Ручний доступ надано.');m.users=null;m.payments=null;}
  catch(error){m.errors.set(id,error.message);}
  finally{m.pending.delete(id);paint(ctx);if(alive(ctx)&&!m.loading)void loadList(ctx);}
}

async function syncPayments(ctx) {
  const m=model(ctx),sync=m.sync;if(sync.busy)return;if(m.pending.size)return ctx.setMessage('error','Завершіть поточну зміну доступу перед звіркою.');sync.busy=true;paint(ctx);
  try{let pages=0,more=true;while(more&&pages++<20&&alive(ctx)&&m.view==='payments'){const data=await ctx.api('/api/admin/payments/sync',{method:'POST',body:{offset:sync.offset,limit:100}});sync.imported+=Number(data.imported||0);sync.skipped+=Number(data.skipped||0);more=!!data.has_more;if(more&&!(Number(data.next_offset)>sync.offset))throw new Error('Telegram не повернув наступну сторінку. Спробуйте звірку знову.');sync.offset=more?Number(data.next_offset):0;sync.message=`${more?'Звіряємо історію':'Звірку завершено'} · додано ${sync.imported} · пропущено ${sync.skipped}`;paint(ctx);}if(sync.offset)sync.message+= ' · продовжіть звірку для решти записів.';}
  catch(error){sync.message=`${error.message} · додано ${sync.imported} · пропущено ${sync.skipped} · ${sync.offset?'продовжіть':'повторіть'} звірку.`;}
  finally{sync.busy=false;m.payments=null;m.users=null;m.detail.clear();for(const[id,version]of m.versions)m.versions.set(id,version+1);paint(ctx);if(alive(ctx)){void loadList(ctx);if(m.selected&&!m.pending.has(m.selected))void loadCustomer(ctx,m.selected);}}
}

function bind(ctx) {
  const root=ctx.refs.mainPanel,m=model(ctx);
  root.querySelectorAll('[data-aw-user]').forEach(b=>b.addEventListener('click',()=>{m.selected=Number(b.dataset.awUser);ctx.state.selectedAdminUserId=m.selected;m.focusSelector='[data-aw-focus="customer-title"]';paint(ctx);if(!m.detail.has(m.selected))void loadCustomer(ctx,m.selected);if(window.innerWidth<960)window.scrollTo({top:0,behavior:'instant'});}));
  root.querySelectorAll('[data-aw-view]').forEach(b=>b.addEventListener('click',()=>{m.offset=0;m.selected=null;ctx.navigate(b.dataset.awView==='payments'?'admin-payments':'admin-users',{replace:true});}));
  root.querySelectorAll('[data-aw-filter]').forEach(b=>b.addEventListener('click',()=>{m.filter=b.dataset.awFilter;m.offset=0;void loadList(ctx);}));
  root.querySelector('[data-aw-filter-select]')?.addEventListener('change',event=>{m.filter=event.target.value;m.offset=0;void loadList(ctx);});
  root.querySelector('[data-aw-search]')?.addEventListener('submit',event=>{event.preventDefault();m.query=new FormData(event.target).get('query').trim();m.searchDraft=m.query;m.offset=0;void loadList(ctx);});
  root.querySelector('[data-aw-search] input')?.addEventListener('input',event=>{m.searchDraft=event.target.value;});
  root.querySelectorAll('[data-aw-page]').forEach(b=>b.addEventListener('click',()=>{m.offset=Math.max(0,m.offset+(b.dataset.awPage==='next'?20:-20));void loadList(ctx);}));
  root.querySelector('[data-aw-close]')?.addEventListener('click',()=>{m.focusSelector=`[data-aw-user="${m.selected}"]`;m.selected=null;paint(ctx);});
  root.querySelector('[data-aw-retry-list]')?.addEventListener('click',()=>void loadList(ctx));
  root.querySelector('[data-aw-retry-customer]')?.addEventListener('click',()=>void loadCustomer(ctx,m.selected));
  root.querySelector('[data-aw-sync]')?.addEventListener('click',()=>void syncPayments(ctx));
  const start=(kind,key)=>{const id=m.selected;if(m.pending.has(id))return;const productKey=key||m.detail.get(id)?.grant_products?.[0]?.product_key||'full';m.drafts.set(id,{kind,phase:'edit',values:{product_key:productKey,currency:'UAH',reason:''},requestId:uuid()});m.errors.delete(id);m.notices.delete(id);paint(ctx);root.querySelector('[name="reason"]')?.focus({preventScroll:true});};
  root.querySelectorAll('[data-aw-action]').forEach(b=>b.addEventListener('click',()=>start(b.dataset.awAction==='record-payment'?'payment':b.dataset.awAction)));
  root.querySelectorAll('[data-aw-revoke]').forEach(b=>b.addEventListener('click',()=>start('revoke',b.dataset.awRevoke)));
  root.querySelector('[data-aw-cancel]')?.addEventListener('click',()=>{if(m.pending.has(m.selected))return;m.drafts.delete(m.selected);m.errors.delete(m.selected);paint(ctx);});
  root.querySelector('[data-aw-confirm]')?.addEventListener('click',()=>void confirm(ctx));
  const form=root.querySelector('[data-aw-form="customer"]');
  const updateAmount=()=>{const amount=form?.querySelector('[name="amount"]');if(amount){const stars=form.querySelector('[name="currency"]').value==='XTR';amount.min=stars?'1':'0.01';amount.step=stars?'1':'0.01';}};updateAmount();
  form?.addEventListener('input',event=>{const draft=m.drafts.get(m.selected);if(draft&&event.target.name)draft.values[event.target.name]=event.target.value;});
  form?.addEventListener('change',event=>{const draft=m.drafts.get(m.selected);if(draft&&event.target.name)draft.values[event.target.name]=event.target.value;updateAmount();});
  form?.addEventListener('submit',event=>{event.preventDefault();const draft=m.drafts.get(m.selected);if(!draft)return;const values={...draft.values,...Object.fromEntries(new FormData(form))};values.reason=String(values.reason||'').trim();if(values.reason.length<3)return ctx.setMessage('error','Вкажіть причину зміни.');if(draft.kind==='payment'){const amount=Number(values.amount);const units=values.currency==='UAH'?Math.round(amount*100):amount;if(!Number.isFinite(amount)||amount<=0||!Number.isInteger(units)||units>1e9)return ctx.setMessage('error','Перевірте суму оплати.');values.amount=units;}draft.values=values;draft.phase='review';m.focusSelector='[data-aw-confirm]';paint(ctx);});
  root.querySelector('[data-aw-delete]')?.addEventListener('click',async()=>{const id=m.selected,detail=m.detail.get(id);if(m.pending.has(id))return;if((detail.purchases||[]).length||(detail.legacy_access||[]).length)return ctx.setMessage('error','Профіль із покупками або старим неперевіреним доступом захищено від видалення.');if(!window.confirm(`Видалити профіль ID ${id} та його навчальні результати?`))return;m.pending.add(id);m.versions.set(id,(m.versions.get(id)||0)+1);m.notices.set(id,'Видаляємо профіль…');paint(ctx);try{await ctx.api(`/api/admin/users/${id}`,{method:'DELETE'});m.detail.delete(id);m.drafts.delete(id);if(m.selected===id)m.selected=null;}catch(error){m.errors.set(id,error.message);m.notices.delete(id);}finally{m.pending.delete(id);paint(ctx);if(alive(ctx))void loadList(ctx);}});
}

export function renderAdminActivity(ctx) {
  ctx.setChrome({showBack:true});const offset=ctx.state.adminActivityOffset||0;
  ctx.refs.mainPanel.innerHTML='<section class="screen-content aw-workspace"><header class="aw-page-heading"><div><h1>Журнал адміністративних дій</h1><p>Хто, коли й чому змінював доступ.</p></div></header><div class="aw-activity" aria-live="polite">Завантажуємо події…</div></section>';
  const node=ctx.refs.mainPanel.querySelector('.aw-activity');
  const load=async()=>{try{const data=await ctx.api(`/api/admin/activity?offset=${offset}&limit=30`);if(!node.isConnected)return;node.innerHTML=renderEvents(ctx,data.items||[])+`<div class="aw-pagination">${data.has_prev?'<button class="aw-button" data-page="prev" type="button">Попередні</button>':''}${data.has_next?'<button class="aw-button" data-page="next" type="button">Наступні</button>':''}</div>`;node.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>{ctx.state.adminActivityOffset=Math.max(0,offset+(b.dataset.page==='next'?30:-30));ctx.render();}));}catch(error){if(node.isConnected){node.innerHTML=`<p class="aw-error">${ctx.escapeHtml(error.message)}</p><button class="aw-button" type="button">Спробувати ще раз</button>`;node.querySelector('button').addEventListener('click',load);}}};void load();
}
