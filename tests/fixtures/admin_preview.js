// Synthetic data only. All API requests are intercepted before the real app boots.
const sections = [
  {key:'attestation:7', title:'Атестація посадових осіб митних органів — підготовка до тестування', kind:'attestation', bank_id:7, bank_slug:'test-bank', group:'primary', visible:true, has_access:true, price:100, questions_count:800, icon:'document', deletable:true},
  {key:'customs', title:'Митні компетенції', group:'primary', visible:true, has_access:true, price:250, content_screen:'admin-questions', icon:'graduation'},
  {key:'cases', title:'Кейси', group:'materials', visible:true, has_access:true, price:100, content_screen:'admin-cases', icon:'folder'},
  {key:'test_questions', title:'Тестові питання', group:'materials', visible:true, has_access:true, price:250, content_screen:'admin-test-questions', icon:'clipboard'},
  {key:'question_search', title:'Пошук питань', group:'materials', visible:true, has_access:true, price:250, icon:'search'},
  {key:'support', title:'Підтримка', group:'help', visible:true, has_access:true, price:0, icon:'support'},
];
const detail = {
  user_id:42, first_name:'Тестова', last_name:'Користувачка', is_admin:false, created_at:'2026-01-21T12:58:00Z',
  access:{tier:'none', sub_infinite:false, sub_end:null}, stats:{count:4, avg:95, last:{correct:113,total:130,finished_at:'2026-10-12T13:01:00Z'}},
  section_controls:[
    {key:'attestation:7',title:sections[0].title,group:'primary',control_mode:'access',has_access:false,visible:true},
    {key:'customs',title:'Митні компетенції',group:'primary',control_mode:'access',has_access:true,visible:true,admin_decision:true},
    ...sections.slice(2,5).map(section=>({key:section.key,title:section.title,group:'materials',control_mode:'visibility',has_access:false,visible:section.key==='cases'})),
  ],
};
const settings = {price_full:250,admin_contact_url:'https://t.me/test_support'};
const clone = value => JSON.parse(JSON.stringify(value));
const nativeFetch = window.fetch.bind(window);
window.adminPreview = {requests:[],failNext:false,delayMs:25,detail,settings,sections};
window.fetch = async (path, options={}) => {
  if (!String(path).startsWith('/api/')) return nativeFetch(path,options);
  const url = new URL(path,window.location.origin);
  const body = options.body ? JSON.parse(options.body) : {};
  const preview = window.adminPreview;
  preview.requests.push({path:url.pathname,query:url.search,method:options.method||'GET',body});
  // Snapshot GETs at request time, so delayed responses really can be stale.
  const snapshot = clone(detail);
  if (preview.holdPost && options.method==='POST') {
    await new Promise(resolve=>{preview.releasePost=resolve;});
    preview.releasePost=null;
  }
  await new Promise(resolve=>setTimeout(resolve,url.pathname==='/api/admin/users/42' ? (preview.getDetailDelayMs || preview.delayMs) : preview.delayMs));
  if (preview.failNext && options.method==='POST') {
    preview.failNext=false;
    return new Response(JSON.stringify({detail:{message:'Тестова помилка збереження'}}),{status:503});
  }
  let data;
  if (url.pathname==='/api/bootstrap') data={user:{id:1,is_admin:true,first_name:'Адміністратор'},catalog:{attestation_banks:[],ok_modules:[]},sections:[],saved_view:null,payment_prices:{full:settings.price_full},links:{admin_url:settings.admin_contact_url},stats:{count:0,avg:0,last:null}};
  else if (url.pathname==='/api/admin/settings') {
    if (options.method==='POST') {
      Object.assign(settings,body);
      if (settings.admin_contact_url.startsWith('@')) settings.admin_contact_url=`https://t.me/${settings.admin_contact_url.slice(1)}`;
    }
    data=clone(settings);
  }
  else if (url.pathname==='/api/admin/sections') data={items:clone(sections)};
  else if (url.pathname==='/api/admin/global-search') data={ok:[{id:1,question:'Тестове питання для перевірки навігації',topic:'Тестовий розділ'}],attestation:[],cases:[],test:[]};
  else if (url.pathname==='/api/admin/questions/1') data={question:{id:1,question:'Тестове питання для перевірки навігації',choices:[{index:1,text:'Тестова відповідь',is_correct:true}]}};
  else if (url.pathname.startsWith('/api/admin/sections/')) {
    const section=sections.find(item=>item.key===decodeURIComponent(url.pathname.split('/')[4]));
    if (options.method==='PATCH') Object.assign(section,body);
    data={section:clone(section)};
  }
  else if (url.pathname==='/api/admin/users/mini-app-notice') data={status:'idle',sent:0,failed:0,total:1};
  else if (url.pathname==='/api/admin/users') {
    const query=(url.searchParams.get('q')||'').toLowerCase();
    const status=url.searchParams.get('status');
    const match=(!query || 'тестова користувачка 42'.includes(query)) && (status!=='active' || detail.access.tier==='full') && (status!=='inactive' || detail.access.tier!=='full');
    data={items:match?[{user_id:42,display_name:'Тестова Користувачка',access:clone(detail.access)}]:[],offset:0,limit:10,has_prev:false,has_next:false,counts:{active:detail.access.tier==='full'?1:0,expired:detail.access.tier==='full'?0:1}};
  }
  else if (url.pathname==='/api/admin/users/42') data=snapshot;
  else if (url.pathname==='/api/admin/users/42/access') {
    detail.access={tier:body.access,sub_infinite:body.access==='full',sub_end:null};
    detail.section_controls.forEach(section=>{section.has_access=body.access==='full';});
    data=clone(detail);
  }
  else if (url.pathname.startsWith('/api/admin/users/42/sections/')) {
    const section=detail.section_controls.find(item=>item.key===decodeURIComponent(url.pathname.split('/').at(-1)));
    if (section.control_mode==='visibility') section.visible=body.enabled;
    else {section.has_access=body.enabled;section.admin_decision=body.enabled;}
    data=clone(detail);
  }
  else data={items:[],counts:{active:0,expired:1},offset:0,limit:10,status:'idle',total:0};
  return new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}});
};
await import('/static/app.js?v=20261014-admin-compact-01');
