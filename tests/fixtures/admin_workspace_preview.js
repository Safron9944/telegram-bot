// Synthetic records only. No production API request is made by this fixture.
const sections=[
 {key:'attestation:7',title:'Атестація посадових осіб митних органів — підготовка до тестування',kind:'attestation',bank_id:7,bank_slug:'test-bank',group:'primary',visible:true,has_access:true,price:100,questions_count:800,icon:'document',deletable:true},
 {key:'customs',title:'Митні компетенції',group:'primary',visible:true,has_access:true,price:250,content_screen:'admin-questions',icon:'graduation'},
 {key:'cases',title:'Кейси',group:'materials',visible:true,has_access:true,price:100,content_screen:'admin-cases',icon:'folder'},
 {key:'test_questions',title:'Тестові питання',group:'materials',visible:true,has_access:true,price:250,content_screen:'admin-test-questions',icon:'clipboard'},
 {key:'question_search',title:'Пошук питань',group:'materials',visible:true,has_access:true,price:250,icon:'search'},
 {key:'support',title:'Підтримка',group:'help',visible:true,has_access:true,price:0,icon:'support'},
];
const clone=value=>JSON.parse(JSON.stringify(value));
const purchase={id:1,product_key:'full',product_title:'Повний пакет',amount:250,currency:'XTR',paid_at:'2026-10-12T13:01:00Z',source:'telegram',status:'confirmed',reference:'TEST-CHARGE-42'};
const products=[{product_key:'full',product_title:'Повний пакет'},{product_key:'cases',product_title:'Пакет атестації'},...sections.filter(s=>s.price>0).map(s=>({product_key:`section:${s.key}`,product_title:s.title}))];
const makeDetail=(id,name,purchases=[],legacy=[])=>({user_id:id,first_name:name,last_name:'Користувачка',is_admin:false,created_at:'2026-01-21T12:58:00Z',access:{tier:purchases.length||legacy.length?'full':'none',sub_infinite:!!(purchases.length||legacy.length),sub_end:null},stats:{count:4,avg:95,last:{correct:113,total:130,finished_at:'2026-10-12T13:01:00Z'}},purchases,manual_grants:[],legacy_access:legacy,access_events:[],grant_products:clone(products),section_controls:sections.filter(s=>s.price>0).map(s=>({key:s.key,title:s.title,group:s.group,has_access:!!(purchases.length||legacy.length),visible:!!(purchases.length||legacy.length),control_mode:'access',access_source:purchases.length?'paid':legacy.length?'legacy':'none',purchased:!!purchases.length}))});
const details=new Map([[42,makeDetail(42,'Тестова',[clone(purchase)])],[43,makeDetail(43,'Друга')],[44,makeDetail(44,'Старий',[],[{product_key:'full',product_title:'Повний пакет'}])]]);
const settings={price_full:250,price_cases:100,admin_contact_url:'https://t.me/test_support'};
const nativeFetch=window.fetch.bind(window);
window.adminPreview={requests:[],failNext:false,delayMs:20,details,detail:details.get(42),settings,sections};
const summary=d=>({paid_count:d.purchases.filter(p=>p.status==='confirmed').length,payment_totals:{XTR:d.purchases.filter(p=>p.status==='confirmed'&&p.currency==='XTR').reduce((v,p)=>v+(p.amount||0),0),UAH:d.purchases.filter(p=>p.status==='confirmed'&&p.currency==='UAH').reduce((v,p)=>v+(p.amount||0),0)},purchase_titles:d.purchases.filter(p=>p.status==='confirmed').map(p=>p.product_title),manual_count:d.manual_grants.length,legacy_count:d.legacy_access.length,attention_count:d.legacy_access.length});
window.fetch=async(path,options={})=>{
 if(!String(path).startsWith('/api/'))return nativeFetch(path,options);
 const url=new URL(path,window.location.origin),body=options.body?JSON.parse(options.body):{},preview=window.adminPreview;
 preview.requests.push({path:url.pathname,query:url.search,method:options.method||'GET',body});
 const match=url.pathname.match(/^\/api\/admin\/users\/(\d+)(?:\/(.+))?$/),id=match?Number(match[1]):null,detail=details.get(id),snapshot=detail?clone(detail):null;
 if(preview.holdPost&&options.method==='POST'){await new Promise(resolve=>{preview.releasePost=resolve});preview.releasePost=null;}
 if(preview.holdDelete&&options.method==='DELETE'){await new Promise(resolve=>{preview.releaseDelete=resolve});preview.releaseDelete=null;}
 if(preview.holdPatch&&options.method==='PATCH'){await new Promise(resolve=>{preview.releasePatch=resolve});preview.releasePatch=null;}
 await new Promise(resolve=>setTimeout(resolve,match&&!match[2]?(preview.getDetailDelayMs||preview.delayMs):preview.delayMs));
 if(preview.failNext&&options.method==='POST'){preview.failNext=false;return new Response(JSON.stringify({detail:{message:'Тестова помилка збереження'}}),{status:503});}
 if(url.pathname==='/api/admin/payments/sync'&&preview.partialSync){
  if(Number(body.offset)===0){details.get(43).purchases=[{...clone(purchase),id:9}];return new Response(JSON.stringify({imported:1,skipped:2,next_offset:100,has_more:true}));}
  preview.partialSync=false;return new Response(JSON.stringify({detail:{message:'Тестова помилка другої сторінки'}}),{status:503});
 }
 let data;
 if(url.pathname==='/api/bootstrap')data={user:{id:1,is_admin:true,first_name:'Адміністратор'},catalog:{attestation_banks:[],ok_modules:[]},sections:clone(sections),saved_view:null,payment_prices:{full:settings.price_full,cases:100},links:{admin_url:settings.admin_contact_url},stats:{count:0,avg:0,last:null}};
 else if(url.pathname==='/api/admin/settings'){if(options.method==='POST')Object.assign(settings,body);data=clone(settings);}
 else if(url.pathname==='/api/admin/sections')data={items:clone(sections)};
 else if(url.pathname.startsWith('/api/admin/sections/')){const s=sections.find(s=>s.key===decodeURIComponent(url.pathname.split('/')[4]));if(options.method==='PATCH')Object.assign(s,body);data={section:clone(s)};}
 else if(url.pathname==='/api/admin/users/mini-app-notice')data={state:'idle',sent:0,failed:0,total:3};
 else if(url.pathname==='/api/admin/users'){
  const q=(url.searchParams.get('q')||'').toLowerCase(),filter=url.searchParams.get('status')||'all';
  data={items:[...details.values()].filter(d=>`${d.first_name} користувачка ${d.user_id}`.toLowerCase().includes(q)).filter(d=>filter==='all'||filter==='paid'&&d.purchases.length||filter==='manual'&&d.manual_grants.length||filter==='attention'&&d.legacy_access.length).map(d=>({...clone(d),display_name:`${d.first_name} Користувачка`,commerce_summary:summary(d)})),offset:0,limit:20,has_prev:false,has_next:false,counts:{active:1,expired:2}};
 }else if(match&&detail){
  if(!match[2]){if(options.method==='DELETE'){details.delete(id);data={ok:true};}else data=snapshot;}
  else if(match[2]==='grants'){
   const p=products.find(p=>p.product_key===body.product_key);detail.manual_grants=detail.manual_grants.filter(g=>g.product_key!==body.product_key);
   if(body.enabled)detail.manual_grants.push({...clone(p),created_at:'2026-10-14T10:00:00Z',actor_id:1,reason:body.reason});
   detail.access_events.unshift({user_id:id,actor_id:1,action:body.enabled?'grant_added':'grant_removed',product_key:body.product_key,product_title:p?.product_title,reason:body.reason,created_at:'2026-10-14T10:00:00Z'});data=clone(detail);
  }else if(match[2]==='payments'){
   if(!detail.purchases.some(p=>p.request_id===body.request_id)){const p=products.find(p=>p.product_key===body.product_key);detail.purchases.push({id:detail.purchases.length+2,...clone(p),amount:body.amount,currency:body.currency,paid_at:'2026-10-14T10:00:00Z',source:'admin_external',status:'confirmed',reference:body.reference||body.request_id,request_id:body.request_id});}data=clone(detail);
  }else if(match[2]==='restore-purchases'){detail.section_controls.filter(s=>s.purchased).forEach(s=>{s.visible=true;s.has_access=true});data=clone(detail);}
 }else if(url.pathname==='/api/admin/payments/sync')data={imported:0,skipped:1,next_offset:1,has_more:false};
 else if(url.pathname==='/api/admin/payments'){const q=(url.searchParams.get('q')||'').toLowerCase();data={items:[...details.values()].flatMap(d=>d.purchases.map(p=>({...clone(p),user_id:d.user_id,display_name:`${d.first_name} Користувачка`}))).filter(p=>`${p.user_id} ${p.display_name} ${p.product_title}`.toLowerCase().includes(q)),offset:0,limit:20,has_prev:false,has_next:false};}
 else if(url.pathname==='/api/admin/activity')data={items:[...details.values()].flatMap(d=>clone(d.access_events)),offset:0,limit:20,has_prev:false,has_next:false};
 else if(url.pathname==='/api/admin/global-search')data={ok:[{id:1,question:'Тестове питання',topic:'Тестовий розділ'}],attestation:[],cases:[],test:[]};
 else if(url.pathname==='/api/admin/questions/1')data={question:{id:1,question:'Тестове питання',choices:[{index:1,text:'Відповідь',is_correct:true}]}};
 else data={items:[],counts:{active:1,expired:2},offset:0,limit:20,total:0};
 return new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}});
};
await import('/static/app.js?v=20261014-admin-workspace-01');
