import assert from 'node:assert/strict';
import path from 'node:path';

export async function runAdminPanelChecks(browser,{baseUrl,artifactDir}) {
  const reports=[];
  for(const width of [360,1280]){
    const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce'}),page=await context.newPage();
    page.setDefaultTimeout(5000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const capture=async name=>{await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:path.join(artifactDir,`${name}-${width}.png`),fullPage:true,animations:'disabled'});};
    try{
      await page.goto(`${baseUrl}/tests/fixtures/admin_preview.html`);
      await page.addStyleTag({content:'html { scrollbar-width: none; }'});
      await page.locator('[data-screen-target="admin"]').click();
      assert.equal(await page.locator('.aw-client-layout').count(),1,'admin must open the new customer workspace');
      await page.locator('[data-aw-user="42"]').click();
      await page.locator('.aw-purchase').waitFor();
      assert.match(await page.locator('.aw-inspector').innerText(),/250.*Telegram|Telegram.*250/s);
      await capture('workspace-customer');
      await page.locator('[data-aw-action="grant"]').click();
      await page.locator('[name="product_key"]').selectOption('full');
      await page.locator('[name="reason"]').fill('Подарунок для перевірки');
      await page.getByRole('button',{name:'Переглянути зміну',exact:true}).click();
      await page.locator('[data-aw-confirm]').click();
      await page.locator('[data-aw-revoke="full"]').waitFor();
      await page.locator('[data-aw-revoke="full"]').click();
      await page.locator('[name="reason"]').fill('Скасування подарунка');
      await page.getByRole('button',{name:'Переглянути зміну',exact:true}).click();
      await page.locator('[data-aw-confirm]').click();
      await page.getByText('Ручний дозвіл скасовано.',{exact:true}).waitFor();
      assert.equal(await page.locator('.aw-purchase').count(),1);
      assert.equal(await page.evaluate(()=>window.adminPreview.requests.some(r=>r.path.endsWith('/access'))),false);
      await page.locator('[data-aw-action="grant"]').click();
      await page.locator('[name="reason"]').fill('Помилка у прикладі');
      await page.getByRole('button',{name:'Переглянути зміну',exact:true}).click();
      await page.evaluate(()=>{window.adminPreview.failNext=true});
      await page.locator('[data-aw-confirm]').click();
      await page.getByText('Тестова помилка збереження',{exact:true}).waitFor();
      assert.equal(await page.locator('.aw-purchase').count(),1);
      await page.locator('[data-aw-cancel]').click();
      await page.locator('[data-aw-action="record-payment"]').click();
      await page.locator('[name="amount"]').fill('500');
      await page.locator('[name="currency"]').selectOption('UAH');
      await page.locator('[name="reason"]').fill('Підтверджений переказ');
      await page.getByRole('button',{name:'Переглянути зміну',exact:true}).click();
      await page.locator('[data-aw-confirm]').click();
      await page.getByText('Оплату записано.',{exact:true}).waitFor();
      assert.equal(await page.locator('.aw-purchase').count(),2);
      await page.locator('[data-aw-destination="clients"]').click();
      await page.locator('[data-aw-view="payments"]').click();
      await page.locator('.aw-payment-row').first().waitFor();
      await capture('workspace-payments');
      await page.locator('[data-aw-sync]').click();
      await page.getByText(/Звірку завершено/).waitFor();
      await page.locator('[data-aw-destination="content"]').click();
      await page.locator('[data-aw-section="customs"]').click();
      await page.locator('[name="price"]').fill('275');
      await page.getByRole('button',{name:'Зберегти налаштування',exact:true}).click();
      await page.getByText('Налаштування збережено.',{exact:true}).waitFor();
      await capture('workspace-content');
      await page.locator('[data-aw-destination="system"]').click();
      await page.locator('#admin-contact-url').waitFor();
      await capture('workspace-system');
      await page.locator('[data-aw-system="activity"]').click();
      await page.locator('.aw-event').first().waitFor();
      await capture('workspace-activity');
      assert.deepEqual(errors,[]);reports.push({width,passed:true});
    }finally{await context.close();}
  }return reports;
}

export async function runAdminRaceChecks(browser,{baseUrl}) {
  const reports=[];
  for(const width of [360,1280]){
    const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce'}),page=await context.newPage();page.setDefaultTimeout(5000);
    const close=async()=>{if(width<960)await page.locator('[data-aw-close]').click();};
    try{
      await page.goto(`${baseUrl}/tests/fixtures/admin_preview.html`);await page.locator('[data-screen-target="admin"]').click();await page.locator('[data-aw-user="42"]').waitFor();
      await page.evaluate(()=>{window.adminPreview.getDetailDelayMs=350});
      await page.locator('[data-aw-user="42"]').click();
      if(width<960){await page.locator('[data-aw-destination="clients"]').click();await page.locator('[data-aw-view="payments"]').click();await page.locator('[data-aw-view="users"]').click();}
      await page.locator('[data-aw-user="43"]').click();await page.locator('[data-aw-customer="43"]').waitFor();
      assert.equal(await page.locator('.aw-purchase').count(),0,'slow previous customer response must not overwrite selection');
      await close();await page.locator('[data-aw-user="42"]').click();await page.locator('.aw-purchase').waitFor();
      await page.locator('[data-aw-action="grant"]').click();await page.locator('[name="reason"]').fill('Причина для клієнта 42');await page.getByRole('button',{name:'Переглянути зміну',exact:true}).click();
      await page.evaluate(()=>{window.adminPreview.holdPost=true});await page.locator('[data-aw-confirm]').click();await page.waitForFunction(()=>!!window.adminPreview.releasePost);
      await close();await page.locator('[data-aw-user="43"]').click();await page.locator('[data-aw-customer="43"]').waitFor();assert.equal(await page.locator('[data-aw-confirm]').count(),0);
      await close();await page.locator('[data-aw-user="42"]').click();assert.equal(await page.locator('[data-aw-confirm]').isDisabled(),true,'pending mutation remains locked across reentry');
      await page.evaluate(()=>{const p=window.adminPreview;p.holdPost=false;p.releasePost()});await page.locator('[data-aw-revoke="full"]').waitFor();assert.equal(await page.locator('.aw-purchase').count(),1);
      const posts=await page.evaluate(()=>window.adminPreview.requests.filter(r=>r.path.endsWith('/grants')&&r.method==='POST'));assert.equal(posts.length,1);assert.equal(posts[0].path,'/api/admin/users/42/grants');
      reports.push({width,staleCustomerProtected:true,pendingReentryProtected:true});
    }finally{await context.close();}
  }return reports;
}

export async function runAdminFollowupChecks(browser,{baseUrl}) {
  const reports=[];
  for(const width of [360,1280]){
    const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce'}),page=await context.newPage();page.setDefaultTimeout(5000);
    page.on('dialog',dialog=>dialog.accept());
    const closeCustomer=async()=>{if(width<960)await page.locator('[data-aw-close]').click();};
    try{
      await page.goto(`${baseUrl}/tests/fixtures/admin_preview.html`);await page.locator('[data-screen-target="admin"]').click();await page.locator('[data-aw-user="43"]').click();await page.locator('[data-aw-customer="43"]').waitFor();
      await page.locator('summary').filter({hasText:'Навчальні результати'}).click();await page.evaluate(()=>{window.adminPreview.holdDelete=true});await page.locator('[data-aw-delete]').click();await page.waitForFunction(()=>!!window.adminPreview.releaseDelete);
      await closeCustomer();await page.locator('[data-aw-user="42"]').click();await page.locator('[data-aw-customer="42"]').waitFor();
      await page.evaluate(()=>{const p=window.adminPreview;p.holdDelete=false;p.releaseDelete()});await page.waitForFunction(()=>!window.adminPreview.details.has(43));
      await page.locator('[data-aw-user="44"]').waitFor({state:'attached'});assert.equal(await page.locator('[data-aw-customer="42"]').count(),1,'delete completion must retain a different selected customer');
      await page.locator('[data-aw-view="payments"]').click();await page.locator('.aw-payment-row').first().waitFor();
      if(width>=960)await page.locator('.aw-payment-row').first().click();
      await page.locator('[data-aw-sync]').click();await page.getByText(/Звірку завершено/).waitFor();
      if(width<960)await page.locator('.aw-payment-row').first().click();await page.locator('[data-aw-customer="42"]').waitFor();assert.equal(await page.locator('.aw-purchase').count(),1,'sync must reload the selected inspector');
      await page.locator('[data-aw-destination="content"]').click();await page.locator('[data-aw-section="customs"]').click();await page.locator('[name="price"]').fill('275');await page.evaluate(()=>{window.adminPreview.holdPatch=true});await page.getByRole('button',{name:'Зберегти налаштування',exact:true}).click();await page.waitForFunction(()=>!!window.adminPreview.releasePatch);
      if(width<960)await page.locator('[data-aw-content-close]').click();await page.locator('[data-aw-section="cases"]').click();
      await page.evaluate(()=>{const p=window.adminPreview;p.holdPatch=false;p.releasePatch()});await page.waitForFunction(()=>window.adminPreview.sections.find(s=>s.key==='customs').price===275);
      await page.locator('[name="title"]').waitFor();assert.equal(await page.locator('[name="title"]').inputValue(),'Кейси');
      await page.locator('[data-aw-material-tool="settings"]').click();await page.locator('#admin-section-title').waitFor();assert.equal(await page.locator('#admin-section-title').inputValue(),'Кейси','late catalog save must not target a different section editor');
      reports.push({width,deleteSelectionProtected:true,syncInspectorReloaded:true,contentSelectionProtected:true});
    }finally{await context.close();}
  }return reports;
}

export async function runAdminPartialSyncCheck(browser,{baseUrl}) {
  const context=await browser.newContext({viewport:{width:1280,height:900}}),page=await context.newPage();page.setDefaultTimeout(5000);
  try{
    await page.goto(`${baseUrl}/tests/fixtures/admin_preview.html`);await page.locator('[data-screen-target="admin"]').click();await page.locator('[data-aw-user="43"]').click();await page.locator('[data-aw-customer="43"]').waitFor();
    await page.locator('[data-aw-view="payments"]').click();await page.locator('.aw-payment-row').first().click();await page.locator('[data-aw-customer="42"]').waitFor();
    await page.evaluate(()=>{window.adminPreview.partialSync=true});await page.locator('[data-aw-sync]').click();await page.getByText(/Тестова помилка другої сторінки.*додано 1.*пропущено 2/).waitFor();
    assert.equal(await page.locator('[data-aw-sync]').textContent(),'Продовжити звірку Telegram');
    await page.locator('[data-aw-user="43"]').click();await page.locator('[data-aw-customer="43"] .aw-purchase').waitFor();
    await page.locator('[data-aw-sync]').click();await page.getByText(/Звірку завершено/).waitFor();
    const offsets=await page.evaluate(()=>window.adminPreview.requests.filter(r=>r.path.endsWith('/sync')).map(r=>r.body.offset));assert.equal(offsets.join(','),'0,100,100');
    return {partialCountsPreserved:true,unselectedCustomerRefreshed:true,resumedOffset:true};
  }finally{await context.close();}
}

export async function runAdminSearchFocusChecks(browser,{baseUrl}) {
  const reports=[];
  for(const width of [360,1280]){
    const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce'}),page=await context.newPage();page.setDefaultTimeout(5000);
    try{
      await page.goto(`${baseUrl}/tests/fixtures/admin_preview.html`);await page.locator('[data-screen-target="admin"]').click();await page.locator('[data-aw-user="42"]').click();await page.locator('.aw-purchase').waitFor();
      assert.equal(await page.evaluate(()=>document.activeElement.dataset.awFocus),'customer-title');
      await page.locator('[data-aw-action="grant"]').click();await page.locator('[name="reason"]').fill('Перевірка фокуса');await page.getByRole('button',{name:'Переглянути зміну',exact:true}).click();assert.equal(await page.evaluate(()=>document.activeElement.hasAttribute('data-aw-confirm')),true);
      if(width<960){await page.locator('[data-aw-close]').click();assert.equal(await page.evaluate(()=>document.activeElement.dataset.awUser),'42');}
      await page.locator('[data-aw-view="payments"]').click();await page.locator('[name="query"]').fill('42');await page.getByRole('button',{name:'Знайти',exact:true}).click();await page.locator('.aw-payment-row').first().waitFor();
      await page.locator('[data-aw-view="users"]').click();await page.waitForFunction(()=>document.querySelectorAll('.aw-client-row').length===1);assert.equal(await page.locator('.aw-client-row').first().getAttribute('data-aw-user'),'42');
      await page.locator('[name="query"]').fill('43');await page.getByRole('button',{name:'Знайти',exact:true}).click();await page.locator('[data-aw-user="43"]').waitFor();await page.locator('[data-aw-view="payments"]').click();await page.getByRole('heading',{name:'Нічого не знайдено',exact:true}).waitFor();assert.equal(await page.locator('.aw-payment-row').count(),0);
      await page.locator('[data-aw-destination="content"]').click();await page.locator('[data-aw-section="customs"]').click();assert.equal(await page.evaluate(()=>document.activeElement.dataset.awFocus),'material-title');
      if(width<960){await page.locator('[data-aw-content-close]').click();assert.equal(await page.evaluate(()=>document.activeElement.dataset.awSection),'customs');}
      reports.push({width,crossViewSearchConsistent:true,inspectorReviewReturnFocus:true});
    }finally{await context.close();}
  }return reports;
}
