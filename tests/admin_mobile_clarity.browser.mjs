import assert from 'node:assert/strict';
import path from 'node:path';

export async function runMobileClarityChecks(browser,{baseUrl,artifactDir}){
  const reports=[];
  for(const [width,height] of [[360,820],[390,820],[640,390],[1280,820]]){
    const mobile=width<960,context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'}),page=await context.newPage();page.setDefaultTimeout(6000);
    try{
      await page.goto(`${baseUrl}/tests/fixtures/admin_preview.html`);
      await page.addStyleTag({content:'html{scrollbar-width:none}'});
      if(mobile)await page.evaluate(()=>{document.documentElement.classList.add('telegram-fullscreen');document.documentElement.style.setProperty('--tg-safe-area-inset-top','24px');document.documentElement.style.setProperty('--tg-content-safe-area-inset-top','48px');document.documentElement.style.setProperty('--tg-safe-area-inset-bottom','24px');});
      if(width===640)await page.evaluate(()=>{document.documentElement.style.setProperty('--tg-safe-area-inset-left','24px');document.documentElement.style.setProperty('--tg-content-safe-area-inset-left','8px');});
      if(mobile)await page.evaluate(()=>{const overlay=document.createElement('div');overlay.dataset.telegramEmulation='true';overlay.setAttribute('aria-hidden','true');overlay.style.cssText='position:fixed;top:24px;left:12px;right:12px;height:40px;display:flex;justify-content:space-between;align-items:center;pointer-events:none;z-index:1000';overlay.innerHTML='<span style="background:#b2b4b8;color:white;border-radius:20px;padding:7px 12px;font:14px system-ui">Назад</span><svg width="56" height="34" viewBox="0 0 56 34"><rect width="56" height="34" rx="17" fill="#b2b4b8"/><path d="m13 14 5 5 5-5" fill="none" stroke="white" stroke-width="2"/><circle cx="40" cy="10" r="1.5" fill="white"/><circle cx="40" cy="17" r="1.5" fill="white"/><circle cx="40" cy="24" r="1.5" fill="white"/></svg>';document.body.append(overlay)});
      await page.locator('[data-screen-target="admin"]').click();await page.locator('[data-aw-user="42"]').waitFor();
      const geometry=await page.evaluate(()=>{const rect=s=>{const r=document.querySelector(s).getBoundingClientRect();return{top:r.top,bottom:r.bottom,left:r.left,right:r.right,height:r.height}};return{heading:rect('.aw-page-heading'),row:rect('.aw-client-row'),nav:rect('.aw-destinations'),controls:rect('.aw-destinations button'),ink:getComputedStyle(document.body).getPropertyValue('--aw-accent').trim(),overflow:document.documentElement.scrollWidth>innerWidth};});
      assert.equal(geometry.ink,'#10233f');assert.equal(geometry.overflow,false);
      if(mobile){assert.ok(geometry.heading.top>=72,'Telegram buttons must not overlap content');assert.ok(geometry.row.top<Math.min(320,height-100),'client records must be visible immediately');assert.ok(geometry.row.height<=104,'typical client row must be compact');assert.ok(geometry.nav.top>height-140,'main spaces belong in bottom navigation');assert.ok(geometry.controls.bottom<=height-24,'navigation must respect system bottom inset');if(width===640)assert.ok(geometry.row.left>=32,'landscape must respect both side insets');}
      const paidText=await page.locator('[data-aw-user="42"]').innerText();assert.match(paidText,/Повний пакет/);assert.match(paidText,/Telegram/);assert.match(paidText,/250/);
      const emptyText=await page.locator('[data-aw-user="43"]').innerText();assert.match(emptyText,/Оплата не зафіксована/);assert.doesNotMatch(emptyText,/Покупок немає|Без покупок|Підтверджених оплат немає/);
      assert.match(await page.locator('[data-aw-user="45"]').innerText(),/Кейси.*Доступ вручну/s);
      assert.match(await page.locator('[data-aw-user="47"]').innerText(),/Переказ.*500|500.*Переказ/s);
      assert.match(await page.locator('[data-aw-user="48"]').innerText(),/Вручну: Кейси/);
      assert.match(await page.locator('[data-aw-user="49"]').innerText(),/Вручну: Митні компетенції/);
      await page.screenshot({path:path.join(artifactDir,`compact-clients-${width}.png`),fullPage:true,animations:'disabled'});
      if(mobile){await page.locator('#aw-client-filter').focus();await page.locator('#aw-client-filter').selectOption('paid');await page.waitForFunction(()=>document.activeElement.id==='aw-client-filter'&&!document.querySelector('.aw-loading'));assert.equal(await page.locator('[data-aw-user="43"]').count(),0);await page.locator('#aw-client-filter').selectOption('all');await page.locator('[data-aw-user="43"]').waitFor();}
      if(mobile){await page.evaluate(()=>document.documentElement.style.setProperty('--tg-content-safe-area-inset-top','64px'));const top=await page.locator('.aw-page-heading').evaluate(n=>n.getBoundingClientRect().top);assert.ok(top>=88,'updated Telegram safe area must be honored');}
      await page.locator('[data-aw-user="42"]').click();await page.locator('.aw-purchase').waitFor();assert.equal(await page.evaluate(()=>document.activeElement.dataset.awFocus),'customer-title');
      await page.screenshot({path:path.join(artifactDir,`compact-customer-${width}.png`),fullPage:true,animations:'disabled'});
      await page.locator('[data-aw-destination="content"]').click();await page.locator('[data-aw-section="customs"]').click();await page.locator('[name="price"]').waitFor();
      if(mobile){const last=page.locator('[data-aw-material-tool="order"]');await page.evaluate(()=>window.scrollTo(0,document.documentElement.scrollHeight));const bounds=await last.boundingBox(),nav=await page.locator('.aw-destinations').boundingBox();assert.ok(bounds.y+bounds.height<=nav.y,'last editor tool must scroll above navigation');await page.evaluate(()=>window.scrollTo(0,0));}
      await page.screenshot({path:path.join(artifactDir,`compact-content-${width}.png`),fullPage:true,animations:'disabled'});
      await page.locator('[data-aw-destination="system"]').click();await page.locator('#admin-contact-url').waitFor();if(mobile){const exit=page.locator('[data-aw-exit]');assert.equal(await exit.evaluate(n=>{const r=n.getBoundingClientRect();return !!document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.closest('[data-aw-exit]')}),true,'legacy headers must not cover the exit');}await page.screenshot({path:path.join(artifactDir,`compact-system-${width}.png`),fullPage:true,animations:'disabled'});
      reports.push({width,compact:true,safeAreas:true,truthfulSummary:true});
    }finally{await context.close();}
  }return reports;
}
