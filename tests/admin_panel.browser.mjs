import assert from 'node:assert/strict';
import path from 'node:path';

/** Run against a local static server and synthetic API fixture, never production. */
export async function runAdminPanelChecks(browser, { baseUrl, artifactDir }) {
  const reports = [];
  for (const viewport of [{width:360,height:800},{width:1280,height:900}]) {
    const context = await browser.newContext({viewport});
    const page = await context.newPage();
    page.setDefaultTimeout(7000);
    const errors = [];
    page.on('pageerror',error=>errors.push(error.message));
    const suffix = viewport.width < 700 ? 'mobile' : 'desktop';
    const capture = async name => {
      if (await page.locator('#dismiss-message').count()) await page.locator('#dismiss-message').click();
      await page.evaluate(()=>window.scrollTo(0,0));
      await page.emulateMedia({reducedMotion:'reduce'});
      const overflow = await page.evaluate(()=>document.documentElement.scrollWidth > window.innerWidth);
      assert.equal(overflow,false,`${name} overflows at ${viewport.width}px`);
      await page.screenshot({path:path.join(artifactDir,`${name}-${suffix}.png`),fullPage:true,animations:'disabled'});
    };
    const menu = async () => { await page.locator('[data-admin-home]').click(); };
    const open = async screen => { await page.locator(`[data-screen-target="${screen}"]`).click(); };
    const releasePost = async () => page.evaluate(()=>{
      const preview=window.adminPreview;
      if (!preview.releasePost) throw new Error('Expected a pending test POST');
      preview.holdPost=false;
      preview.releasePost();
    });
    try {
      await page.goto(`${baseUrl}/tests/fixtures/admin_preview.html`);
      await open('admin');
      assert.equal(await page.locator('.admin-hub-screen .cell').count(),5);
      await capture('admin-hub');
      await open('admin-users');
      await page.locator('.admin-user-row').waitFor();
      await page.locator('#admin-user-search').fill('42');
      await page.locator('#admin-user-search-form button[type="submit"]').click();
      await page.locator('.admin-user-row').waitFor();
      await capture('admin-users');
      await page.locator('.admin-user-row').click();
      await page.locator('#admin-manage-access').waitFor();
      assert.match(await page.locator('.admin-access-overview').innerText(),/Окремий доступ/);
      assert.match(await page.locator('.admin-access-overview').innerText(),/1 з 5/);
      await capture('admin-profile');
      await page.evaluate(()=>{window.adminPreview.getDetailDelayMs=300;});
      await page.locator('#admin-manage-access').click();
      await page.locator('.admin-access-group').first().waitFor();
      await capture('admin-access');
      // A delayed GET started when opening the page must not undo a newer POST.
      const firstLabel = page.locator('.admin-toggle-line').first();
      await firstLabel.click();
      await page.locator('.message--success').waitFor();
      await page.waitForTimeout(350);
      assert.equal(await page.locator('[data-section-control="attestation:7"]').isChecked(),true);
      // The same user's save remains locked after leaving and reopening the page.
      await page.evaluate(()=>{window.adminPreview.holdPost=true;});
      await page.locator('.admin-toggle-line').nth(1).click();
      await page.locator('#admin-access-done').click();
      await page.locator('#admin-manage-access').click();
      assert.equal(await page.locator('#admin-package-action').isDisabled(),true);
      await releasePost();
      await page.getByText('Доступ до «Митні компетенції» закрито.',{exact:true}).waitFor();
      assert.equal(await page.locator('[data-section-control="customs"]').isChecked(),false);
      assert.equal(await page.locator('#admin-package-action').isDisabled(),false);
      // Failed save restores the checkbox and unlocks all access controls.
      await page.evaluate(()=>{window.adminPreview.failNext=true;window.adminPreview.holdPost=true;});
      await page.locator('.admin-toggle-line').first().click();
      assert.equal(await page.locator('#admin-package-action').isDisabled(),true);
      await page.locator('#admin-access-done').click();
      await page.locator('[data-admin-back]').click();
      await page.locator('.admin-user-row').click();
      await releasePost();
      await page.locator('.message--error').waitFor();
      await page.locator('#admin-manage-access').waitFor();
      await page.locator('#admin-manage-access').click();
      assert.equal(await page.locator('[data-section-control="attestation:7"]').isChecked(),true);
      assert.equal(await page.locator('#admin-package-action').isDisabled(),false);
      await page.evaluate(()=>{window.adminPreview.delayMs=25;});
      const cases = page.locator('.admin-toggle-line').nth(2);
      await cases.click();
      await page.locator('.message--success').waitFor();
      assert.equal(await page.evaluate(()=>window.adminPreview.detail.section_controls.find(s=>s.key==='cases').has_access),false);
      await page.locator('#admin-access-done').click();
      await page.locator('[data-admin-back]').click();
      assert.equal(await page.locator('#admin-user-search').inputValue(),'42');
      await menu();
      await open('admin-tariffs');
      await page.locator('#admin-price-full:not([disabled])').waitFor();
      await page.locator('#admin-section-prices .admin-section-row').first().waitFor();
      assert.equal(await page.locator('#admin-section-prices .admin-section-row').count(),6);
      await capture('admin-tariffs');
      await page.locator('#admin-price-full').fill('350');
      await page.locator('#admin-full-price-form button').click();
      await page.getByText('Ціну пакета збережено.',{exact:true}).waitFor();
      assert.equal(await page.evaluate(()=>window.adminPreview.settings.price_full),350);
      await page.locator('#admin-section-prices .admin-section-row').nth(1).click();
      await page.locator('#admin-section-price').waitFor();
      await capture('admin-section-settings');
      await menu();
      await open('admin-attestation-banks');
      await page.locator('.admin-attestation-bank-cell').first().waitFor();
      await capture('admin-materials');
      await open('admin-global-search');
      await page.locator('#global-search-input').fill('питання');
      await page.locator('#global-search-results .cell').first().click();
      await page.locator('#qv-choices .cell').waitFor();
      assert.equal(await page.locator('[data-admin-home]').count(),1,'navigation must survive asynchronous question rendering');
      await menu();
      await open('admin-attestation-banks');
      await page.locator('.admin-attestation-bank-cell').first().waitFor();
      await page.locator('.admin-attestation-bank-cell').first().click();
      assert.equal(await page.locator('[data-screen-target="admin-section-questions"]').count(),1);
      await menu();
      await open('admin-settings');
      await page.locator('#admin-contact-url:not([disabled])').waitFor();
      await capture('admin-settings');
      await page.locator('#admin-contact-url').fill('@new_support');
      await page.locator('#admin-contact-form button').click();
      await page.getByText('Контакт підтримки збережено.',{exact:true}).waitFor();
      assert.equal(await page.evaluate(()=>window.adminPreview.settings.admin_contact_url),'https://t.me/new_support');
      // A committed contact change refreshes support even after navigation away.
      await page.locator('#admin-contact-url:not([disabled])').waitFor();
      await page.locator('#admin-contact-url').fill('@later_support');
      await page.evaluate(()=>{window.adminPreview.holdPost=true;});
      await page.locator('#admin-contact-form button').click();
      await menu();
      await releasePost();
      await page.waitForFunction(async()=> (await import('/static/js/core/state.js?v=20260923-testing-labels-14')).state.bootstrap.links.admin_url==='https://t.me/later_support');
      await open('admin-messages');
      await page.locator('#admin-mini-app-notice').waitFor();
      await capture('admin-messages');
      assert.equal(await page.evaluate(()=>window.adminPreview.requests.filter(r=>r.path.includes('mini-app-notice') && r.method==='POST').length),0);
      assert.deepEqual(errors,[]);
      reports.push({viewport:viewport.width,checks:'navigation, search, individual access, visibility, stale response, failed save, prices, support, messages',screenshots:9});
    } finally {
      await context.close();
    }
  }
  return reports;
}
