"""Admin commerce service methods kept separate from Mini App learning flows."""
from fastapi import HTTPException

from commerce import CommerceStore, PACKAGE_TITLES, sync_star_transactions
from sections import ALWAYS_FREE_SECTION_KEYS, build_sections


def reject(status, code, message):
    raise HTTPException(status_code=status, detail={"code":code,"message":message})


class CommerceAdminMixin:
    def commerce_admin(self, auth):
        if not auth.is_admin:
            reject(403,"forbidden","Потрібні права адміністратора.")

    def commerce_reason(self, reason):
        if not isinstance(reason,str) or not reason.strip() or len(reason.strip()) > 1000:
            reject(400,"invalid_reason","Вкажіть причину зміни (до 1000 символів).")
        return reason.strip()

    async def commerce_user(self, user_id):
        user = await self.store.get_user(user_id)
        if not user:
            reject(404,"user_not_found","Користувача не знайдено.")
        return user

    async def grant_products(self):
        products = [{"product_key":key,"product_title":title} for key,title in PACKAGE_TITLES.items()]
        for section in await build_sections(self.store,is_admin=True):
            if section["key"] not in ALWAYS_FREE_SECTION_KEYS:
                products.append({"product_key":f"section:{section['key']}","product_title":section["title"]})
        # Bundled/manual-only banks may have no database section row.
        qb = getattr(self.runtime,"qb",None)
        for bank in getattr(qb,"attestation_banks",{}).values():
            key = getattr(bank,"manual_grant_section_key","")
            if key and all(row["product_key"] != f"section:{key}" for row in products):
                products.append({"product_key":f"section:{key}","product_title":bank.title})
        return products

    async def commerce_product(self, key):
        products = await self.grant_products()
        product = next((row for row in products if row["product_key"] == key),None)
        if not product:
            reject(400,"invalid_product","Оберіть чинний товар зі списку доступів.")
        return product

    async def admin_set_grant(self,auth,target_id,product_key,enabled,reason):
        self.commerce_admin(auth)
        reason = self.commerce_reason(reason)
        await self.commerce_user(target_id)
        product = None
        if not enabled:
            detail = await CommerceStore(self.store).user_commerce(target_id)
            product = next((row for row in detail["manual_grants"] if row["product_key"] == product_key),None)
        if product is None:
            product = await self.commerce_product(product_key)
        try:
            await CommerceStore(self.store).set_manual_grant(target_id,product_key,product["product_title"],enabled,auth.user_id,reason)
        except ValueError as error:
            reject(400,"invalid_product",str(error))
        return await self.admin_user_detail(auth,target_id)

    async def admin_restore_purchases(self,auth,target_id,reason):
        self.commerce_admin(auth)
        reason = self.commerce_reason(reason)
        await self.commerce_user(target_id)
        await CommerceStore(self.store).restore_purchases(target_id,auth.user_id,reason)
        return await self.admin_user_detail(auth,target_id)

    async def admin_record_payment(self,auth,target_id,payload):
        self.commerce_admin(auth)
        reason = self.commerce_reason(payload.reason)
        await self.commerce_user(target_id)
        product = await self.commerce_product(payload.product_key)
        amount = payload.amount
        if isinstance(amount,bool) or not isinstance(amount,int) or not 1 <= amount <= 1_000_000_000 or not isinstance(payload.currency,str) or payload.currency not in {"XTR","UAH"}:
            reject(400,"invalid_amount","Вкажіть додатну цілу суму (до 1 000 000 000) та валюту XTR або UAH. Для UAH сума в копійках.")
        reference,request_id = payload.reference.strip(),payload.request_id.strip()
        if not reference or len(reference)>200 or not request_id or len(request_id)>100:
            reject(400,"invalid_reference","Вкажіть номер платежу та унікальний номер запиту.")
        try:
            await CommerceStore(self.store).record_payment(user_id=target_id,product_key=product["product_key"],product_title=product["product_title"],
                amount=amount,currency=payload.currency,reference=reference,source="admin_external",request_id=request_id,actor_id=auth.user_id,reason=reason)
        except ValueError as error:
            reject(409,"payment_request_conflict",str(error))
        return await self.admin_user_detail(auth,target_id)

    async def admin_payments(self,auth,query,offset,limit):
        self.commerce_admin(auth)
        return await CommerceStore(self.store).payments_page(query.strip()[:100],max(0,offset),max(1,min(100,limit)))

    async def admin_activity(self,auth,offset,limit):
        self.commerce_admin(auth)
        return await CommerceStore(self.store).activity_page(max(0,offset),max(1,min(100,limit)))

    async def admin_sync_payments(self,auth,offset,limit):
        self.commerce_admin(auth)
        if not self.runtime.bot:
            reject(503,"bot_unavailable","Telegram-бот зараз недоступний.")
        products = {row["product_key"]:row["product_title"] for row in await self.grant_products()}
        # Previously purchased products can have been removed from today's sales catalog.
        for row in await self.store._fetch("SELECT DISTINCT product_key,product_title FROM payments"):
            products.setdefault(row["product_key"],row["product_title"])
        return await sync_star_transactions(self.runtime.bot,self.store,products,offset=offset,limit=limit,actor_id=auth.user_id)
