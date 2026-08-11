# Attestation Section Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the protected Stage 1 home card and let administrators hide, reorder, and permanently delete APK-created attestation sections.

**Architecture:** Keep bundled Stage 1 outside PostgreSQL management and prepend it to the public catalog. Add focused storage operations for dynamic bank status, ordering, and deletion; expose them through admin-only routes that reload the runtime catalog after mutation. Add a mobile-friendly admin screen consuming those routes.

**Tech Stack:** Python 3.13+, FastAPI, asyncpg/PostgreSQL, vanilla JavaScript Mini App, unittest.

---

### Task 1: Restore the protected Stage 1 card

**Files:**
- Modify: `app.py:575-610`
- Modify: `static/js/screens/user.js:100-125`
- Test: `tests/test_dynamic_attestation_runtime.py`
- Test: `tests/test_dynamic_attestation_assets.py`

- [ ] **Step 1: Write failing catalog and asset tests**

```python
async def test_catalog_keeps_stage_1_before_dynamic_banks(self):
    bank = QuestionBank("unused.json")
    bank.load_attestation_stage_1(str(ROOT / "attestation_stage_1.json"))
    await bank.load_published_attestation_banks(PublishedStore())
    catalog = MiniAppService(SimpleNamespace(qb=bank, store=None)).serialize_catalog(auth)
    self.assertEqual(["stage-1", "stage-2"], [item["slug"] for item in catalog["attestation_banks"]])
```

Assert in the asset test that `renderHome` renders `catalog.attestation_stage_1` as the fixed `stage-1` card before looping through `catalog.attestation_banks`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_dynamic_attestation_runtime tests.test_dynamic_attestation_assets`

Expected: FAIL because the existing public list mixes Stage 1 into the dynamic loop and has no protected prepend behavior.

- [ ] **Step 3: Separate fixed and dynamic catalog entries**

Build the serialized list as:

```python
attestation_banks = [{
    "slug": "stage-1",
    "title": "Атестація посадових осіб — 1 етап",
    "count": len(self.qb.attestation_stage_1),
    "topics": len(attestation_sections),
    "sections": attestation_sections,
    "system": True,
}]
for bank in self.qb.published_attestation_banks():
    if bank.slug == "stage-1":
        continue
    attestation_banks.append({..., "system": False})
```

Keep the home renderer driven by this ordered list so the fixed card always precedes dynamic cards.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m unittest tests.test_dynamic_attestation_runtime tests.test_dynamic_attestation_assets`

Expected: PASS.

### Task 2: Add dynamic-bank management storage operations

**Files:**
- Modify: `storage.py:794-900`
- Test: `tests/test_attestation_management.py`

- [ ] **Step 1: Write failing storage contract tests**

Define tests against a recording async connection for these methods:

```python
await store.list_attestation_banks_for_admin()
await store.set_attestation_bank_visibility(7, visible=False)
await store.move_attestation_bank(7, direction="up")
await store.delete_attestation_bank(7)
```

Assert that the admin list includes `hidden`, visibility maps to `published`/`hidden`, movement swaps adjacent `display_order` values transactionally, and deletion targets only the requested database id.

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_attestation_management`

Expected: ERROR because the storage methods do not exist.

- [ ] **Step 3: Implement minimal storage methods**

Add signatures:

```python
async def list_attestation_banks_for_admin(self) -> list[dict]: ...
async def set_attestation_bank_visibility(self, bank_id: int, *, visible: bool) -> dict | None: ...
async def move_attestation_bank(self, bank_id: int, *, direction: str) -> bool: ...
async def delete_attestation_bank(self, bank_id: int) -> bool: ...
```

Use `SELECT ... FOR UPDATE` inside transactions for reorder, normalize orders to consecutive integers after swap/delete, and rely on the existing foreign-key cascade for questions. Reject any row whose slug is `stage-1` or source is `bundled-stage-1`. Keep `list_published_attestation_banks()` filtered to `status='published'`.

- [ ] **Step 4: Run the storage test and verify GREEN**

Run: `python -m unittest tests.test_attestation_management`

Expected: PASS.

### Task 3: Add admin-only management API

**Files:**
- Modify: `admin_apk_import_extension.py:20-150`
- Test: `tests/test_attestation_management_api.py`

- [ ] **Step 1: Write failing route tests**

Cover these routes with admin and non-admin auth:

```text
GET    /api/admin/attestation-banks
PATCH  /api/admin/attestation-banks/{bank_id}/visibility  {"visible": false}
POST   /api/admin/attestation-banks/{bank_id}/move        {"direction": "up"}
DELETE /api/admin/attestation-banks/{bank_id}
```

Assert 403 for non-admins, 404 for missing ids, 400 for invalid direction, and one runtime reload after every successful mutation.

- [ ] **Step 2: Run the API test and verify RED**

Run: `python -m unittest tests.test_attestation_management_api`

Expected: FAIL with 404 because the routes are absent.

- [ ] **Step 3: Implement request models and routes**

```python
class VisibilityRequest(BaseModel):
    visible: bool

class MoveRequest(BaseModel):
    direction: Literal["up", "down"]
```

Each route calls `require_admin(auth)`, delegates to the storage method, translates missing ids to `attestation_bank_not_found`, and invokes `await runtime.qb.load_published_attestation_banks(runtime.store)` after successful mutation.

- [ ] **Step 4: Run the API test and verify GREEN**

Run: `python -m unittest tests.test_attestation_management_api`

Expected: PASS.

### Task 4: Add the mobile admin management screen

**Files:**
- Modify: `static/js/screens/admin.js:1-70`
- Modify: `static/js/app.js:45-100,250-365`
- Modify: `static/styles/components.css`
- Modify: `static/index.html`
- Test: `tests/test_dynamic_attestation_assets.py`

- [ ] **Step 1: Write failing asset tests**

Assert that the admin hub links to `admin-attestation-banks`, the screen calls `/api/admin/attestation-banks`, and includes controls for visibility, `up`, `down`, and `DELETE` with confirmation.

- [ ] **Step 2: Run the asset test and verify RED**

Run: `python -m unittest tests.test_dynamic_attestation_assets`

Expected: FAIL because the management screen is absent.

- [ ] **Step 3: Implement the admin screen**

Add `renderAdminAttestationBanks(ctx)` and `loadAdminAttestationBanks(ctx)`. Render the protected Stage 1 row first without controls; render every dynamic row with a switch, up/down buttons, question count, current position, and a destructive delete button. After each successful request, reload the admin list and update `ctx.state.bootstrap.catalog.attestation_banks` from a fresh bootstrap reload or page reload.

Use full-width, wrapping action rows below 560px so controls remain tappable on phones. Confirm deletion with the bank title:

```javascript
if (!window.confirm(`Видалити «${bank.title}» разом із питаннями?`)) return;
await ctx.api(`/api/admin/attestation-banks/${bank.id}`, { method: "DELETE" });
```

- [ ] **Step 4: Run the asset test and verify GREEN**

Run: `python -m unittest tests.test_dynamic_attestation_assets`

Expected: PASS.

### Task 5: Verify and publish

**Files:**
- Verify all modified files

- [ ] **Step 1: Run all automated checks**

Run:

```powershell
python -m unittest discover -s tests
python -m py_compile app.py storage.py questions.py admin_apk_import_extension.py attestation_publishing.py
git diff --check
```

Expected: all tests PASS, compilation succeeds, and diff check reports no errors.

- [ ] **Step 2: Commit and push directly to main**

```powershell
git add app.py storage.py questions.py admin_apk_import_extension.py static tests docs
git commit -m "feat: manage attestation sections"
git push origin main
```

- [ ] **Step 3: Verify publication state**

Run: `git status --short --branch` and compare `git rev-parse HEAD` with `git rev-parse origin/main`.

Expected: clean `main`, identical revisions.
