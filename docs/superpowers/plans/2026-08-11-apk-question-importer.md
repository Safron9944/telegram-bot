# APK Question Importer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an admin-only Mini App workflow that safely scans APK/XAPK/APKS files, lists encrypted banks, parses supported TestMS banks, previews normalized questions, and downloads JSON without changing the working question database.

**Architecture:** A standalone `apk_importer` package owns archive inspection, cryptography, TestMS parsing, validation, temporary sessions, and orchestration. A thin FastAPI extension and a self-contained admin JavaScript module expose that domain model without coupling it to the existing stage-1 `QuestionBank` implementation.

**Tech Stack:** Python 3.14, FastAPI, `cryptography`, standard-library `zipfile`, vanilla JavaScript, `unittest`

---

### Task 1: Domain models and strict validation

**Files:**
- Create: `apk_importer/__init__.py`
- Create: `apk_importer/models.py`
- Create: `apk_importer/validation.py`
- Create: `tests/test_apk_import_models.py`

- [ ] **Step 1: Write failing model and validation tests**

Define tests that construct the following public objects and assert stable serialization:

```python
question = ParsedQuestion(
    source_key="constitution:1",
    qnum=1,
    topic="Конституція України",
    question="Питання?",
    choices=("A", "B", "C", "D"),
    correct=(3,),
    correct_texts=("C",),
    shuffle_choices=False,
)
bank = ParsedBank(
    adapter="testms",
    source="testmsat.enc",
    source_version="3",
    source_hash="a" * 64,
    sections=(ParsedSection("Конституція України", 0, 1),),
    questions=(question,),
)
validate_bank(bank)
self.assertEqual(1, bank.to_dict()["count"])
```

Also assert that empty text, fewer than two choices, duplicate choices, invalid correct indexes, mismatched `correct_texts`, duplicate `source_key`, and a question referencing an absent section raise `BankValidationError` with stable issue codes.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m unittest tests.test_apk_import_models -v`

Expected: import failure because `apk_importer.models` does not exist.

- [ ] **Step 3: Implement immutable domain models and validation**

Create dataclasses with the exact fields above plus `ArchiveBank`, `ValidationIssue`, and `BankSummary`. Implement `ParsedBank.to_dict()` with root keys `source`, `source_version`, `source_hash`, `count`, `sections`, and `questions`; do not emit DB IDs or a stage title.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_apk_import_models -v`

Expected: all model/validation tests pass.

### Task 2: OpenSSL/CryptoJS decryption adapter

**Files:**
- Create: `apk_importer/crypto.py`
- Create: `tests/test_apk_import_crypto.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing cryptography tests with a generated fixture**

Tests generate deterministic OpenSSL-compatible ciphertext from a tiny cp1251 payload and assert:

```python
key, iv = evp_bytes_to_key_md5(b"secret", bytes.fromhex("0102030405060708"))
self.assertEqual(32, len(key))
self.assertEqual(16, len(iv))
self.assertEqual("testmsat 3\n$слово\n#I. 1. Питання?", decrypt_testms_payload(trimmed, "secret"))
```

Add separate failures for missing `Salted__`, invalid Base64, wrong passphrase/PKCS7, invalid cp1251 structure, and missing `testmsat` header.

- [ ] **Step 2: Run the crypto tests and confirm they fail**

Run: `python -m unittest tests.test_apk_import_crypto -v`

Expected: import failure because `apk_importer.crypto` does not exist.

- [ ] **Step 3: Implement the minimal crypto API**

Implement the public callables `evp_bytes_to_key_md5(password: bytes, salt: bytes)
-> tuple[bytes, bytes]`, `repair_openssl_prefix(payload: bytes) -> bytes`, and
`decrypt_testms_payload(payload: bytes, passphrase: str, expected_header: str =
"testmsat") -> str`.

Add `cryptography>=46.0,<47.0` to `requirements.txt`. Never log passphrase, plaintext, key, IV, or decrypted payload.

- [ ] **Step 4: Run the crypto tests**

Run: `python -m unittest tests.test_apk_import_crypto -v`

Expected: all crypto tests pass.

### Task 3: Safe APK/XAPK/APKS archive inspection

**Files:**
- Create: `apk_importer/archive.py`
- Create: `tests/test_apk_import_archive.py`

- [ ] **Step 1: Write failing in-memory ZIP tests**

Create tiny archives with `io.BytesIO` and assert:

```python
result = inspect_package(apk_bytes, "sample.apk")
self.assertEqual(["assets/www/a.enc"], [bank.path for bank in result.banks])
```

Cover direct APK, XAPK/APKS containing one `base.apk`, absent/ambiguous `base.apk`, `../`, absolute/backslash paths, duplicate critical paths, symlink entries, encrypted flags, entry-count limit, expanded-size limit, `.enc` size limit, suspicious compression ratio, invalid ZIP, and archives without banks.

- [ ] **Step 2: Run the archive tests and confirm they fail**

Run: `python -m unittest tests.test_apk_import_archive -v`

Expected: import failure because `apk_importer.archive` does not exist.

- [ ] **Step 3: Implement bounded archive inspection**

Implement this configuration exactly:

```python
@dataclass(frozen=True)
class ArchiveLimits:
    upload_bytes: int = 50 * 1024 * 1024
    entries: int = 2000
    expanded_bytes: int = 150 * 1024 * 1024
    bank_bytes: int = 10 * 1024 * 1024
    compression_ratio: int = 200

```

Expose `inspect_package(payload: bytes, filename: str, limits: ArchiveLimits =
ArchiveLimits()) -> InspectedPackage` and `read_bank(package: InspectedPackage,
bank_id: str) -> bytes`. Read entries through `ZipFile.open()` with byte counters;
never call `extract()` or derive a filesystem path from an archive name.

- [ ] **Step 4: Run the archive tests**

Run: `python -m unittest tests.test_apk_import_archive -v`

Expected: all archive security tests pass.

### Task 4: TestMS dictionary/macro parser

**Files:**
- Create: `apk_importer/testms.py`
- Create: `tests/fixtures/testms_plaintext_small.txt`
- Create: `tests/test_testms_bank_parser.py`

- [ ] **Step 1: Write failing parser tests from a small synthetic grammar**

The fixture contains a header, `$`-separated dictionary, section marker, two questions, `+`, `-`, `^`, and `*`. Tests assert macro expansion before tokenization, explanation omission, one shuffled question, one non-shuffled question, stable source keys, and an error for an unresolved macro.

```python
bank = parse_testms_bank(fixture_text, source="testmsat.enc", source_hash="a" * 64)
self.assertEqual(2, len(bank.questions))
self.assertTrue(bank.questions[0].shuffle_choices)
self.assertFalse(bank.questions[1].shuffle_choices)
self.assertNotIn("explanation", bank.to_dict()["questions"][0])
```

- [ ] **Step 2: Run the parser tests and confirm they fail**

Run: `python -m unittest tests.test_testms_bank_parser -v`

Expected: import failure because `apk_importer.testms` does not exist.

- [ ] **Step 3: Implement header/dictionary decoding and question tokenization**

Implement a deterministic two-pass parser with public functions
`expand_testms_macros(text: str) -> str` and `parse_testms_bank(text: str, *,
source: str, source_hash: str) -> ParsedBank`.

Pass one reads the version and dictionary and resolves every macro reference; pass two recognizes section/question/answer records. `*` records are consumed and discarded. An unresolved macro, malformed record, zero/multiple correct answers, or fewer than two choices raises `TestMsParseError`; no partial bank is returned.

- [ ] **Step 4: Run parser and domain tests**

Run: `python -m unittest tests.test_testms_bank_parser tests.test_apk_import_models -v`

Expected: all tests pass.

### Task 5: Adapter registry, service, and expiring sessions

**Files:**
- Create: `apk_importer/service.py`
- Create: `apk_importer/sessions.py`
- Create: `tests/test_apk_import_service.py`
- Create: `tests/test_apk_import_sessions.py`

- [ ] **Step 1: Write failing service/session tests**

Assert that upload creates an owner-bound 30-minute token, all banks are listed, only `testmsat.enc` is supported when its environment passphrase exists, unsupported banks cannot parse, parsed results support section/search pagination, JSON bytes are UTF-8, another admin cannot access the session, and cancel/expiry removes the temporary directory.

```python
session = service.create_session(admin_id=7, filename="base.apk", payload=apk)
self.assertEqual(4, len(session.banks))
parsed = service.parse_bank(7, session.token, supported_bank_id)
self.assertEqual(2, parsed.summary.questions_count)
with self.assertRaises(SessionAccessError):
    service.get_session(8, session.token)
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m unittest tests.test_apk_import_service tests.test_apk_import_sessions -v`

Expected: imports fail because service/session modules do not exist.

- [ ] **Step 3: Implement adapter registry and filesystem session store**

Use server-generated directories under `tempfile.gettempdir()/telegram-bot-apk-import`, `secrets.token_urlsafe(32)`, an owner/expiry metadata file, atomic writes, and cleanup on every public operation. Persist only the uploaded archive and normalized parsed JSON; never persist passphrases or decrypted plaintext.

- [ ] **Step 4: Run service/session tests**

Run: `python -m unittest tests.test_apk_import_service tests.test_apk_import_sessions -v`

Expected: all tests pass.

### Task 6: Admin-only FastAPI endpoints

**Files:**
- Create: `admin_apk_import_extension.py`
- Modify: `sitecustomize.py`
- Create: `tests/test_admin_apk_import.py`

- [ ] **Step 1: Write failing API contract and authorization tests**

Register the extension on a small FastAPI test app with fake auth dependencies. Assert 403 for non-admin upload and session access; 200 contracts for upload/list, parse, preview, download, and delete; 404/410 for unknown/expired tokens; JSON download headers; and bounded multipart reads that reject payloads over 50 MiB.

- [ ] **Step 2: Run the API tests and confirm they fail**

Run: `python -m unittest tests.test_admin_apk_import -v`

Expected: import failure because `admin_apk_import_extension` does not exist.

- [ ] **Step 3: Implement thin API routes**

Routes call one `ApkImportService` stored on `app.state`, translate domain exceptions to stable HTTP `{code,message}` details, and repeat `auth.is_admin` plus session ownership checks on every request. Import the extension from `sitecustomize.py` beside the existing admin extension.

- [ ] **Step 4: Run API and existing authorization tests**

Run: `python -m unittest tests.test_admin_apk_import tests.test_admin_controls tests.test_access -v`

Expected: all tests pass.

### Task 7: Admin Mini App workflow

**Files:**
- Create: `static/js/admin_apk_import.js`
- Modify: `static/index.html`
- Modify: `static/js/screens/admin.js`
- Modify: `static/styles/components.css`
- Create: `tests/test_admin_apk_import_assets.py`

- [ ] **Step 1: Write failing static integration tests**

Assert the HTML loads the versioned module, the admin screen exposes one entry point, and the module contains the five API paths and required accessible labels. These tests prevent an unreferenced/dead frontend module.

- [ ] **Step 2: Run static tests and confirm they fail**

Run: `python -m unittest tests.test_admin_apk_import_assets -v`

Expected: failures because the module and entry point are absent.

- [ ] **Step 3: Implement the workflow**

Implement upload progress/cancel, bank cards with support status, parse action, summary cards, section/search filters, paginated preview, correct-answer highlighting, no-shuffle badge, JSON download, cancel cleanup, Telegram haptics, escaped content, and Ukrainian error messages. Do not add an import-to-DB button.

- [ ] **Step 4: Run static integration tests**

Run: `python -m unittest tests.test_admin_apk_import_assets -v`

Expected: all static integration tests pass.

### Task 8: Real APK regression, full verification, and publication

**Files:**
- Create: `scripts/check_apk_question_import.py`
- Modify: `docs/superpowers/plans/2026-08-11-apk-question-importer.md`

- [ ] **Step 1: Add a local-only integration checker**

The script accepts `--apk` and reads `APK_BANK_TESTMSAT_PASSPHRASE` from the environment. It prints only counts/status and exits nonzero unless the fixture yields 4 banks, a supported `testmsat.enc`, 800 questions, 4 sections × 200, valid answer mappings, and at least one `shuffle_choices = false`. It never prints plaintext or secrets.

- [ ] **Step 2: Run the real APK regression**

Run with the passphrase supplied only in the process environment:

`python scripts/check_apk_question_import.py --apk C:\adb-tools\testms-apk\base.apk`

Expected: `PASS banks=4 questions=800 sections=4 no_shuffle=2`.

- [ ] **Step 3: Run full verification**

Run:

```text
python -m unittest discover -s tests -v
python -m compileall -q apk_importer admin_apk_import_extension.py
git diff --check
git status --short
```

Expected: all tests pass, compile succeeds, diff check is clean, and only planned files are changed.

- [ ] **Step 4: Commit, push, merge, and verify main**

Commit with a conventional message, push `codex/apk-question-importer`, merge it into current `main` only after updating from `origin/main`, rerun the complete tests on merged `main`, and push `origin/main` without force.
