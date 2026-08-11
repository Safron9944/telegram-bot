# Безпечний імпорт питань з APK — дизайн першої фази

## Мета

Додати в адмінську Telegram Mini App автоматичне завантаження APK/XAPK/APKS,
пошук банків `assets/www/*.enc`, вибір одного банку адміністратором,
розшифрування підтримуваного формату, preview питань і завантаження JSON.

Перша фаза не змінює робочу базу питань. Її результатом є стабільний
`ParsedBank`, який у наступній фазі без повторного парсингу підключається до
динамічних етапів атестації та транзакційного імпорту в PostgreSQL.

## Межі першої фази

Входить у scope:

- admin-only завантаження `.apk`, `.xapk` і `.apks`;
- безпечне читання пакетів виключно як ZIP-контейнерів;
- пошук усіх `assets/www/*.enc`;
- список підтримуваних і непідтримуваних банків;
- підтримка відомого `testmsat.enc`;
- розгортання словника/макросів формату TestMS;
- парсинг питання, варіантів, правильної відповіді та маркера `^`;
- сувора валідація, preview, пошук, фільтр, пагінація і JSON-download;
- тимчасове зберігання сесії протягом 30 хвилин;
- synthetic encrypted fixtures та локальна regression-перевірка на `base.apk`.

Не входить у scope:

- запис питань у робочу БД;
- створення або оновлення етапів атестації;
- зміни головного екрана користувача;
- виконання APK, DEX, JavaScript або будь-якого стороннього коду;
- автоматичний підбір невідомих passphrase чи криптографічних схем;
- збереження пояснень/посилань із маркера `*`.

## Архітектура

### Пакет `apk_importer`

- `models.py` — `ArchiveBank`, `ParsedBank`, `ParsedSection`,
  `ParsedQuestion`, validation issues і summary.
- `archive.py` — ліміти ZIP, APK/XAPK/APKS, безпечний пошук `base.apk` та
  `assets/www/*.enc` без розпакування всього архіву.
- `crypto.py` — Base64 prefix repair, OpenSSL salted envelope,
  EVP_BytesToKey MD5, AES-256-CBC, PKCS7 і cp1251.
- `testms.py` — адаптер TestMS: структурна перевірка, макроси, tokenizer і
  формування нормалізованих питань.
- `validation.py` — перевірки банку й питань без залежності від UI/API.
- `sessions.py` — випадкові токени, прив'язка до admin user ID, TTL і cleanup.
- `service.py` — orchestration `inspect → parse → validate → preview/export`.

### Інтеграція застосунку

- `admin_apk_import_extension.py` реєструє admin-only API й не додає логіку
  парсингу до `app.py`.
- `static/js/admin_apk_import.js` реалізує екрани upload, bank selection,
  analysis result, preview і download.
- Існуюча адмін-панель отримує одну нову точку входу «Витягнути питання з APK».
- Робочі `QuestionBank`, `Storage` і поточний перший етап не змінюються.

## Доменний контракт

`ParsedQuestion` містить:

- стабільний source key;
- source question number;
- section/topic;
- question text;
- `choices`;
- 1-based `correct`;
- `correct_texts`;
- `shuffle_choices`.

`ParsedBank` містить:

- adapter code і source filename;
- source/version із заголовка;
- SHA-256 encrypted payload;
- упорядковані sections;
- упорядковані questions;
- validation summary;
- кількість питань із `shuffle_choices = false`.

JSON-download має кореневі поля `source`, `source_version`, `source_hash`,
`count`, `sections` і `questions`. Кожен елемент `questions` містить
`source_key`, `qnum`, `topic`, `question`, `choices`, `correct`, `correct_texts`
і `shuffle_choices`. Поля `id`, `section` і назва майбутнього етапу не виводяться
з діапазону `1500xxx`; наступна фаза призначить DB identity та stage entity
окремо.

Маркери TestMS трактуються так:

- `#` починає питання;
- `+` позначає правильну відповідь із `shuffle_choices = true`;
- `^` позначає правильну відповідь із `shuffle_choices = false`;
- `-` позначає неправильну відповідь;
- `*` розпізнається як службове пояснення і пропускається.

Нерозгорнутий macro reference, невідома структура або неоднозначна правильна
відповідь є критичною помилкою. Парсер не повертає частково успішний JSON.

## Відомий криптографічний адаптер

`testmsat.enc` використовує:

- відновлення Base64-префікса `U2FsdGVkX1`;
- envelope `Salted__` і 8-byte salt;
- EVP_BytesToKey з MD5 до 48 bytes;
- 32-byte AES key і 16-byte IV;
- AES-256-CBC;
- строгий PKCS7;
- cp1251;
- структурний заголовок `testmsat`.

Passphrase читається лише із серверної змінної
`APK_BANK_TESTMSAT_PASSPHRASE`. Вона не повертається API, не потрапляє у frontend,
логи, fixtures або Git. Якщо змінна відсутня чи не підходить, банк має статус
«ключ або формат не підтримується».

У тестовому `base.apk` знайдено чотири банки. Відомим ключем підтверджено лише
`testmsat.enc`; інші банки відображаються, але не парсяться до появи окремих
адаптерів/ключів.

## Безпека архівів

Початкові hard limits:

- upload до 50 MiB;
- не більше 2 000 ZIP entries;
- до 150 MiB сумарного uncompressed size;
- один `.enc` до 10 MiB;
- не більше одного вкладеного APK для XAPK/APKS;
- preview session TTL 30 хвилин.

Імпортер:

- перевіряє ZIP magic/central directory, а не лише extension/MIME;
- відхиляє absolute paths, `..`, backslash traversal, symlinks, encrypted ZIP
  entries, дубльовані/неоднозначні critical paths і підозрілий compression ratio;
- не довіряє filenames і ніколи не використовує їх як filesystem destination;
- не розпаковує пакет цілком;
- для XAPK/APKS приймає єдиний однозначний `base.apk`, інакше повертає помилку;
- використовує server-generated session directory і random URL-safe token;
- прив'язує session до admin user ID;
- видаляє файли при cancel або expiry; download не завершує сесію, щоб адміністратор
  міг повторно переглянути чи завантажити результат до завершення TTL;
- не виконує вміст пакета за жодних умов.

## API

- `POST /api/admin/apk-import/sessions` — multipart upload, archive inspection,
  session token і список банків.
- `POST /api/admin/apk-import/sessions/{token}/banks/{bank_id}/parse` — decrypt,
  parse, validate і summary вибраного банку.
- `GET /api/admin/apk-import/sessions/{token}/questions` — paginated preview з
  section/search filters.
- `GET /api/admin/apk-import/sessions/{token}/download` — UTF-8 JSON attachment.
- `DELETE /api/admin/apk-import/sessions/{token}` — cancel і cleanup.

Кожен endpoint повторно перевіряє admin access та ownership session. API не
повертає filesystem paths, passphrase, plaintext bank або stack traces.

## Адмінський UX

1. Адміністратор відкриває «Витягнути питання з APK».
2. Завантажує файл і бачить progress/cancel.
3. Отримує всі знайдені банки з filename, size і support status.
4. Вибирає один підтримуваний банк.
5. Бачить summary: sections, questions, valid/rejected, no-shuffle count.
6. Переглядає питання з pagination, section filter і search. Правильна відповідь
   виділяється, `shuffle_choices = false` має окремий badge.
7. Завантажує JSON або скасовує сесію.

Кнопки імпорту в БД у першій фазі немає.

## Помилки

Користувацькі помилки мають стабільні codes і короткі українські повідомлення:

- invalid/oversized archive;
- unsafe or ambiguous archive structure;
- no banks found;
- unsupported bank;
- missing/invalid server passphrase;
- decrypt/padding/encoding failure;
- macro/parse/validation failure;
- expired/not-owned session.

Деталі для діагностики логуються без secret, plaintext questions і сирого APK.

## Перевірки

Automated tests покривають:

- prefix repair, EVP_BytesToKey MD5, AES-CBC, PKCS7, cp1251;
- synthetic TestMS dictionary/macros;
- `#`, `+`, `-`, `^`, пропуск `*`;
- `shuffle_choices = false`;
- malformed question і unresolved macro;
- APK scan, XAPK/APKS `base.apk`, traversal, symlink, duplicate path,
  compression/size/count/depth limits;
- session ownership, expiry і cleanup;
- admin-only API;
- preview filters/pagination і JSON-download;
- відсутність змін у `questions`/`QuestionBank`.

Маленький synthetic encrypted bank комітиться як fixture. Реальний
`C:\adb-tools\testms-apk\base.apk` не комітиться; локальна integration-перевірка
має підтвердити 800 питань, 4 × 200, чотири choices, одну correct і наявність
`shuffle_choices = false`.

## Наступна фаза без переписування

Транзакційний DB importer прийматиме той самий `ParsedBank`. Він додасть stage
metadata, diff/confirm і persistence, але не змінюватиме archive, crypto, parser,
validation, preview або JSON contracts першої фази.
