# Telegram Test Bot (aiogram) + PostgreSQL (Railway)

Це стартовий репозиторій Telegram-бота для тестування:
- реєстрація через номер телефону (кнопка "Поділитися номером")
- "кабінет" користувача (статус підписки, trial 3 дні)
- адмін-команди: видати/забрати підписку
- база даних PostgreSQL (статистика та дані користувачів)
- healthcheck сервер `/healthz` (щоб Railway не валив деплой)

## 1) Локальний запуск

1. Створіть `.env` (можете скопіювати з `.env.example`):
2. Встановіть залежності:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Запустіть:
   ```bash
   python -m app
   ```

## 2) Деплой на Railway через GitHub

1. Залийте код в GitHub (не комітьте токени!).
2. Railway → New Project → Deploy from GitHub Repo.
3. Додайте PostgreSQL (Add → Database → PostgreSQL).
4. В Service (бота) → Variables:
   - `TELEGRAM_BOT_TOKEN` = токен від @BotFather
   - `ADMIN_TG_IDS` = наприклад `123456789,987654321`
   - `DATABASE_URL` = `${{ Postgres.DATABASE_URL }}` (або виберіть зі списку)
   - (опційно) `PORT` = 8080 (якщо хочете фіксований порт)
5. Settings → Healthcheck path: `/healthz`
6. Settings → Start Command: `python -m app`

Після деплою відкрийте бота в Telegram і натисніть `/start`.

## 3) Імпорт ваших питань/блоків

У репозиторії є файли:
- `data/questions_flat.json` (плоский список питань)
- `data/questions_by_block.json` (та сама база, згрупована по блоках/темах)

Після запуску адміністратор може виконати:
- `/seed` — імпортує блоки та питання з `data/questions_flat.json` у PostgreSQL (повторний запуск пропускає вже існуючі питання)

## 4) Структура

- `app/main.py` — точка входу
- `app/db/*` — БД (SQLAlchemy async)
- `app/handlers/*` — хендлери команд/меню
- `app/services/*` — бізнес-логіка (підписки, тести)
