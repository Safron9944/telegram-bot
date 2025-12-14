import re
import json
import argparse
import pdfplumber
from collections import defaultdict

# Верхні розділи (в PDF)
SECTION_LAW = "Тестування на знання законодавства"
SECTION_OPS = "Тестування рівня операційних митних компетененцій"
# У деяких версіях PDF слово "компетенцій" може бути без помилки.
SECTION_OPS_ALT = "Тестування рівня операційних митних компетенцій"

SECTION_TITLES = {SECTION_LAW, SECTION_OPS, SECTION_OPS_ALT}

# Старт питання: "123 Текст питання..."
QSTART_RE = re.compile(r"^(\d{1,5})\s+(.+)$")

# Старт теми всередині блоку (у вас це виглядає як "1. Питання ...")
TOPIC_RE = re.compile(r"^\d+\.\s+Питання")

# Старт ОК-блоку: "[ОК-4] ..." (використовується як БЛОК, а не як тема)
OK_RE = re.compile(r"^\[(ОК-\d+)\]\s*(.*)$")

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def group_words_to_lines(words, y_tol=2.0):
    """
    words: list[dict] з полями text,x0,top,fontname
    Повертає list[lines], де line має: top, x0, text, words
    """
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    cur = []
    cur_top = None

    for w in words:
        if cur_top is None or abs(w["top"] - cur_top) > y_tol:
            if cur:
                lines.append(cur)
            cur = [w]
            cur_top = w["top"]
        else:
            cur.append(w)

    if cur:
        lines.append(cur)

    out = []
    for ws in lines:
        ws = sorted(ws, key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws).strip()
        if not text:
            continue
        out.append({
            "top": ws[0]["top"],
            "x0": ws[0]["x0"],
            "text": text,
            "words": ws
        })
    return out

def is_page_number(text: str) -> bool:
    return re.fullmatch(r"\d{1,4}", text) is not None

def is_topic_start(text: str) -> bool:
    # ВАЖЛИВО: тут лишаємо тільки теми типу "1. Питання ..."
    # [ОК-..] обробляємо окремо як "block"
    return bool(TOPIC_RE.match(text))

def has_bold(words) -> bool:
    return any("Bold" in (w.get("fontname") or "") for w in words)

def parse_pdf(pdf_path: str, start_page: int, end_page: int | None, progress_every: int):
    questions = []

    current_section = None      # SECTION_LAW або SECTION_OPS/ALT
    current_block = None        # Для LAW = SECTION_LAW, для OPS = "ОК-4"/"ОК-10"/...
    current_topic = None

    pending_topic = None  # якщо заголовок теми переноситься на 2 рядки
    current_q = None      # {number, question, options:[{text,bold}], page_start}
    mode = None           # "q" або "opts"
    qid = 0

    def finalize_question():
        nonlocal qid, current_q
        if not current_q:
            return
        qid += 1
        q = {
            "id": qid,
            "block": current_block,      # <-- тут тепер або SECTION_LAW, або "ОК-xx"
            "topic": current_topic,
            "page_start": current_q["page_start"],
            "number": current_q["number"],
            "question": norm_space(current_q["question"]),
            "choices": [norm_space(o["text"]) for o in current_q["options"]],
            "correct": [i for i, o in enumerate(current_q["options"]) if o["bold"]],
        }
        questions.append(q)
        current_q = None

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if start_page < 1:
            start_page = 1
        if end_page is None or end_page > total_pages:
            end_page = total_pages

        print(f"PDF pages: {total_pages}. Parsing pages {start_page}..{end_page}")

        for page_i in range(start_page, end_page + 1):
            page = pdf.pages[page_i - 1]

            # Важливо: keep_blank_chars=True щоб витягнути "маркери" варіантів (вони йдуть як SymbolMT + ' ')
            words_all = page.extract_words(extra_attrs=["fontname"], keep_blank_chars=True)

            # Маркери варіантів: SymbolMT + пробіл. Беремо top і округляємо до int — цього досить стабільно.
            bullet_set = {
                int(round(w["top"]))
                for w in words_all
                if w.get("fontname") == "SymbolMT" and w.get("text") == " "
            }

            # Текстові слова (без пустих і без SymbolMT)
            words = [
                w for w in words_all
                if w.get("text", "").strip() != "" and w.get("fontname") != "SymbolMT"
            ]

            lines = group_words_to_lines(words)

            for ln in lines:
                t = ln["text"].strip()
                if not t or is_page_number(t):
                    continue

                # 1) Верхні секції (LAW / OPS)
                if t in SECTION_TITLES:
                    finalize_question()
                    current_section = t
                    current_topic = None
                    pending_topic = None

                    if t == SECTION_LAW:
                        current_block = SECTION_LAW
                    else:
                        # Для OPS блок задається тільки коли зустрінемо [ОК-..]
                        current_block = None
                    mode = None
                    continue

                is_bullet = int(round(ln["top"])) in bullet_set

                # 2) Старт ОК-блоку (тільки якщо ми в секції OPS)
                if current_section in (SECTION_OPS, SECTION_OPS_ALT):
                    m_ok = OK_RE.match(t)
                    if m_ok:
                        finalize_question()
                        ok_code = m_ok.group(1)  # "ОК-4"
                        current_block = ok_code  # <-- ключове: ОК стає "block"
                        # як тему на старті залишимо повний заголовок, щоб не було "Без теми"
                        pending_topic = t
                        current_topic = norm_space(t)
                        mode = None
                        continue

                is_topic = is_topic_start(t)

                # 3) Чи це старт питання? (важливо: не bullet-рядок і x0 “лівіше”, щоб не ловити цифри в варіантах)
                m_q = QSTART_RE.match(t)
                is_qstart = (m_q is not None) and (not is_bullet) and (ln["x0"] < 200)

                # Якщо ми збираємо заголовок теми, який перенісся на наступний рядок
                if pending_topic is not None:
                    if (not is_topic) and (not is_qstart) and (not is_bullet):
                        pending_topic = pending_topic + " " + t
                        current_topic = norm_space(pending_topic)
                        continue
                    else:
                        pending_topic = None  # далі обробляємо цей рядок як звичайно

                # 4) Старт теми (всередині поточного block)
                if is_topic:
                    finalize_question()
                    pending_topic = t
                    current_topic = norm_space(t)
                    mode = None
                    continue

                # 5) Старт питання
                if is_qstart:
                    # Якщо ще нема current_block (наприклад, в OPS до першого [ОК-..]) — пропустимо
                    if current_block is None:
                        continue
                    finalize_question()
                    num = int(m_q.group(1))
                    qtext = m_q.group(2).strip()
                    current_q = {
                        "page_start": page_i,
                        "number": num,
                        "question": qtext,
                        "options": []
                    }
                    mode = "q"
                    continue

                # Якщо ще нема активного питання — ігноруємо рядок
                if current_q is None:
                    continue

                # 6) Варіанти
                if is_bullet:
                    current_q["options"].append({
                        "text": t,
                        "bold": has_bold(ln["words"])
                    })
                    mode = "opts"
                else:
                    # перенос рядків у питанні/варіанті
                    if mode == "q":
                        current_q["question"] += " " + t
                    elif mode == "opts" and current_q["options"]:
                        current_q["options"][-1]["text"] += " " + t
                        if has_bold(ln["words"]):
                            current_q["options"][-1]["bold"] = True
                    else:
                        current_q["question"] += " " + t

            if page_i == start_page or page_i % progress_every == 0:
                print(f"Processed page {page_i}, questions so far: {len(questions)}")

    finalize_question()
    print(f"DONE. Total questions: {len(questions)}")
    return questions

def build_nested(questions):
    nested = defaultdict(lambda: defaultdict(list))
    for q in questions:
        b = q.get("block") or "Без блоку"
        t = q.get("topic") or "Без теми"
        nested[b][t].append(q)
    return {b: dict(topics) for b, topics in nested.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default="questions.pdf")
    ap.add_argument("--out_flat", default="questions_flat.json")
    ap.add_argument("--out_nested", default="questions_by_block.json")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--progress", type=int, default=20)
    args = ap.parse_args()

    qs = parse_pdf(args.pdf, args.start, args.end, args.progress)

    with open(args.out_flat, "w", encoding="utf-8") as f:
        json.dump(qs, f, ensure_ascii=False, indent=2)

    nested = build_nested(qs)
    with open(args.out_nested, "w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out_flat}")
    print(f"Saved: {args.out_nested}")
    if qs:
        print("Example:")
        print(json.dumps(qs[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
