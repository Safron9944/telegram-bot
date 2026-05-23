from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Q:
    id: int
    section: str
    topic: str
    ok: Optional[str]
    level: Optional[int]
    qnum: Optional[int]
    question: str
    choices: List[str]
    correct: List[int]
    correct_texts: List[str]

    @property
    def is_valid_mcq(self) -> bool:
        return bool(self.choices) and isinstance(self.correct, list) and len(self.correct) > 0


class QuestionBank:
    def __init__(self, path: str):
        self.path = path
        self.by_id: Dict[int, Q] = {}
        self.law: List[int] = []
        self.law_groups: Dict[str, List[int]] = {}
        self.ok_modules: Dict[str, Dict[int, List[int]]] = {}
        self._law_group_titles: Dict[str, str] = {}

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.by_id.clear()
        self.law.clear()
        self.law_groups.clear()
        self.ok_modules.clear()
        self._law_group_titles.clear()

        for item in self._iter_raw_questions(raw):
            norm = self._normalize_item(item)
            if not norm:
                continue
            q = Q(
                id=norm["id"], section=norm.get("section", ""), topic=norm.get("topic", ""),
                ok=norm.get("ok"), level=norm.get("level"), qnum=norm.get("qnum"),
                question=norm.get("question", ""), choices=norm.get("choices", []),
                correct=norm.get("correct", []), correct_texts=norm.get("correct_texts", []),
            )
            if q.id in self.by_id:
                nid = q.id
                while nid in self.by_id:
                    nid += 1
                q.id = nid
            self.by_id[q.id] = q

        self._build_indexes()

    async def load_from_db(self, store: "Any"):
        rows = await store.fetch_questions()

        self.by_id.clear()
        self.law.clear()
        self.law_groups.clear()
        self.ok_modules.clear()
        self._law_group_titles.clear()

        def _norm_json(v: Any):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return v
            return v

        for r in rows:
            rid = int(r.get("id"))
            q = Q(
                id=rid, section=(r.get("section") or ""), topic=(r.get("topic") or ""),
                ok=r.get("ok"), level=r.get("level"), qnum=r.get("qnum"),
                question=(r.get("question") or ""),
                choices=_norm_json(r.get("choices")) or [],
                correct=_norm_json(r.get("correct")) or [],
                correct_texts=_norm_json(r.get("correct_texts")) or [],
            )
            self.by_id[q.id] = q

        self._build_indexes()

    def _build_indexes(self):
        for qid, q in self.by_id.items():
            if not q.is_valid_mcq:
                continue
            sec = (q.section or "").lower()
            is_ok = bool(q.ok) or ("операцій" in sec and "компет" in sec)
            if not is_ok:
                self.law.append(qid)
                key = self._law_group_key(q.topic or q.section)
                self.law_groups.setdefault(key, []).append(qid)
            if is_ok:
                mod = q.ok or "ОК"
                self.ok_modules.setdefault(mod, {})
                lvl = int(q.level or 1)
                self.ok_modules[mod].setdefault(lvl, []).append(qid)

        def _ord_key(qid: int):
            qq = self.by_id.get(qid)
            n = getattr(qq, "qnum", None)
            return (n if isinstance(n, int) else 10 ** 9, int(qid))

        for k in self.law_groups:
            self.law_groups[k].sort(key=_ord_key)
        self.law.sort(key=_ord_key)
        for ok in self.ok_modules:
            for lvl in self.ok_modules[ok]:
                self.ok_modules[ok][lvl].sort(key=_ord_key)

    def law_group_title(self, key: str) -> str:
        if key in self._law_group_titles:
            return self._law_group_titles[key]
        if key.isdigit():
            for qid in self.law_groups.get(key, []):
                t = (self.by_id[qid].topic or "").strip()
                if t.startswith(f"{key}."):
                    return t.split(".", 1)[1].strip()
        return key

    def pick_random(self, qids: List[int], n: int) -> List[int]:
        if len(qids) <= n:
            return list(qids)
        return random.sample(qids, n)

    def _iter_raw_questions(self, raw: Any):
        if isinstance(raw, list):
            for it in raw:
                if isinstance(it, dict):
                    yield it
            return
        if not isinstance(raw, dict):
            return
        qlist = raw.get("questions") or raw.get("items")
        if isinstance(qlist, list):
            for it in qlist:
                if isinstance(it, dict):
                    yield it
        law = raw.get("law") or raw.get("laws") or raw.get("legislation")
        if isinstance(law, list):
            for it in law:
                if isinstance(it, dict):
                    it = dict(it)
                    it.setdefault("section", "Законодавство")
                    yield it
        if isinstance(law, dict):
            for law_title, arr in law.items():
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    it = dict(it)
                    it.setdefault("section", "Законодавство")
                    it.setdefault("topic", str(law_title))
                    yield it
        ok = (raw.get("ok") or raw.get("ok_questions") or raw.get("ok_modules")
              or raw.get("operational_competencies") or raw.get("operationalCompetencies"))
        if isinstance(ok, list):
            for it in ok:
                if isinstance(it, dict):
                    it = dict(it)
                    it.setdefault("section", "ОК")
                    yield it
        if isinstance(ok, dict):
            for module_name, v in ok.items():
                if isinstance(v, dict) and isinstance(v.get("levels"), dict):
                    module_title = v.get("name") or v.get("title") or v.get("module_name") or ""
                    levels_dict = v.get("levels") or {}
                    for lvl, arr in levels_dict.items():
                        if not isinstance(arr, list):
                            continue
                        for it in arr:
                            if not isinstance(it, dict):
                                continue
                            it = dict(it)
                            it.setdefault("section", "ОК")
                            it.setdefault("ok", str(module_name))
                            it.setdefault("level", lvl)
                            if module_title:
                                it.setdefault("topic", str(module_title))
                            yield it
                    continue
                if isinstance(v, dict):
                    for lvl, arr in v.items():
                        if not isinstance(arr, list):
                            continue
                        for it in arr:
                            if not isinstance(it, dict):
                                continue
                            it = dict(it)
                            it.setdefault("section", "ОК")
                            it.setdefault("ok", str(module_name))
                            it.setdefault("level", lvl)
                            yield it
                elif isinstance(v, list):
                    for it in v:
                        if not isinstance(it, dict):
                            continue
                        it = dict(it)
                        it.setdefault("section", "ОК")
                        it.setdefault("ok", str(module_name))
                        yield it
        secs = raw.get("sections")
        if isinstance(secs, list):
            for sec in secs:
                if not isinstance(sec, dict):
                    continue
                sec_name = sec.get("name") or sec.get("title") or sec.get("section") or "Секція"
                sec_q = sec.get("questions") or sec.get("items")
                if isinstance(sec_q, list):
                    for it in sec_q:
                        if not isinstance(it, dict):
                            continue
                        it = dict(it)
                        it.setdefault("section", str(sec_name))
                        yield it
                topics = sec.get("topics")
                if isinstance(topics, list):
                    for tp in topics:
                        if not isinstance(tp, dict):
                            continue
                        topic_name = tp.get("name") or tp.get("title") or tp.get("topic") or ""
                        tp_q = tp.get("questions") or tp.get("items")
                        if isinstance(tp_q, list):
                            for it in tp_q:
                                if not isinstance(it, dict):
                                    continue
                                it = dict(it)
                                it.setdefault("section", str(sec_name))
                                it.setdefault("topic", str(topic_name))
                                yield it

    def _normalize_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        qtext = (item.get("question_text") or item.get("questionText") or item.get("question")
                 or item.get("q") or item.get("text") or item.get("title") or "")
        qtext = str(qtext).strip()
        if not qtext:
            return None
        section = str(item.get("section") or item.get("category") or item.get("type") or "").strip()
        topic = str(item.get("topic") or item.get("group") or item.get("chapter") or "").strip()
        if not topic:
            topic = section
        ok = item.get("ok") or item.get("module") or item.get("ok_module") or item.get("okModule")
        ok = str(ok).strip() if ok is not None else None
        if ok == "":
            ok = None
        ok_code = item.get("ok_code") or item.get("okCode")
        ok_name = item.get("ok_name") or item.get("okName")
        if ok is None:
            if ok_code is not None and str(ok_code).strip():
                ok = str(ok_code).strip()
            elif ok_name is not None and str(ok_name).strip():
                ok = str(ok_name).strip()
        lvl_raw = item.get("level") or item.get("lvl") or item.get("difficulty") or item.get("diff")
        level: Optional[int] = None
        try:
            if lvl_raw is not None and str(lvl_raw).strip() != "":
                level = int(str(lvl_raw).strip())
        except Exception:
            level = None
        qnum_raw = (item.get("question_number") or item.get("questionNumber")
                    or item.get("number") or item.get("num") or item.get("no"))
        qnum: Optional[int] = None
        try:
            if qnum_raw is not None and str(qnum_raw).strip() != "":
                qnum = int(str(qnum_raw).strip())
        except Exception:
            qnum = None
        choices_raw = (item.get("choices") or item.get("options") or item.get("answers")
                       or item.get("variants") or item.get("variants_list") or [])
        if isinstance(choices_raw, dict):
            choices_raw = list(choices_raw.values())
        choices: List[str] = []
        inferred_correct: List[int] = []
        if isinstance(choices_raw, list) and choices_raw and all(isinstance(x, dict) for x in choices_raw):
            for d in choices_raw:
                t = str(d.get("text") or d.get("answer") or d.get("value") or d.get("title") or "").strip()
                if not t:
                    continue
                choices.append(t)
                flag = d.get("is_correct")
                if isinstance(flag, bool) and flag:
                    inferred_correct.append(len(choices))
        elif isinstance(choices_raw, list):
            for ch in choices_raw:
                txt = str(ch).strip()
                if txt:
                    choices.append(txt)
        if not choices:
            return None
        correct: List[int] = []
        if inferred_correct:
            correct = sorted(set(inferred_correct))
        if not correct:
            idx0 = (item.get("correct_answer_index") or item.get("correctAnswerIndex")
                    or item.get("correct_index") or item.get("correctIndex"))
            try:
                if idx0 is not None and str(idx0).strip() != "":
                    i0 = int(str(idx0).strip())
                    if 0 <= i0 < len(choices):
                        correct = [i0 + 1]
            except Exception:
                pass
        if not correct:
            idxs0 = item.get("correct_answer_indices") or item.get("correctAnswerIndices")
            if isinstance(idxs0, list):
                tmp: List[int] = []
                for v in idxs0:
                    try:
                        i0 = int(v)
                        if 0 <= i0 < len(choices):
                            tmp.append(i0 + 1)
                    except Exception:
                        continue
                if tmp:
                    correct = sorted(set(tmp))
        if not correct:
            correct_raw = (item.get("correct") or item.get("correct_answers") or item.get("correctAnswers")
                           or item.get("right") or item.get("right_answers") or item.get("answer"))
            if isinstance(correct_raw, list) and correct_raw and all(isinstance(x, bool) for x in correct_raw):
                correct = [i + 1 for i, flag in enumerate(correct_raw) if flag]
            else:
                nums: List[int] = []
                if isinstance(correct_raw, int):
                    nums = [correct_raw]
                elif isinstance(correct_raw, str):
                    nums = [int(x) for x in re.findall(r"\d+", correct_raw)]
                elif isinstance(correct_raw, list):
                    for x in correct_raw:
                        if isinstance(x, int):
                            nums.append(x)
                        elif isinstance(x, str) and x.strip().isdigit():
                            nums.append(int(x.strip()))
                nums = [n for n in nums if 1 <= n <= len(choices)]
                if nums:
                    correct = sorted(set(nums))
        correct_texts = item.get("correct_texts") or item.get("correctTexts") or []
        if isinstance(correct_texts, str):
            correct_texts = [correct_texts]
        if isinstance(correct_texts, list):
            correct_texts = [str(x).strip() for x in correct_texts if str(x).strip()]
        else:
            correct_texts = []
        if not correct_texts and correct:
            correct_texts = [choices[i - 1] for i in correct if 1 <= i <= len(choices)]
        raw_id = item.get("id") or item.get("uid") or item.get("qid") or item.get("question_id")
        qid = self._make_int_id(raw_id, fallback=(raw_id, section, topic, qnum, qtext))
        return {
            "id": int(qid), "section": section, "topic": topic, "ok": ok,
            "level": level, "qnum": qnum, "question": qtext,
            "choices": choices, "correct": correct, "correct_texts": correct_texts,
        }

    def _law_group_key(self, topic: str) -> str:
        topic = (topic or "").strip()
        if len(topic) >= 2 and topic[0].isdigit() and topic[1] == ".":
            return topic.split(".", 1)[0].strip()
        if not topic:
            topic = "Законодавство"
        key = "t" + hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]
        self._law_group_titles.setdefault(key, topic)
        return key

    def _make_int_id(self, raw_id: Any, fallback: Any) -> int:
        try:
            if raw_id is not None and str(raw_id).strip().lstrip("-").isdigit():
                v = int(str(raw_id).strip())
                if -2147483648 <= v <= 2147483647:
                    return v
        except Exception:
            pass
        s = json.dumps(fallback, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha1(s.encode("utf-8")).digest()
        v = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        return v or 1
