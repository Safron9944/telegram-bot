from __future__ import annotations

from aiogram.filters.callback_data import CallbackData


# -------------------------
# CallbackData
# -------------------------
MULTI_OK_CODE = "__MULTI_OK__"
MULTI_OK_LEVEL = 0

class MultiTopicsPageCb(CallbackData, prefix="mtp"):
    mode: str
    page: int

class MultiTopicToggleCb(CallbackData, prefix="mtt"):
    mode: str
    topic_idx: int
    page: int

class MultiTopicDoneCb(CallbackData, prefix="mtd"):
    mode: str

class MultiTopicClearCb(CallbackData, prefix="mtc"):
    mode: str
    page: int

class MultiTopicAllCb(CallbackData, prefix="mta"):
    mode: str


class AnswerCb(CallbackData, prefix="ans"):
    mode: str   # "train" | "exam"
    qid: int
    ci: int     # choice index

class SkipCb(CallbackData, prefix="sk"):
    qid: int

# продовжити після фідбеку (коли показали правильну відповідь)
class NextCb(CallbackData, prefix="nx"):
    mode: str   # "train" | "exam"
    expected_index: int  # який current_index очікуємо у сесії

class AdminToggleQCb(CallbackData, prefix="qt"):
    qid: int
    enable: int  # 1 enable, 0 disable

# вибір scope
class OkPickCb(CallbackData, prefix="ok"):
    ok_code: str

class OkPageCb(CallbackData, prefix="okp"):
    page: int

class OkMultiPageCb(CallbackData, prefix="okmp"):
    mode: str   # train | exam
    page: int

class OkToggleCb(CallbackData, prefix="okt"):
    mode: str
    ok_code: str
    page: int

class OkDoneCb(CallbackData, prefix="okd"):
    mode: str

class OkClearCb(CallbackData, prefix="okc"):
    mode: str
    page: int

class OkAllCb(CallbackData, prefix="oka"):
    mode: str

class StartMultiOkCb(CallbackData, prefix="stmok"):
    mode: str   # train | exam

class LevelPickCb(CallbackData, prefix="lvl"):
    ok_code: str
    level: int

# старт сесій / вибір тем
class StartScopeCb(CallbackData, prefix="st"):
    mode: str        # train/exam
    ok_code: str
    level: int

class TopicPageCb(CallbackData, prefix="tp"):
    mode: str
    ok_code: str
    level: int
    page: int

class TopicPickCb(CallbackData, prefix="tk"):
    mode: str
    ok_code: str
    level: int
    topic_idx: int

# multi-select topics
class TopicToggleCb(CallbackData, prefix="tt"):
    mode: str
    ok_code: str
    level: int
    topic_idx: int
    page: int

class TopicDoneCb(CallbackData, prefix="td"):
    mode: str
    ok_code: str
    level: int

class TopicClearCb(CallbackData, prefix="tc"):
    mode: str
    ok_code: str
    level: int
    page: int

class TopicAllCb(CallbackData, prefix="ta"):
    mode: str
    ok_code: str
    level: int

class TrainModeCb(CallbackData, prefix="tm"):
    mode: str   # train / exam
    kind: str   # position / manual

class PosMenuCb(CallbackData, prefix="pm"):
    mode: str      # 't' або 'e'
    pid: int       # position id
    action: str    # 'r' | 'b' | 'm'

class PosTopicPageCb(CallbackData, prefix="ptp"):
    mode: str
    pid: int
    page: int

class PosTopicToggleCb(CallbackData, prefix="ptt"):
    mode: str
    pid: int
    topic_idx: int
    page: int

class PosTopicDoneCb(CallbackData, prefix="ptd"):
    mode: str
    pid: int

class PosTopicClearCb(CallbackData, prefix="ptc"):
    mode: str
    pid: int
    page: int

class PosTopicAllCb(CallbackData, prefix="pta"):
    mode: str
    pid: int

class TopicBackCb(CallbackData, prefix="tbk"):
    mode: str
    ok_code: str
    level: int


