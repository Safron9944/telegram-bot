from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import (
    AnswerRequest,
    AttestationReviewPatch,
    AuthContext,
    MiniAppService,
    StartAttestationRequest,
)
from attestation import AttestationBank, AttestationQuestion, SECTION_KEYS
from utils import now


def auth(tier="none", admin=False):
    user = {}
    if tier == "full":
        user = {"sub_tier": "full", "sub_infinite": True}
    elif tier == "trial_full":
        user = {"trial_end": now() + timedelta(days=1)}
    return AuthContext({}, user, 7, admin)


def question(section="constitution", number=1, qid=1):
    return AttestationQuestion(
        id=qid,
        section=section,
        section_title=SECTION_KEYS[section],
        qnum=number,
        question=f"Питання {section} {number}?",
        choices=["Варіант А", "Варіант Б"],
        correct=[2],
        source_page=3,
        source_hash=f"hash-{section}-{number}",
        verification_method="pdf_visual",
    )


class MemoryStore:
    def __init__(self):
        self.state = {}
        self.save_test = AsyncMock()

    async def get_ui(self, user_id):
        return {"state": self.state}

    async def set_state(self, user_id, state):
        self.state = state

    async def stats(self, user_id):
        return {"count": 0, "avg": 0.0, "last": None}

    async def get_setting(self, key, default=None):
        return default


@pytest.fixture
def fake_runtime():
    questions = []
    qid = 1
    for section in SECTION_KEYS:
        for number in range(1, 121):
            questions.append(question(section=section, number=number, qid=qid))
            qid += 1
    return SimpleNamespace(
        store=MemoryStore(),
        qb=SimpleNamespace(by_id={}, law_groups={}, ok_modules={}, law=[]),
        attestation_qb=AttestationBank(questions),
        admin_ids=set(),
        bot_token="",
        webapp_url="",
        allow_dev_auth=True,
        auth_max_age_seconds=0,
    )


def completed_attestation_state(section="constitution", mode="demo"):
    return {
        "mode": "attestation",
        "pending": [],
        "correct_count": 8,
        "total": 10,
        "answers": {str(number): number <= 8 for number in range(1, 11)},
        "chosen": {str(number): 0 for number in range(1, 11)},
        "started_at": now().isoformat(),
        "meta": {
            "bank": "attestation",
            "section": section,
            "selection_mode": mode,
            "access": "demo",
        },
    }


def test_catalog_exposes_four_sections_and_all(fake_runtime):
    service = MiniAppService(fake_runtime)

    catalog = service.serialize_attestation_catalog(auth())

    assert [item["key"] for item in catalog["sections"]] == [
        "constitution",
        "civil_service",
        "customs_code",
        "anti_corruption",
        "all",
    ]
    assert catalog["access"] == "demo"
    assert catalog["sections"][0]["demo_count"] == 10
    assert catalog["sections"][-1]["demo_count"] == 40


@pytest.mark.asyncio
async def test_demo_cannot_request_random_or_part(fake_runtime):
    service = MiniAppService(fake_runtime)

    with pytest.raises(Exception) as exc:
        await service.start_attestation(
            auth(),
            StartAttestationRequest(section="constitution", mode="random"),
        )

    assert exc.value.status_code == 403


def test_trial_also_receives_only_demo(fake_runtime):
    service = MiniAppService(fake_runtime)

    catalog = service.serialize_attestation_catalog(auth("trial_full"))

    assert catalog["access"] == "demo"
    with pytest.raises(Exception) as exc:
        service.select_attestation(
            auth("trial_full"), "constitution", "part", 1
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_full_and_admin_may_request_full_modes(fake_runtime):
    service = MiniAppService(fake_runtime)

    assert len(
        service.select_attestation(auth("full"), "constitution", "random", 1)
    ) == 50
    assert len(
        service.select_attestation(auth(admin=True), "constitution", "part", 1)
    ) == 50


@pytest.mark.asyncio
async def test_attestation_answer_always_returns_feedback(fake_runtime):
    service = MiniAppService(fake_runtime)
    await service.start_attestation(
        auth(),
        StartAttestationRequest(section="constitution", mode="demo"),
    )

    view = await service.answer(auth(), AnswerRequest(choice=0))

    assert view["mode"] == "attestation"
    assert view["screen"] == "feedback"
    assert any(
        option["status"] == "correct" for option in view["question"]["options"]
    )


@pytest.mark.asyncio
async def test_attestation_finish_uses_separate_stats(fake_runtime):
    service = MiniAppService(fake_runtime)
    fake_runtime.store.save_test.reset_mock()
    state = completed_attestation_state(section="all", mode="demo")

    result = await service.finish_attestation(7, state)

    assert result["mode"] == "attestation_result"
    fake_runtime.store.save_test.assert_awaited_once()
    assert (
        fake_runtime.store.save_test.await_args.kwargs["test_type"]
        == "attestation"
    )


@pytest.mark.asyncio
async def test_saved_full_session_is_rejected_after_access_expires(fake_runtime):
    service = MiniAppService(fake_runtime)
    fake_runtime.store.state = {
        "mode": "attestation",
        "pending": [1],
        "current_qid": 1,
        "meta": {"bank": "attestation", "access": "full"},
    }

    with pytest.raises(Exception) as exc:
        await service.answer(auth(), AnswerRequest(choice=0))

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_bootstrap_contains_attestation_catalog(fake_runtime):
    result = await MiniAppService(fake_runtime).bootstrap(auth())

    assert result["catalog"]["attestation"]["access"] == "demo"


@pytest.mark.asyncio
async def test_empty_attestation_bank_has_explicit_error(fake_runtime):
    fake_runtime.attestation_qb = AttestationBank()
    service = MiniAppService(fake_runtime)

    with pytest.raises(Exception) as exc:
        await service.start_attestation(
            auth(),
            StartAttestationRequest(section="constitution", mode="demo"),
        )

    assert exc.value.detail["code"] == "attestation_bank_empty"


@pytest.mark.asyncio
async def test_admin_approval_reloads_runtime_bank(fake_runtime):
    fake_runtime.store.approve_attestation_review = AsyncMock(
        return_value={"status": "approved"}
    )
    fake_runtime.store.attestation_review_summary = AsyncMock(
        return_value={"verified": 481, "needs_review": 2}
    )
    existing = list(fake_runtime.attestation_qb.by_id.values())
    new_question = question(section="constitution", number=121, qid=9999)
    fake_runtime.store.fetch_attestation_questions = AsyncMock(
        return_value=[item.__dict__ for item in [*existing, new_question]]
    )
    service = MiniAppService(fake_runtime)
    before = len(fake_runtime.attestation_qb.by_id)

    result = await service.admin_approve_attestation_review(
        auth(admin=True),
        9,
        AttestationReviewPatch(
            question="Повне питання?",
            choices=["А", "Б"],
            correct=[2],
        ),
    )

    assert len(fake_runtime.attestation_qb.by_id) == before + 1
    assert result["review"]["status"] == "approved"


@pytest.mark.asyncio
async def test_non_admin_cannot_use_attestation_review_methods(fake_runtime):
    service = MiniAppService(fake_runtime)

    calls = [
        lambda: service.admin_attestation_reviews(auth(), "needs_review", 0, 20),
        lambda: service.admin_attestation_review(auth(), 1),
        lambda: service.admin_attestation_summary(auth()),
        lambda: service.admin_reject_attestation_review(auth(), 1),
        lambda: service.admin_approve_attestation_review(
            auth(),
            1,
            AttestationReviewPatch(
                question="Питання?", choices=["А", "Б"], correct=[1]
            ),
        ),
    ]
    for call in calls:
        with pytest.raises(Exception) as exc:
            await call()
        assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_invalid_admin_approval_returns_field_safe_400(fake_runtime):
    fake_runtime.store.approve_attestation_review = AsyncMock(
        side_effect=ValueError("порожній варіант відповіді")
    )
    service = MiniAppService(fake_runtime)

    with pytest.raises(Exception) as exc:
        await service.admin_approve_attestation_review(
            auth(admin=True),
            9,
            AttestationReviewPatch(
                question="Питання?", choices=["А", ""], correct=[1]
            ),
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_attestation_review"
