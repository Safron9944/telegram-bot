"""Exercise independent HTTP entry points with real storage and QuestionBank."""
import unittest
from types import SimpleNamespace

from app import (AuthContext, MiniAppService, AnswerRequest, ReviewIndexRequest, StartAttestationRequest,
    api_attestation_start, api_session, api_session_answer, api_session_skip, api_session_next,
    api_test_review_open, api_test_review_index, api_test_review_back, api_mistakes_start)
from commerce import CommerceStore
from questions import QuestionBank, dynamic_attestation_runtime_id
from storage import Storage
from tests.test_commerce import Context, SQLConnection
from utils import now


class RetainedSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.con = SQLConnection()
        self.addCleanup(self.con.db.close)
        self.store = Storage("unused")
        self.store.pool = SimpleNamespace(acquire=lambda:Context(self.con))
        self.ledger = CommerceStore(self.store)
        await self.ledger.initialize(self.con)
        self.runtime = SimpleNamespace(store=self.store,qb=QuestionBank("unused"))
        self.con.db.execute("INSERT INTO attestation_banks(id,slug,title,source_id,status,ever_published,display_order,questions_count) VALUES(7,'withdrawn','Куплений банк','withdrawn','hidden',TRUE,0,3)")
        self.con.db.execute("INSERT INTO attestation_banks(id,slug,title,source_id,status,ever_published,display_order,questions_count) VALUES(8,'new-draft','Нова чернетка','draft','draft',FALSE,1,1)")
        for qid in (501,502,503):
            self.con.db.execute("INSERT INTO published_attestation_questions(id,bank_id,source_key,qnum,topic,question,choices,correct,correct_texts) VALUES(?,7,?,?,'Тема',?,'[\"A\",\"B\"]','[1]','[\"A\"]')",(qid,str(qid),qid-500,f"Питання {qid}"))
        self.con.db.execute("INSERT INTO published_attestation_questions(id,bank_id,topic,question,choices,correct) VALUES(801,8,'Тема','DRAFT SECRET','[\"A\",\"B\"]','[1]')")

    async def buyer(self,key,user_id):
        await self.ledger.record_payment(user_id=user_id,product_key=key,product_title="Куплений доступ",amount=100,currency="XTR",reference=f"buyer-{user_id}",source="telegram",paid_at=now())
        return AuthContext({},await self.store.get_user(user_id),user_id,False)

    async def assert_isolated(self,auth):
        self.assertEqual({},self.runtime.qb.attestation_banks)
        self.assertEqual({},self.runtime.qb.by_id)
        request = MiniAppService(self.runtime)
        await request.prepare_attestation_catalog(auth)
        self.assertIn("withdrawn",request.qb.attestation_banks)
        self.assertNotIn("new-draft",request.qb.attestation_banks)
        self.assertNotIn(dynamic_attestation_runtime_id(801),request.qb.by_id)

    async def test_learning_survives_fresh_answer_feedback_skip_and_resume_requests(self):
        for user_id,key in ((42,"section:attestation:7"),(43,"full")):
            with self.subTest(product_key=key):
                auth = await self.buyer(key,user_id)
                view = await api_attestation_start("withdrawn",StartAttestationRequest(section="Тема",block="1-3"),auth,self.runtime)
                self.assertEqual("question",view["screen"])
                view = await api_session_answer(AnswerRequest(choice=1),auth,self.runtime)
                self.assertEqual("feedback",view["screen"])
                resumed = await api_session(auth,self.runtime)
                self.assertEqual("feedback",resumed["screen"])
                view = await api_session_next(auth,self.runtime)
                self.assertEqual("question",view["screen"])
                view = await api_session_skip(auth,self.runtime)
                self.assertEqual("question",view["screen"])
                view = await api_session_answer(AnswerRequest(choice=0),auth,self.runtime)
                self.assertEqual("question",view["screen"])
                view = await api_session_answer(AnswerRequest(choice=0),auth,self.runtime)
                self.assertEqual("learn_result",view["mode"])
                await self.assert_isolated(auth)

    async def test_completed_test_review_uses_owned_catalog_even_without_saved_bank_meta(self):
        for user_id,key in ((42,"section:attestation:7"),(43,"full")):
            with self.subTest(product_key=key):
                auth = await self.buyer(key,user_id)
                view = await api_attestation_start("withdrawn",StartAttestationRequest(section="",block="combined"),auth,self.runtime)
                self.assertEqual("question",view["screen"])
                for _ in range(3):
                    view = await api_session_answer(AnswerRequest(choice=1),auth,self.runtime)
                self.assertEqual("test_result",view["mode"])
                self.assertNotIn("meta",(await self.store.get_ui(user_id))["state"])
                view = await api_test_review_open(auth,self.runtime)
                self.assertEqual("review",view["screen"])
                self.assertEqual(3,view["total"])
                view = await api_test_review_index(ReviewIndexRequest(index=1),auth,self.runtime)
                self.assertEqual("review",view["screen"])
                self.assertEqual(1,view["index"])
                self.assertEqual("test_result",(await api_test_review_back(auth,self.runtime))["mode"])
                await self.assert_isolated(auth)

    async def test_mistakes_session_loads_retained_questions_on_a_new_request(self):
        auth = await self.buyer("full",42)
        qid = dynamic_attestation_runtime_id(501)
        self.con.db.execute("INSERT INTO errors(user_id,qid,wrong_count,in_mistakes) VALUES(42,?,3,1)",(qid,))
        view = await api_mistakes_start(auth,self.runtime)
        self.assertEqual("question",view["screen"])
        self.assertEqual(qid,view["question"]["id"])
        view = await api_session_answer(AnswerRequest(choice=0),auth,self.runtime)
        self.assertEqual("mistakes_result",view["mode"])
        await self.assert_isolated(auth)
