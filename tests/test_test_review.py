from types import SimpleNamespace
import unittest

from app import MiniAppService
from questions import Q, QuestionBank


class TestReviewViewTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self):
        bank = QuestionBank("unused.json")
        bank.by_id[42] = Q(
            42,
            "Атестація",
            "Тема",
            None,
            None,
            1,
            "Тестове питання?",
            ["Варіант A", "Варіант B"],
            [2],
            ["Варіант B"],
            shuffle_choices=False,
        )
        return MiniAppService(SimpleNamespace(qb=bank, store=None))

    async def test_saved_first_choice_is_not_marked_missing(self):
        service = self.make_service()
        state = {
            "wrong_qids": [42],
            "chosen": {"42": 0},
            "choice_orders": {"42": [0, 1]},
            "review_index": 0,
        }

        view = await service.build_test_review_view(1, state)

        self.assertEqual("review", view["screen"])
        self.assertFalse(view["question"]["selected_missing"])

    async def test_missing_choice_is_marked_missing(self):
        service = self.make_service()
        state = {
            "wrong_qids": [42],
            "chosen": {},
            "choice_orders": {"42": [0, 1]},
            "review_index": 0,
        }

        view = await service.build_test_review_view(1, state)

        self.assertTrue(view["question"]["selected_missing"])


if __name__ == "__main__":
    unittest.main()
