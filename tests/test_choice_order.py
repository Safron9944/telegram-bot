import unittest

from choice_order import ensure_choice_order, normalized_choice_order, ordered_choice_indices


class ChoiceOrderTests(unittest.TestCase):
    def test_creates_and_reuses_one_order_per_question(self):
        state = {}

        def reverse(values):
            values.reverse()

        first, created = ensure_choice_order(state, 42, 4, shuffle=reverse)
        second, created_again = ensure_choice_order(state, 42, 4, shuffle=lambda values: None)

        self.assertEqual([3, 2, 1, 0], first)
        self.assertTrue(created)
        self.assertEqual(first, second)
        self.assertFalse(created_again)
        self.assertEqual({"42": first}, state["choice_orders"])

    def test_invalid_saved_order_is_replaced(self):
        state = {"choice_orders": {"7": [0, 0, 1, 2]}}

        order, created = ensure_choice_order(state, 7, 4, shuffle=lambda values: values.reverse())

        self.assertEqual([3, 2, 1, 0], order)
        self.assertTrue(created)

    def test_new_session_never_keeps_the_source_order(self):
        state = {}

        order, created = ensure_choice_order(state, 9, 4, shuffle=lambda values: None)

        self.assertEqual([1, 2, 3, 0], order)
        self.assertTrue(created)

    def test_order_validation_and_fallback(self):
        self.assertEqual([2, 0, 1], normalized_choice_order([2, 0, 1], 3))
        self.assertIsNone(normalized_choice_order([2, 2, 1], 3))
        self.assertEqual([0, 1, 2], ordered_choice_indices(3, [2, 2, 1]))


if __name__ == "__main__":
    unittest.main()
