import unittest
from datetime import timedelta

from access import access_status, access_tier, has_attestation_access
from utils import now


class AccessTierTests(unittest.TestCase):
    def test_infinite_cases_tier_stays_limited(self):
        user = {"sub_infinite": True, "sub_tier": "cases"}
        self.assertEqual("cases", access_tier(user))
        self.assertEqual((True, "sub_cases"), access_status(user))
        self.assertTrue(has_attestation_access(user))

    def test_infinite_full_tier_has_full_access(self):
        user = {"sub_infinite": True, "sub_tier": "full"}
        self.assertEqual("full", access_tier(user))
        self.assertEqual((True, "sub_infinite"), access_status(user))

    def test_trial_does_not_include_attestation(self):
        user = {"trial_end": now() + timedelta(days=3)}
        self.assertEqual("trial_full", access_tier(user))
        self.assertFalse(has_attestation_access(user))

    def test_expired_user_has_no_access(self):
        user = {"trial_end": now() - timedelta(seconds=1)}
        self.assertEqual("none", access_tier(user))
        self.assertEqual((False, "expired"), access_status(user))


if __name__ == "__main__":
    unittest.main()
