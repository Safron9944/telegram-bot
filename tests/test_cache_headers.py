import unittest

from cache_policy import cache_headers


class CacheHeadersTests(unittest.TestCase):
    def test_index_is_never_cached(self):
        self.assertIn("no-store", cache_headers("/")["Cache-Control"])

    def test_versioned_static_assets_are_cached_immutably(self):
        self.assertEqual(
            "public, max-age=31536000, immutable",
            cache_headers("/static/app.js", versioned=True)["Cache-Control"],
        )

    def test_unversioned_static_assets_must_be_revalidated(self):
        self.assertEqual(
            "public, max-age=0, must-revalidate",
            cache_headers("/static/app.js")["Cache-Control"],
        )

    def test_api_responses_are_never_cached(self):
        self.assertEqual(
            "private, no-store",
            cache_headers("/api/bootstrap")["Cache-Control"],
        )

    def test_unrelated_routes_do_not_get_cache_headers(self):
        self.assertEqual({}, cache_headers("/healthz"))


if __name__ == "__main__":
    unittest.main()
