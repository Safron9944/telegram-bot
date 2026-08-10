import unittest

from attestation_export_security import (
    decode_download_token,
    encode_download_token,
)


class AttestationExportTokenTests(unittest.TestCase):
    def test_token_round_trip(self):
        token = encode_download_token(12345, "test-bot-token", expires_at=2_000)

        self.assertEqual(
            (12345, 2_000),
            decode_download_token(
                token,
                "test-bot-token",
                now_timestamp=1_999,
            ),
        )

    def test_tampered_token_is_rejected(self):
        token = encode_download_token(12345, "test-bot-token", expires_at=2_000)
        payload, signature = token.split(".", 1)
        tampered = f"{payload}.{signature[:-1]}A"

        self.assertIsNone(
            decode_download_token(
                tampered,
                "test-bot-token",
                now_timestamp=1_999,
            )
        )

    def test_expired_token_is_rejected(self):
        token = encode_download_token(12345, "test-bot-token", expires_at=2_000)

        self.assertIsNone(
            decode_download_token(
                token,
                "test-bot-token",
                now_timestamp=2_001,
            )
        )

    def test_malformed_token_is_rejected(self):
        self.assertIsNone(
            decode_download_token(
                "not-a-valid-token",
                "test-bot-token",
                now_timestamp=1_999,
            )
        )


if __name__ == "__main__":
    unittest.main()
