import json
import unittest
from datetime import date

from attestation_export_security import (
    build_export_document,
    decode_download_token,
    encode_download_token,
)


class AttestationExportTokenTests(unittest.TestCase):
    def test_export_document_has_expected_name_and_utf8_json(self):
        payload = {
            "section": "Атестація посадових осіб — 1 етап",
            "count": 1,
            "questions": [{"question": "Тестове питання"}],
        }

        file_name, content = build_export_document(
            payload,
            export_date=date(2026, 8, 11),
        )

        self.assertEqual("attestation_stage_1_current_2026-08-11.json", file_name)
        self.assertEqual(payload, json.loads(content.decode("utf-8")))

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
