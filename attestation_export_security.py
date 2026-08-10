from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time


DOWNLOAD_TOKEN_TTL_SECONDS = 5 * 60
_DOWNLOAD_TOKEN_PURPOSE = b"attestation-stage-1-export"


def _token_key(bot_token: str) -> bytes:
    return hmac.new(
        bot_token.encode("utf-8"),
        _DOWNLOAD_TOKEN_PURPOSE,
        hashlib.sha256,
    ).digest()


def encode_download_token(
    user_id: int,
    bot_token: str,
    *,
    expires_at: int | None = None,
) -> str:
    expires_at = int(expires_at or (time.time() + DOWNLOAD_TOKEN_TTL_SECONDS))
    payload = f"{int(user_id)}:{expires_at}".encode("ascii")
    encoded_payload = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    signature = hmac.new(_token_key(bot_token), payload, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded_payload}.{encoded_signature}"


def decode_download_token(
    token: str,
    bot_token: str,
    *,
    now_timestamp: int | None = None,
) -> tuple[int, int] | None:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
        signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        expected = hmac.new(_token_key(bot_token), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        raw_user_id, raw_expires_at = payload.decode("ascii").split(":", 1)
        user_id = int(raw_user_id)
        expires_at = int(raw_expires_at)
    except (binascii.Error, TypeError, ValueError, UnicodeDecodeError):
        return None

    current_timestamp = int(now_timestamp if now_timestamp is not None else time.time())
    if expires_at < current_timestamp:
        return None
    return user_id, expires_at
