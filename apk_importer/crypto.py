from __future__ import annotations

import base64
import binascii
import hashlib

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


OPENSSL_PREFIX = b"U2FsdGVkX1"


class BankDecryptError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def evp_bytes_to_key_md5(password: bytes, salt: bytes) -> tuple[bytes, bytes]:
    if len(salt) != 8:
        raise ValueError("OpenSSL salt must contain 8 bytes")
    material = b""
    digest = b""
    while len(material) < 48:
        digest = hashlib.md5(digest + password + salt).digest()
        material += digest
    return material[:32], material[32:48]


def repair_openssl_prefix(payload: bytes) -> bytes:
    compact = b"".join(bytes(payload).split())
    return compact if compact.startswith(OPENSSL_PREFIX) else OPENSSL_PREFIX + compact


def decrypt_testms_payload(
    payload: bytes,
    passphrase: str,
    expected_header: str = "testmsat",
) -> str:
    try:
        blob = base64.b64decode(repair_openssl_prefix(payload), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BankDecryptError("invalid_base64", "Банк має некоректне Base64-кодування.") from exc

    if len(blob) < 32 or blob[:8] != b"Salted__":
        raise BankDecryptError("invalid_openssl_envelope", "Банк не має підтримуваного OpenSSL-заголовка.")
    ciphertext = blob[16:]
    if not ciphertext or len(ciphertext) % 16:
        raise BankDecryptError("invalid_ciphertext", "Зашифровані дані мають некоректний розмір.")

    try:
        key, iv = evp_bytes_to_key_md5(passphrase.encode("utf-8"), blob[8:16])
        decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        padding = padded[-1]
        if not 1 <= padding <= 16 or padded[-padding:] != bytes([padding]) * padding:
            raise ValueError("invalid padding")
        plaintext = padded[:-padding].decode("cp1251")
    except Exception as exc:
        raise BankDecryptError("decrypt_failed", "Не вдалося розшифрувати банк підтримуваним ключем.") from exc

    first_line = plaintext.splitlines()[0].strip() if plaintext.splitlines() else ""
    if not first_line.startswith(f"{expected_header} "):
        raise BankDecryptError("unexpected_bank_header", "Розшифрований банк має неочікуваний формат.")
    return plaintext
