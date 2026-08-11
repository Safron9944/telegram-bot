import base64
import hashlib
import unittest

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from apk_importer.crypto import (
    BankDecryptError,
    decrypt_testms_payload,
    evp_bytes_to_key_md5,
    repair_openssl_prefix,
)


def encrypt_trimmed(plaintext: str, passphrase: str, salt: bytes = b"12345678") -> bytes:
    key, iv = evp_bytes_to_key_md5(passphrase.encode("utf-8"), salt)
    raw = plaintext.encode("cp1251")
    pad = 16 - len(raw) % 16
    padded = raw + bytes([pad]) * pad
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    blob = b"Salted__" + salt + encryptor.update(padded) + encryptor.finalize()
    encoded = base64.b64encode(blob)
    self_prefix = b"U2FsdGVkX1"
    if not encoded.startswith(self_prefix):
        raise AssertionError("Unexpected OpenSSL Base64 prefix")
    return encoded[len(self_prefix) :]


class ApkImportCryptoTests(unittest.TestCase):
    def test_evp_bytes_to_key_matches_reference_digest_chain(self):
        salt = bytes.fromhex("0102030405060708")
        key, iv = evp_bytes_to_key_md5(b"secret", salt)

        first = hashlib.md5(b"secret" + salt).digest()
        second = hashlib.md5(first + b"secret" + salt).digest()
        third = hashlib.md5(second + b"secret" + salt).digest()

        self.assertEqual((first + second)[:32], key)
        self.assertEqual(third, iv)

    def test_repairs_removed_openssl_base64_prefix(self):
        self.assertEqual(b"U2FsdGVkX19hYmM=", repair_openssl_prefix(b"9hYmM="))
        self.assertEqual(b"U2FsdGVkX19hYmM=", repair_openssl_prefix(b"U2FsdGVkX19hYmM="))

    def test_decrypts_cryptojs_compatible_cp1251_payload(self):
        plaintext = "testmsat 3\n$слово\n#I. 1. Питання?"
        encrypted = encrypt_trimmed(plaintext, "secret")

        self.assertEqual(plaintext, decrypt_testms_payload(encrypted, "secret"))

    def test_rejects_wrong_passphrase_without_exposing_details(self):
        encrypted = encrypt_trimmed("testmsat 3\n#I. 1. Питання?", "secret")

        with self.assertRaises(BankDecryptError) as raised:
            decrypt_testms_payload(encrypted, "wrong")

        self.assertEqual("decrypt_failed", raised.exception.code)
        self.assertNotIn("wrong", str(raised.exception))

    def test_rejects_invalid_base64_and_non_openssl_payload(self):
        for payload in (b"%%%", base64.b64encode(b"not-salted")):
            with self.subTest(payload=payload):
                with self.assertRaises(BankDecryptError):
                    decrypt_testms_payload(payload, "secret")

    def test_rejects_valid_padding_when_header_is_wrong(self):
        encrypted = encrypt_trimmed("otherbank 1\n#I. 1. Питання?", "secret")

        with self.assertRaises(BankDecryptError) as raised:
            decrypt_testms_payload(encrypted, "secret")

        self.assertEqual("unexpected_bank_header", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
