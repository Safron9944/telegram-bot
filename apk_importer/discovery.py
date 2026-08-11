from __future__ import annotations

import base64
from dataclasses import dataclass
import html
import io
import re
import struct
import zipfile

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


@dataclass(frozen=True)
class BankProfile:
    title: str
    passphrase: str
    expected_header: str


BANK_PROFILES = {
    "testmsat.enc": BankProfile("Перший етап", "yYR4XEef3MugI3jb", "testmsat"),
    "testmsmo.enc": BankProfile("Митних органів", "A6KPIz8Rci2ZF3sy", "testmsmo"),
    "testmsto.enc": BankProfile("Територіальних органів", "ERT2penZTQaDh6mQ", "testmsto"),
    "testmsca.enc": BankProfile("Центрального апарату", "jYfYZHqt6TuKUaB5", "testms"),
}


def extract_bank_titles(document: str, filenames) -> dict[str, str]:
    result = {}
    for filename in filenames:
        page = filename.rsplit(".", 1)[0].casefold().removeprefix("test")
        match = re.search(
            rf"<a\b[^>]*href=[\"']#{re.escape(page)}[\"'][^>]*>(.*?)</a>",
            document,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            title = re.sub(r"<[^>]+>", " ", match.group(1))
            title = " ".join(html.unescape(title).split())
            if title:
                result[filename] = title
    return result


def _read_uleb128(data: bytes, position: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, position
        shift += 7
        if shift > 35:
            raise ValueError("invalid DEX ULEB128")


def _dex_strings(data: bytes) -> list[str]:
    if len(data) < 0x70 or not data.startswith(b"dex\n"):
        raise ValueError("invalid DEX")
    count, table = struct.unpack_from("<II", data, 0x38)
    if count > 500_000 or table + count * 4 > len(data):
        raise ValueError("invalid DEX string table")
    strings = []
    for index in range(count):
        position = struct.unpack_from("<I", data, table + index * 4)[0]
        _, position = _read_uleb128(data, position)
        end = data.find(b"\0", position)
        if end < 0:
            raise ValueError("unterminated DEX string")
        try:
            strings.append(data[position:end].decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return strings


def _resource_cipher(dex: bytes, encrypted_index: bytes) -> tuple[bytes, bytes]:
    strings = _dex_strings(dex)
    keys = [item.encode() for item in strings if item.isascii() and len(item) in (16, 24, 32)]
    ivs = {item.encode() for item in strings if item.isascii() and len(item) == 16}
    first_block = base64.b64decode(encrypted_index, validate=True)[:16]
    expected = b"<!DOCTYPE html>\n"
    if len(first_block) != 16:
        raise ValueError("encrypted index is too short")
    for key in keys:
        decrypted = Cipher(algorithms.AES(key), modes.ECB()).decryptor().update(first_block)
        needed_iv = bytes(left ^ right for left, right in zip(decrypted, expected))
        if needed_iv in ivs:
            return key, needed_iv
    raise ValueError("APK resource cipher was not found")


def _decrypt_resource(payload: bytes, key: bytes, iv: bytes) -> str:
    encrypted = base64.b64decode(payload, validate=True)
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(encrypted) + decryptor.finalize()
    padding = padded[-1]
    if not 1 <= padding <= 16 or padded[-padding:] != bytes([padding]) * padding:
        raise ValueError("invalid APK resource padding")
    return padded[:-padding].decode("utf-8")


def discover_bank_titles(apk_payload: bytes, filenames) -> dict[str, str]:
    fallback = {
        filename: BANK_PROFILES.get(filename.casefold(), BankProfile(filename, "", "")).title
        for filename in filenames
    }
    try:
        with zipfile.ZipFile(io.BytesIO(apk_payload)) as archive:
            dex = archive.read("classes.dex")
            encrypted_index = archive.read("assets/www/index.html")
        key, iv = _resource_cipher(dex, encrypted_index)
        fallback.update(extract_bank_titles(_decrypt_resource(encrypted_index, key, iv), filenames))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        pass
    return fallback
