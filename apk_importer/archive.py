from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from .models import ArchiveBank


class ArchiveInspectionError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ArchiveLimits:
    upload_bytes: int = 100 * 1024 * 1024
    entries: int = 2000
    expanded_bytes: int = 300 * 1024 * 1024
    bank_bytes: int = 10 * 1024 * 1024
    compression_ratio: int = 200


@dataclass(frozen=True)
class InspectedPackage:
    filename: str
    apk_payload: bytes
    banks: tuple[ArchiveBank, ...]
    limits: ArchiveLimits


def _fail(code: str, message: str) -> None:
    raise ArchiveInspectionError(code, message)


def _validate_name(name: str) -> str:
    if "\\" in name or name.startswith(("/", "\\")):
        _fail("unsafe_archive_path", "Архів містить небезпечний шлях.")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        _fail("unsafe_archive_path", "Архів містить небезпечний шлях.")
    return path.as_posix()


def _validated_infos(archive: zipfile.ZipFile, limits: ArchiveLimits) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > limits.entries:
        _fail("too_many_entries", "Архів містить забагато файлів.")
    if sum(info.file_size for info in infos) > limits.expanded_bytes:
        _fail("expanded_size_limit", "Розпакований архів перевищує дозволений розмір.")

    seen: set[str] = set()
    for info in infos:
        normalized = _validate_name(info.filename).casefold()
        if normalized in seen:
            _fail("duplicate_archive_path", "Архів містить дубльований шлях.")
        seen.add(normalized)
        if info.flag_bits & 0x1:
            _fail("encrypted_zip_entry", "ZIP містить зашифрований запис.")
        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            _fail("archive_symlink", "ZIP містить символічне посилання.")
        if not info.is_dir() and info.file_size:
            if info.compress_size == 0 or info.file_size / info.compress_size > limits.compression_ratio:
                _fail("compression_ratio_limit", "ZIP має підозрілий коефіцієнт стиснення.")
    return infos


def _open_validated(payload: bytes, limits: ArchiveLimits) -> tuple[zipfile.ZipFile, list[zipfile.ZipInfo]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
        return archive, _validated_infos(archive, limits)
    except ArchiveInspectionError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as exc:
        _fail("invalid_zip", "Файл не є коректним ZIP-пакетом.")
        raise AssertionError from exc


def _read_info(archive: zipfile.ZipFile, info: zipfile.ZipInfo, maximum: int, code: str) -> bytes:
    if info.file_size > maximum:
        _fail(code, "Файл усередині архіву перевищує дозволений розмір.")
    with archive.open(info, "r") as source:
        payload = source.read(maximum + 1)
    if len(payload) > maximum:
        _fail(code, "Файл усередині архіву перевищує дозволений розмір.")
    return payload


def inspect_package(
    payload: bytes,
    filename: str,
    limits: ArchiveLimits = ArchiveLimits(),
) -> InspectedPackage:
    raw = bytes(payload)
    if len(raw) > limits.upload_bytes:
        _fail("upload_size_limit", "Завантажений пакет перевищує 100 MiB.")

    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.casefold()
    if suffix not in {".apk", ".xapk", ".apks"}:
        _fail("unsupported_package_type", "Підтримуються лише APK, XAPK та APKS.")

    outer, outer_infos = _open_validated(raw, limits)
    try:
        apk_payload = raw
        if suffix in {".xapk", ".apks"}:
            candidates = [
                info for info in outer_infos
                if not info.is_dir() and PurePosixPath(info.filename).name.casefold() == "base.apk"
            ]
            if not candidates:
                _fail("base_apk_missing", "У пакеті не знайдено base.apk.")
            if len(candidates) != 1:
                _fail("base_apk_ambiguous", "У пакеті знайдено кілька base.apk.")
            apk_payload = _read_info(outer, candidates[0], limits.upload_bytes, "nested_apk_size_limit")
    finally:
        outer.close()

    apk, infos = _open_validated(apk_payload, limits)
    try:
        bank_infos = [
            info for info in infos
            if not info.is_dir()
            and info.filename.casefold().startswith("assets/www/")
            and info.filename.casefold().endswith(".enc")
        ]
        if not bank_infos:
            _fail("no_banks_found", "У пакеті не знайдено банків assets/www/*.enc.")
        banks = []
        for info in sorted(bank_infos, key=lambda item: item.filename.casefold()):
            if info.file_size > limits.bank_bytes:
                _fail("bank_size_limit", "Банк перевищує дозволений розмір.")
            path = PurePosixPath(info.filename).as_posix()
            banks.append(
                ArchiveBank(
                    id=hashlib.sha256(path.encode("utf-8")).hexdigest()[:24],
                    path=path,
                    filename=PurePosixPath(path).name,
                    size=info.file_size,
                )
            )
    finally:
        apk.close()

    return InspectedPackage(filename=filename, apk_payload=apk_payload, banks=tuple(banks), limits=limits)


def read_bank(package: InspectedPackage, bank_id: str) -> bytes:
    selected = next((bank for bank in package.banks if bank.id == bank_id), None)
    if selected is None:
        _fail("bank_not_found", "Банк не знайдено в сесії.")
    archive, infos = _open_validated(package.apk_payload, package.limits)
    try:
        info = next((item for item in infos if item.filename == selected.path), None)
        if info is None:
            _fail("bank_not_found", "Банк не знайдено в пакеті.")
        return _read_info(archive, info, package.limits.bank_bytes, "bank_size_limit")
    finally:
        archive.close()
