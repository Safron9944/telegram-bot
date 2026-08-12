import io
import stat
import unittest
import warnings
import zipfile

from apk_importer.archive import (
    ArchiveInspectionError,
    ArchiveLimits,
    _validate_name,
    inspect_package,
    read_bank,
)


def make_zip(entries, *, stored=False):
    output = io.BytesIO()
    compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, value in entries:
            if isinstance(name, zipfile.ZipInfo):
                archive.writestr(name, value)
            else:
                archive.writestr(name, value)
    return output.getvalue()


class ApkArchiveInspectionTests(unittest.TestCase):
    def test_default_limits_accept_packages_up_to_100_mib(self):
        limits = ArchiveLimits()

        self.assertEqual(100 * 1024 * 1024, limits.upload_bytes)
        self.assertEqual(300 * 1024 * 1024, limits.expanded_bytes)

    def test_finds_all_enc_banks_in_direct_apk(self):
        payload = make_zip(
            [
                ("assets/www/a.enc", b"first"),
                ("assets/www/nested/b.enc", b"second"),
                ("assets/other.enc", b"ignored"),
            ],
            stored=True,
        )

        package = inspect_package(payload, "sample.apk")

        self.assertEqual(
            ["assets/www/a.enc", "assets/www/nested/b.enc"],
            [bank.path for bank in package.banks],
        )
        self.assertEqual(b"first", read_bank(package, package.banks[0].id))

    def test_finds_bank_in_single_nested_base_apk(self):
        base_apk = make_zip([("assets/www/test.enc", b"bank")], stored=True)
        xapk = make_zip([("manifest.json", b"{}"), ("splits/base.apk", base_apk)], stored=True)

        package = inspect_package(xapk, "bundle.xapk")

        self.assertEqual("assets/www/test.enc", package.banks[0].path)
        self.assertEqual(b"bank", read_bank(package, package.banks[0].id))

    def test_rejects_missing_or_ambiguous_base_apk(self):
        missing = make_zip([("manifest.json", b"{}")], stored=True)
        ambiguous = make_zip(
            [("one/base.apk", make_zip([], stored=True)), ("two/base.apk", make_zip([], stored=True))],
            stored=True,
        )

        for payload, code in ((missing, "base_apk_missing"), (ambiguous, "base_apk_ambiguous")):
            with self.subTest(code=code):
                with self.assertRaises(ArchiveInspectionError) as raised:
                    inspect_package(payload, "bundle.apks")
                self.assertEqual(code, raised.exception.code)

    def test_rejects_traversal_absolute_and_backslash_paths(self):
        for path in ("../evil.enc", "/absolute.enc", "assets/www/../../evil.enc"):
            with self.subTest(path=path):
                payload = make_zip([(path, b"bad")], stored=True)
                with self.assertRaises(ArchiveInspectionError) as raised:
                    inspect_package(payload, "sample.apk")
                self.assertEqual("unsafe_archive_path", raised.exception.code)

        with self.assertRaises(ArchiveInspectionError) as raised:
            _validate_name("assets\\www\\evil.enc")
        self.assertEqual("unsafe_archive_path", raised.exception.code)

    def test_rejects_symlinks_and_duplicate_paths(self):
        symlink = zipfile.ZipInfo("assets/www/link.enc")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ArchiveInspectionError) as raised:
            inspect_package(make_zip([(symlink, b"target")], stored=True), "sample.apk")
        self.assertEqual("archive_symlink", raised.exception.code)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate = make_zip(
                [("assets/www/a.enc", b"one"), ("assets/www/a.enc", b"two")],
                stored=True,
            )
        with self.assertRaises(ArchiveInspectionError) as raised:
            inspect_package(duplicate, "sample.apk")
        self.assertEqual("duplicate_archive_path", raised.exception.code)

    def test_enforces_entry_expanded_bank_and_compression_limits(self):
        cases = [
            (
                make_zip([("assets/www/a.enc", b"a"), ("assets/www/b.enc", b"b")], stored=True),
                ArchiveLimits(entries=1),
                "too_many_entries",
            ),
            (
                make_zip([("assets/www/a.enc", b"12345")], stored=True),
                ArchiveLimits(expanded_bytes=4),
                "expanded_size_limit",
            ),
            (
                make_zip([("assets/www/a.enc", b"12345")], stored=True),
                ArchiveLimits(bank_bytes=4),
                "bank_size_limit",
            ),
            (
                make_zip([("assets/www/a.enc", b"0" * 10000)]),
                ArchiveLimits(compression_ratio=5),
                "compression_ratio_limit",
            ),
        ]

        for payload, limits, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(ArchiveInspectionError) as raised:
                    inspect_package(payload, "sample.apk", limits)
                self.assertEqual(code, raised.exception.code)

    def test_rejects_invalid_zip_unsupported_extension_and_missing_banks(self):
        cases = [
            (b"not a zip", "sample.apk", "invalid_zip"),
            (make_zip([], stored=True), "sample.zip", "unsupported_package_type"),
            (make_zip([("assets/www/readme.txt", b"none")], stored=True), "sample.apk", "no_banks_found"),
        ]
        for payload, filename, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(ArchiveInspectionError) as raised:
                    inspect_package(payload, filename)
                self.assertEqual(code, raised.exception.code)


if __name__ == "__main__":
    unittest.main()
