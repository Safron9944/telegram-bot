import unittest

from apk_importer.discovery import extract_bank_titles


class ApkImportDiscoveryTests(unittest.TestCase):
    def test_extracts_friendly_bank_titles_from_decrypted_index(self):
        html = """
        <div id="msat"><a href="#msat" class="button">Перший етап</a></div>
        <div id="msmo"><a href="#msmo" class="button"><b>Митних органів</b></a></div>
        """

        self.assertEqual(
            {
                "testmsat.enc": "Перший етап",
                "testmsmo.enc": "Митних органів",
            },
            extract_bank_titles(html, ["testmsat.enc", "testmsmo.enc"]),
        )


if __name__ == "__main__":
    unittest.main()
