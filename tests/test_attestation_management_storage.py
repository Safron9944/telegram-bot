import unittest

from storage import Storage


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.rows = [
            {"id": 7, "slug": "second", "source_id": "a.enc", "display_order": 0},
            {"id": 8, "slug": "third", "source_id": "b.enc", "display_order": 1},
        ]
        self.executemany_calls = []

    def transaction(self):
        return FakeTransaction()

    async def fetch(self, query, *args):
        if "FOR UPDATE" in query:
            return list(self.rows)
        return [{**row, "title": row["slug"], "status": "published", "questions_count": 10} for row in self.rows]

    async def fetchrow(self, query, *args):
        if query.lstrip().startswith("UPDATE") and args[0] == 7:
            return {**self.rows[0], "title": "second", "status": "hidden", "questions_count": 10}
        if query.lstrip().startswith("DELETE") and args[0] == 7:
            return {"id": 7}
        return None

    async def executemany(self, query, args):
        self.executemany_calls.append((query, list(args)))


class Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return Acquire(self.connection)


class AttestationManagementStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = FakeConnection()
        self.store = Storage("postgresql://unused")
        self.store.pool = FakePool(self.connection)

    async def test_lists_hidden_and_published_banks_for_admin(self):
        rows = await self.store.list_attestation_banks_for_admin()
        self.assertEqual([7, 8], [row["id"] for row in rows])

    async def test_visibility_move_and_delete_are_scoped_to_dynamic_bank(self):
        hidden = await self.store.set_attestation_bank_visibility(7, visible=False)
        moved = await self.store.move_attestation_bank(8, direction="up")
        deleted = await self.store.delete_attestation_bank(7)

        self.assertEqual("hidden", hidden["status"])
        self.assertTrue(moved)
        self.assertTrue(deleted)
        self.assertEqual([(8, 0), (7, 1)], self.connection.executemany_calls[0][1])


if __name__ == "__main__":
    unittest.main()
