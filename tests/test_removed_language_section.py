import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import app as application
from app import AuthContext, MiniAppService, StartAttestationRequest
from questions import QuestionBank
from sections import build_sections, free_section_keys, get_section
from utils import dt_to_iso, now


class RemovedLanguageSectionTests(unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        return SimpleNamespace(
            init=AsyncMock(),
            fetch_questions=AsyncMock(return_value=[{
                'id': 101, 'section': 'Митні компетенції', 'topic': 'Закон',
                'question': 'Питання?', 'choices': ['А', 'Б'],
                'correct': [1], 'correct_texts': ['А'],
            }]),
            list_published_attestation_banks=AsyncMock(return_value=[]),
            list_attestation_banks_for_admin=AsyncMock(return_value=[]),
            get_setting=AsyncMock(return_value=json.dumps({
                'ukrainian_language': {'visible': True, 'price': 0},
            })),
            get_user=AsyncMock(return_value={
                'user_id': 42, 'section_access': ['ukrainian_language'],
                'section_access_overrides': {'ukrainian_language': True},
            }),
            stats=AsyncMock(return_value={}),
            pool=None,
        )

    async def test_old_grants_and_configuration_do_not_restore_the_section(self):
        store = self.make_store()
        user = await store.get_user(42)
        for is_admin in (False, True):
            with self.subTest(is_admin=is_admin):
                sections = await build_sections(store, user, is_admin=is_admin)
                self.assertNotIn('ukrainian_language', [item['key'] for item in sections])
                self.assertNotIn('ukrainian-language', [item.get('bank_slug') for item in sections])
        self.assertNotIn('ukrainian_language', await free_section_keys(store))
        self.assertIsNone(await get_section(store, 'ukrainian_language', is_admin=True))

    async def test_user_access_controls_do_not_include_removed_section(self):
        store = self.make_store()
        service = MiniAppService(SimpleNamespace(store=store, admin_ids={1}))
        auth = AuthContext({}, {}, 1, True)
        detail = await service.admin_user_detail(auth, 42)
        self.assertNotIn('ukrainian_language_access', detail)
        self.assertNotIn('ukrainian_language', [item['key'] for item in detail['section_controls']])

    async def test_recent_sessions_from_removed_section_are_cleared(self):
        for mode in ('pretest', 'learn', 'test'):
            for is_admin in (False, True):
                with self.subTest(mode=mode, is_admin=is_admin):
                    state = {
                        'mode': mode,
                        'header': 'Державна мова',
                        'last_activity_at': dt_to_iso(now()),
                        'meta': {'kind': 'attestation', 'bank_slug': 'ukrainian-language'},
                        'qids': [20_003_026],
                        'pending': [20_003_026],
                    }
                    store = self.make_store()
                    store.get_ui = AsyncMock(return_value={'state': state})
                    store.set_state = AsyncMock()
                    service = MiniAppService(SimpleNamespace(store=store, qb=QuestionBank('unused')))
                    service.build_session_view = AsyncMock(return_value={'screen': 'question'})
                    auth = AuthContext({}, {'sub_infinite': True, 'sub_tier': 'full'}, 42, is_admin)

                    self.assertIsNone(await service.saved_view(auth))

                    store.set_state.assert_awaited_once_with(42, {})
                    service.build_session_view.assert_not_awaited()

    async def test_startup_and_old_links_work_without_the_removed_data_files(self):
        store = self.make_store()
        previous_runtime = getattr(application.app.state, 'runtime', None)
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'DATABASE_URL': 'postgresql://unused', 'BOT_TOKEN': '',
                'QUESTIONS_AUTO_IMPORT': '0',
            }), patch.object(application, 'BASE_DIR', Path(directory)), patch.object(
                application, 'Storage', return_value=store
            ):
                async with application.lifespan(application.app):
                    runtime = application.app.state.runtime
                    self.assertEqual({101}, set(runtime.qb.by_id))
                    self.assertNotIn('ukrainian-language', runtime.qb.attestation_banks)
                    service = MiniAppService(runtime)
                    auth = AuthContext({}, {'section_access': ['ukrainian_language']}, 42, False)
                    for operation in (
                        lambda: service.start_attestation(auth, 'ukrainian-language', StartAttestationRequest(section='', block='random')),
                        lambda: service.attestation_practice_detail(auth, 'ukrainian-language', 20_003_026),
                    ):
                        with self.assertRaises(HTTPException) as raised:
                            await operation()
                        self.assertEqual(404, raised.exception.status_code)
                        self.assertEqual('attestation_bank_not_found', raised.exception.detail['code'])
        finally:
            if previous_runtime is None:
                if hasattr(application.app.state, 'runtime'):
                    del application.app.state.runtime
            else:
                application.app.state.runtime = previous_runtime


class RemovedLanguageRouteTests(unittest.TestCase):
    def test_old_admin_grant_endpoint_returns_not_found(self):
        store = SimpleNamespace(
            set_section_access_override=AsyncMock(return_value=True),
            get_user=AsyncMock(return_value={'user_id': 42}),
            stats=AsyncMock(return_value={}),
            get_setting=AsyncMock(return_value='{}'),
            list_attestation_banks_for_admin=AsyncMock(return_value=[]),
        )
        runtime = SimpleNamespace(store=store, admin_ids={1})
        auth = AuthContext({}, {}, 1, True)
        with patch.dict(application.app.dependency_overrides, {
            application.get_runtime: lambda: runtime,
            application.get_auth_context: lambda: auth,
        }):
            client = TestClient(application.app)
            response = client.post('/api/admin/users/42/ukrainian-language', json={'enabled': True})
        self.assertEqual(404, response.status_code)


if __name__ == '__main__':
    unittest.main()
