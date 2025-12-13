from __future__ import annotations

from aiohttp import web

def make_app() -> web.Application:
    app = web.Application()

    async def healthz(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app.router.add_get("/healthz", healthz)
    return app


async def start_health_server(port: int) -> web.AppRunner:
    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    return runner
