"""Isolated lifecycle & shutdown policy tests for CinematicSRT v2.0.

Zero production app imports, zero FastAPI/Torch/network socket bindings, zero disk mutations.
Extracts the real BrowserLifecycleManager and endpoint handlers from app.py AST and verifies
the UI markup/script contracts in ui/index.html.
"""
import ast
import asyncio
import builtins
from contextlib import ExitStack
import copy
import gc
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
UI_PATH = ROOT / "ui" / "index.html"
VBS_PATH = ROOT / "launch_v2.vbs"
PS1_PATH = ROOT / "launch_v2.ps1"


def denied(*args, **kwargs):
    raise AssertionError("ISOLATION: forbidden external operation")


def load_lifecycle_namespace(
    exit_trigger,
    control_token="test-secret-lifecycle-token-012345",
    child_terminator=None,
):
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    target_funcs = {
        "_parse_bool_env",
        "_parse_grace_seconds",
        "_read_json_body_limited",
        "_default_terminate_descendants",
        "_default_server_exit_trigger",
        "serve_ui",
        "health",
        "trigger_model_download",
        "trigger_ffmpeg_install",
        "shutdown_application",
        "lifecycle_leave",
        "websocket_lifecycle_channel",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BrowserLifecycleManager":
            selected.append(copy.deepcopy(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in target_funcs:
            copied = copy.deepcopy(node)
            copied.decorator_list = []
            selected.append(copied)

    cancelled_jobs = []

    class FakeJobManager:
        def __init__(self):
            self.jobs = {}
            self.tasks = {}
            self.listeners = {}

        async def cancel_job(self, job_id):
            cancelled_jobs.append(job_id)
            t = self.tasks.pop(job_id, None)
            if t and not t.done():
                t.cancel()
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "cancelled"

    class FakeWebSocketDisconnect(Exception):
        pass

    class FakeHTTPException(Exception):
        def __init__(self, status_code=400, detail=""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeResponse:
        def __init__(self, content="", status_code=200, media_type="application/json", headers=None):
            self.content = content
            self.status_code = status_code
            self.media_type = media_type
            self.headers = headers or {}

    class FakeHTMLResponse(FakeResponse):
        def __init__(self, content="", status_code=200, headers=None):
            super().__init__(content=content, status_code=status_code, media_type="text/html", headers=headers)

    class FakeSettings:
        MAX_FILE_SIZE_MB = 4000
        BASE_DIR = None
        TEMP_DIR = None
        UPLOAD_DIR = None

    logger = logging.Logger("isolated-lifecycle-test")
    logger.addHandler(logging.NullHandler())
    job_mgr = FakeJobManager()

    from typing import Any, Optional

    ns = {
        "__name__": "isolated_lifecycle_test",
        "asyncio": asyncio,
        "gc": gc,
        "hmac": hmac,
        "json": json,
        "logging": logger.__class__,
        "logger": logger,
        "os": os,
        "secrets": secrets,
        "signal": signal,
        "sys": __import__("sys"),
        "threading": threading,
        "uuid": uuid,
        "urlparse": urlparse,
        "Optional": Optional,
        "Any": Any,
        "UI_FILE": UI_PATH,
        "Request": object,
        "Response": FakeResponse,
        "HTMLResponse": FakeHTMLResponse,
        "HTTPException": FakeHTTPException,
        "Query": lambda default=None: default,
        "WebSocket": object,
        "WebSocketDisconnect": FakeWebSocketDisconnect,
        "ModelDownloadReq": object,
        "download_local_model": lambda *_a, **_kw: asyncio.sleep(0),
        "install_portable_ffmpeg": lambda *_a, **_kw: asyncio.sleep(0),
        "job_manager": job_mgr,
        "settings": FakeSettings(),
    }

    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, str(APP_PATH), "exec"), ns)

    mgr = ns["BrowserLifecycleManager"](
        job_mgr=job_mgr,
        grace_seconds=0.08,
        auto_shutdown_enabled=True,
        exit_trigger=exit_trigger,
        control_token=control_token,
        child_terminator=(child_terminator if child_terminator is not None else (lambda: [])),
    )
    ns["lifecycle_manager"] = mgr
    ns["_cancelled_jobs"] = cancelled_jobs
    return ns


class FakeWebSocket:
    def __init__(self, incoming=None, disconnect_cls=None, headers=None):
        self.accepted = False
        self.closed_codes = []
        self.sent_json = []
        self.sent_text = []
        self.disconnect_cls = disconnect_cls
        self.headers = headers if headers is not None else {
            "host": "127.0.0.1:8001",
            "origin": "http://127.0.0.1:8001",
            "sec-fetch-site": "same-origin",
        }
        self.incoming = asyncio.Queue()
        for item in (incoming or []):
            self.incoming.put_nowait(item)

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def send_text(self, text):
        self.sent_text.append(text)

    async def close(self, code=1000):
        self.closed_codes.append(code)
        if self.disconnect_cls is not None:
            await self.incoming.put(self.disconnect_cls(code))
        else:
            await self.incoming.put(RuntimeError("WebSocket closed"))

    async def receive_text(self):
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item


class FakeRequest:
    def __init__(self, payload=None, headers=None):
        self._payload = payload
        self.headers = headers if headers is not None else {
            "host": "127.0.0.1:8001",
            "origin": "http://127.0.0.1:8001",
            "sec-fetch-site": "same-origin",
        }

    async def body(self):
        if self._payload is None:
            return b""
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


class TestLifecycleShutdownPolicy(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"app", "core", "services", "fastapi", "starlette", "torch", "transformers", "requests", "httpx", "socket"}:
                denied()
            return original_import(name, *args, **kwargs)

        stack.enter_context(patch.object(builtins, "__import__", guarded_import))
        for obj, key in ((subprocess, "run"), (subprocess, "Popen")):
            stack.enter_context(patch.object(obj, key, denied))

    def test_01_ast_extraction_and_ui_launcher_contracts(self):
        exits = []
        ns = load_lifecycle_namespace(exits.append, control_token="tok-ui-contract")
        self.assertIn("BrowserLifecycleManager", ns)
        self.assertIn("serve_ui", ns)
        self.assertIn("health", ns)
        self.assertIn("shutdown_application", ns)
        self.assertIn("lifecycle_leave", ns)
        self.assertIn("websocket_lifecycle_channel", ns)

        ui_text = UI_PATH.read_text(encoding="utf-8")
        self.assertIn('name="cinematic-lifecycle-token" content="__CINEMATIC_LIFECYCLE_TOKEN__"', ui_text)
        self.assertIn("LIFECYCLE_TOKEN", ui_text)
        self.assertIn("X-Lifecycle-Token", ui_text)
        self.assertIn("activeLifecycleConnId", ui_text)
        self.assertIn("lifecycleConnGen", ui_text)
        self.assertIn("pageHiddenPendingRestore", ui_text)
        self.assertIn('id="btnShutdownApp"', ui_text)
        self.assertIn("shutdownApplication()", ui_text)
        self.assertIn("/api/shutdown", ui_text)
        self.assertIn("/api/lifecycle/ws", ui_text)
        self.assertIn("/api/lifecycle/leave", ui_text)
        self.assertIn("addEventListener('pagehide'", ui_text)
        self.assertIn("addEventListener('pageshow'", ui_text)
        self.assertIn("addEventListener('beforeunload'", ui_text)
        self.assertIn("hasRunningJob", ui_text)
        self.assertIn("sendBeacon", ui_text)

        served = asyncio.run(ns["serve_ui"]())
        self.assertIn('content="tok-ui-contract"', served.content)
        self.assertNotIn("__CINEMATIC_LIFECYCLE_TOKEN__", served.content)
        self.assertEqual(served.headers.get("Cache-Control"), "no-cache")

        self.assertTrue(VBS_PATH.exists())
        self.assertTrue(PS1_PATH.exists())
        vbs_text = VBS_PATH.read_text(encoding="ascii")
        self.assertIn("launch_v2.ps1", vbs_text)
        self.assertIn("WindowStyle Hidden", vbs_text)
        self.assertIn("shell.Run command, 0, False", vbs_text)

        ps1_text = PS1_PATH.read_text(encoding="utf-8-sig")
        self.assertIn(r"Local\CinematicSRT_v2_Launcher_", ps1_text)
        self.assertIn("/api/health", ps1_text)
        self.assertIn("uvicorn", ps1_text)
        self.assertIn("close_fds=True", ps1_text)
        self.assertIn("0x08000200", ps1_text)
        self.assertIn("Get-Process -Id $serverPid", ps1_text)
        self.assertIn("Start-Process $applicationUrl", ps1_text)
        self.assertIn("CINEMATIC_PID=", ps1_text)
        self.assertIn("Get-V2HealthState", ps1_text)
        self.assertIn("hasActiveWork", ui_text)
        self.assertIn("hasActiveUpload", ui_text)
        self.assertIn("hasActiveDownload", ui_text)
        self.assertIn("restoreShutdownControls", ui_text)
        self.assertIn("shutdownInProgress", ui_text)
        self.assertIn("completePositiveShutdown", ui_text)

    def test_02_no_premature_shutdown_before_any_browser_connects(self):
        exits = []
        tok = "tok-02"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]

        async def _run():
            res = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "ghost", "conn_id": "c-ghost", "gen": 1, "token": tok}),
                None,
                None,
                None,
            )
            self.assertFalse(res["tab_removed"])
            await asyncio.sleep(0.15)
            self.assertEqual(exits, [])
            self.assertFalse(mgr.shutting_down)

        asyncio.run(_run())

    def test_03_f5_reload_reconnect_cancels_grace_shutdown(self):
        exits = []
        tok = "tok-03"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]

        async def _run():
            Disconnect = ns["shutdown_application"].__globals__["WebSocketDisconnect"]
            ws1 = FakeWebSocket([Disconnect()])
            await ns["websocket_lifecycle_channel"](ws1, "tab-old", "conn-old", 1, tok)
            self.assertTrue(mgr.has_seen_browser_client)
            self.assertEqual(len(mgr.connections), 0)

            await asyncio.sleep(0.02)
            ws2 = FakeWebSocket()
            task2 = asyncio.create_task(ns["websocket_lifecycle_channel"](ws2, "tab-new", "conn-new", 1, tok))
            await asyncio.sleep(0.01)
            self.assertEqual(len(mgr.connections), 1)

            await asyncio.sleep(0.15)
            self.assertEqual(exits, [])
            self.assertFalse(mgr.shutting_down)

            await ws2.incoming.put(Disconnect())
            await task2

        asyncio.run(_run())

    def test_04_last_browser_tab_close_triggers_cleanup_and_exit(self):
        exits = []
        tok = "tok-04"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]
        job_mgr = ns["job_manager"]

        async def _dummy_job():
            await asyncio.sleep(10)

        async def _run():
            Disconnect = ns["shutdown_application"].__globals__["WebSocketDisconnect"]
            job_mgr.jobs["active-1"] = {"status": "processing"}
            job_mgr.tasks["active-1"] = asyncio.create_task(_dummy_job())
            job_mgr.jobs["done-1"] = {"status": "completed"}

            ws1 = FakeWebSocket(disconnect_cls=Disconnect)
            ws_task = asyncio.create_task(ns["websocket_lifecycle_channel"](ws1, "tab-1", "conn-1", 1, tok))
            await asyncio.sleep(0.01)
            self.assertEqual(len(mgr.connections), 1)

            leave_res = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "tab-1", "conn_id": "conn-1", "gen": 1, "token": tok}),
                None,
                None,
                None,
            )
            self.assertTrue(leave_res["tab_removed"])
            await ws_task

            self.assertEqual(exits, [])

            for _ in range(40):
                if exits:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(exits, ["browser_closed"])
            self.assertTrue(mgr.shutting_down)
            self.assertEqual(ns["_cancelled_jobs"], ["active-1"])
            self.assertEqual(job_mgr.jobs["done-1"]["status"], "completed")

        asyncio.run(_run())

    def test_05_multi_tab_isolation_keeps_server_alive_until_last_tab_closes(self):
        exits = []
        tok = "tok-05"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]

        async def _run():
            Disconnect = ns["shutdown_application"].__globals__["WebSocketDisconnect"]
            ws1 = FakeWebSocket()
            ws2 = FakeWebSocket()
            t1 = asyncio.create_task(ns["websocket_lifecycle_channel"](ws1, "tab-A", "conn-A", 1, tok))
            t2 = asyncio.create_task(ns["websocket_lifecycle_channel"](ws2, "tab-B", "conn-B", 1, tok))
            await asyncio.sleep(0.01)
            self.assertEqual(len(mgr.connections), 2)

            await ws1.incoming.put(Disconnect())
            await t1
            await asyncio.sleep(0.15)
            self.assertEqual(exits, [])
            self.assertFalse(mgr.shutting_down)
            self.assertEqual(len(mgr.connections), 1)

            await ws2.incoming.put(Disconnect())
            await t2
            for _ in range(40):
                if exits:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(exits, ["browser_closed"])

        asyncio.run(_run())

    def test_06_explicit_shutdown_button_is_idempotent_and_notifies_all_tabs(self):
        exits = []
        tok = "tok-06"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]
        job_mgr = ns["job_manager"]

        async def _run():
            Disconnect = ns["shutdown_application"].__globals__["WebSocketDisconnect"]
            job_mgr.jobs["pending-1"] = {"status": "pending"}
            ws = FakeWebSocket(disconnect_cls=Disconnect)
            ws_task = asyncio.create_task(ns["websocket_lifecycle_channel"](ws, "tab-ui", "conn-ui", 1, tok))
            await asyncio.sleep(0.01)

            req = FakeRequest(
                {"token": tok},
                headers={
                    "host": "127.0.0.1:8001",
                    "origin": "http://127.0.0.1:8001",
                    "sec-fetch-site": "same-origin",
                    "x-lifecycle-token": tok,
                },
            )
            first = await ns["shutdown_application"](req)
            self.assertTrue(first["success"])
            self.assertFalse(first["already_shutting_down"])
            self.assertEqual(first["reason"], "user_button")
            self.assertEqual(first["cancelled_jobs"], ["pending-1"])
            self.assertEqual(ws.sent_json, [{"type": "shutdown", "reason": "user_button"}])
            self.assertEqual(ws.closed_codes, [1001])

            second = await ns["shutdown_application"](req)
            self.assertTrue(second["already_shutting_down"])
            self.assertEqual(exits, ["user_button"])
            await ws_task

        asyncio.run(_run())

    def test_07_auto_shutdown_disabled_keeps_manual_button_working(self):
        exits = []
        tok = "tok-07"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]
        mgr.auto_shutdown_enabled = False

        async def _run():
            Disconnect = ns["shutdown_application"].__globals__["WebSocketDisconnect"]
            ws = FakeWebSocket([Disconnect()])
            await ns["websocket_lifecycle_channel"](ws, "tab-only", "conn-only", 1, tok)
            await asyncio.sleep(0.15)
            self.assertEqual(exits, [])
            self.assertFalse(mgr.shutting_down)

            res = await ns["shutdown_application"](FakeRequest({"token": tok}))
            self.assertTrue(res["success"])
            self.assertEqual(exits, ["user_button"])

        asyncio.run(_run())

    def test_08_env_parsers_beacon_json_health_and_bounded_ws(self):
        exits = []
        tok = "tok-08"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]

        with patch.dict(os.environ, {"CINEMATIC_AUTO_SHUTDOWN": "0", "CINEMATIC_CLOSE_GRACE_SECONDS": "2.5"}, clear=False):
            self.assertFalse(ns["_parse_bool_env"]("CINEMATIC_AUTO_SHUTDOWN", True))
            self.assertEqual(ns["_parse_grace_seconds"](), 2.5)
        with patch.dict(os.environ, {"CINEMATIC_CLOSE_GRACE_SECONDS": "999"}, clear=False):
            self.assertEqual(ns["_parse_grace_seconds"](4.5), 4.5)
        with patch.dict(os.environ, {"CINEMATIC_CLOSE_GRACE_SECONDS": "bad"}, clear=False):
            self.assertEqual(ns["_parse_grace_seconds"](4.5), 4.5)

        async def _run():
            healthy = await ns["health"]()
            self.assertEqual(healthy["status"], "ok")
            self.assertEqual(healthy["service"], "CinematicSRT v2.0")

            Disconnect = ns["shutdown_application"].__globals__["WebSocketDisconnect"]
            ws = FakeWebSocket(disconnect_cls=Disconnect)
            ws_task = asyncio.create_task(ns["websocket_lifecycle_channel"](ws, "beacon-tab", "conn-beacon", 1, tok))
            await asyncio.sleep(0.01)

            bad_res = await ns["lifecycle_leave"](
                FakeRequest(
                    b"{broken-json",
                    headers={
                        "host": "127.0.0.1:8001",
                        "origin": "http://127.0.0.1:8001",
                        "sec-fetch-site": "same-origin",
                        "x-lifecycle-token": tok,
                    },
                ),
                None,
                None,
                None,
            )
            self.assertFalse(bad_res["tab_removed"])

            ok_res = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "beacon-tab", "conn_id": "conn-beacon", "gen": 1, "token": tok}),
                None,
                None,
                None,
            )
            self.assertTrue(ok_res["tab_removed"])
            await ws_task

            for _ in range(40):
                if exits:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(exits, ["browser_closed"])
            unhealthy = await ns["health"]()
            self.assertEqual(unhealthy.status_code, 503)
            self.assertEqual(json.loads(unhealthy.content)["status"], "shutting_down")

            class SlowWs:
                async def close(self, code=1000):
                    await asyncio.sleep(10.0)

            mgr2 = ns["BrowserLifecycleManager"](
                job_mgr=ns["job_manager"],
                grace_seconds=0.05,
                auto_shutdown_enabled=True,
                exit_trigger=lambda _r: None,
                control_token=tok,
            )
            await mgr2.register_connection("c-slow", "tab-slow", SlowWs(), conn_gen=1)
            await asyncio.wait_for(mgr2.mark_tab_leaving("tab-slow", conn_id="c-slow", conn_gen=1), timeout=1.0)
            self.assertEqual(len(mgr2.connections), 0)

            release_shield = asyncio.Event()

            class ShieldedSlowWs:
                async def close(self, code=1000):
                    try:
                        await release_shield.wait()
                    except asyncio.CancelledError:
                        await release_shield.wait()

            await mgr2.register_connection("c-shield", "tab-shield", ShieldedSlowWs(), conn_gen=1)
            await asyncio.wait_for(mgr2.mark_tab_leaving("tab-shield", conn_id="c-shield", conn_gen=1), timeout=0.8)
            self.assertEqual(len(mgr2.connections), 0)
            release_shield.set()
            await asyncio.sleep(0)

        asyncio.run(_run())

    def test_09_grace_shutdown_with_yielding_job_cancel_and_listener_exits_cleanly(self):
        """Regression for Blocker 2: grace task must not self-cancel on awaiting job/listener teardown."""
        exits = []
        tok = "tok-09"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]
        job_mgr = ns["job_manager"]

        async def _yielding_cancel(jid):
            await asyncio.sleep(0)
            ns["_cancelled_jobs"].append(jid)
            if jid in job_mgr.jobs:
                job_mgr.jobs[jid]["status"] = "cancelled"

        job_mgr.cancel_job = _yielding_cancel

        class YieldingJobWs:
            def __init__(self):
                self.closed_codes = []

            async def close(self, code=1000):
                await asyncio.sleep(0)
                self.closed_codes.append(code)

        async def _run():
            Disconnect = ns["shutdown_application"].__globals__["WebSocketDisconnect"]
            job_ws = YieldingJobWs()
            job_mgr.jobs["active-yield"] = {"status": "processing"}
            job_mgr.listeners["active-yield"] = [job_ws]

            ws1 = FakeWebSocket([Disconnect()])
            await ns["websocket_lifecycle_channel"](ws1, "tab-yield", "conn-yield", 1, tok)
            for _ in range(40):
                if exits:
                    break
                await asyncio.sleep(0.02)

            self.assertTrue(mgr.shutting_down)
            self.assertEqual(exits, ["browser_closed"])
            self.assertEqual(ns["_cancelled_jobs"], ["active-yield"])
            self.assertEqual(job_mgr.listeners, {})
            self.assertEqual(job_ws.closed_codes, [1001])

        asyncio.run(_run())

    def test_10_foreign_origin_rejection_and_stale_leave_bfcache_orderings(self):
        """Regressions for Blockers 1 and 3: provenance/token defense & generation-scoped leave."""
        exits = []
        tok = "tok-10-unguessable"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]
        HTTPExc = ns["HTTPException"]

        async def _run():
            # --- Part A: Cross-origin / cross-site / missing-token rejections ---
            foreign_cases = [
                # Foreign site even with text/plain simple POST
                FakeRequest(b"x", headers={"host": "127.0.0.1:8001", "origin": "https://unrelated.example", "sec-fetch-site": "cross-site"}),
                # Cross-port local site pretending same-site with leaked token must still fail origin/sec-fetch-site
                FakeRequest({"token": tok}, headers={"host": "127.0.0.1:8001", "origin": "http://127.0.0.1:9999", "sec-fetch-site": "same-site", "x-lifecycle-token": tok}),
                # Sandbox / null origin with token
                FakeRequest({"token": tok}, headers={"host": "127.0.0.1:8001", "origin": "null", "x-lifecycle-token": tok}),
                # Localhost origin missing token
                FakeRequest({}, headers={"host": "127.0.0.1:8001", "origin": "http://127.0.0.1:8001", "sec-fetch-site": "same-origin"}),
                # DNS rebinding Host
                FakeRequest({"token": tok}, headers={"host": "evil.example:8001", "x-lifecycle-token": tok}),
            ]
            for bad_req in foreign_cases:
                with self.assertRaises(HTTPExc) as ctx1:
                    await ns["shutdown_application"](bad_req)
                self.assertEqual(ctx1.exception.status_code, 403)
                with self.assertRaises(HTTPExc) as ctx2:
                    await ns["lifecycle_leave"](bad_req, "tab-x", "conn-x", 1)
                self.assertEqual(ctx2.exception.status_code, 403)

            # Foreign WS origin (both with and without token) rejected BEFORE accept()
            ws_foreign_notok = FakeWebSocket(headers={"host": "127.0.0.1:8001", "origin": "https://unrelated.example", "sec-fetch-site": "cross-site"})
            await ns["websocket_lifecycle_channel"](ws_foreign_notok, "t-bad", "c-bad", 1, None)
            self.assertFalse(ws_foreign_notok.accepted)
            self.assertEqual(ws_foreign_notok.closed_codes, [1008])

            ws_foreign_tok = FakeWebSocket(headers={"host": "127.0.0.1:8001", "origin": "https://unrelated.example", "sec-fetch-site": "cross-site"})
            await ns["websocket_lifecycle_channel"](ws_foreign_tok, "t-bad", "c-bad", 1, tok)
            self.assertFalse(ws_foreign_tok.accepted)
            self.assertEqual(ws_foreign_tok.closed_codes, [1008])

            ws_missing_tok = FakeWebSocket(headers={"host": "127.0.0.1:8001", "origin": "http://127.0.0.1:8001", "sec-fetch-site": "same-origin"})
            await ns["websocket_lifecycle_channel"](ws_missing_tok, "t-bad", "c-bad", 1, "wrong-token")
            self.assertFalse(ws_missing_tok.accepted)
            self.assertEqual(ws_missing_tok.closed_codes, [1008])

            self.assertFalse(mgr.has_seen_browser_client)
            self.assertEqual(mgr.connections, {})
            self.assertEqual(exits, [])

            # --- Part B: Both beacon orderings + bfcache restore on same tab ID ---
            ws_orig = FakeWebSocket()
            ws_restored = FakeWebSocket()

            # Ordering 1: unregister(original) -> register(restored, same-tab) -> late leave(original)
            await mgr.register_connection("original", "same-tab", ws_orig, conn_gen=1)
            mgr.unregister_connection("original")
            await mgr.register_connection("restored", "same-tab", ws_restored, conn_gen=2)

            stale_res = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "same-tab", "conn_id": "original", "gen": 1, "token": tok}),
                None,
                None,
                None,
            )
            self.assertFalse(stale_res["tab_removed"])
            self.assertEqual(stale_res["active_tabs"], 1)
            self.assertIn("restored", mgr.connections)

            # Un-versioned leave (missing conn_id) or stale gen on restored conn_id must also fail safe
            no_conn_res = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "same-tab", "token": tok}),
                None,
                None,
                None,
            )
            self.assertFalse(no_conn_res["tab_removed"])
            self.assertIn("restored", mgr.connections)

            stale_gen_res = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "same-tab", "conn_id": "restored", "gen": 1, "token": tok}),
                None,
                None,
                None,
            )
            self.assertFalse(stale_gen_res["tab_removed"])
            self.assertIn("restored", mgr.connections)

            await asyncio.sleep(0.15)
            self.assertEqual(exits, [])
            self.assertFalse(mgr.shutting_down)

            # Ordering 2: Overlapping WS during bfcache restore -> old leave removes only old conn_id
            ws_orig_2 = FakeWebSocket()
            ws_restored_2 = FakeWebSocket()
            await mgr.register_connection("overlap-old", "tab-bf", ws_orig_2, conn_gen=1)
            await mgr.register_connection("overlap-new", "tab-bf", ws_restored_2, conn_gen=2)
            old_leave = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "tab-bf", "conn_id": "overlap-old", "gen": 1, "token": tok}),
                None,
                None,
                None,
            )
            self.assertTrue(old_leave["tab_removed"])
            self.assertIn("overlap-new", mgr.connections)
            self.assertIn("restored", mgr.connections)

            # Ordering 3: Genuine leave BEFORE unregister for active connections shuts down after grace
            res_r1 = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "same-tab", "conn_id": "restored", "gen": 2, "token": tok}),
                None,
                None,
                None,
            )
            self.assertTrue(res_r1["tab_removed"])
            res_r2 = await ns["lifecycle_leave"](
                FakeRequest({"tab_id": "tab-bf", "conn_id": "overlap-new", "gen": 2, "token": tok}),
                None,
                None,
                None,
            )
            self.assertTrue(res_r2["tab_removed"])
            for _ in range(40):
                if exits:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(exits, ["browser_closed"])
            self.assertTrue(mgr.shutting_down)

        asyncio.run(_run())



    def test_11_ipv6_empty_token_header_duplicate_conn_and_shutdown_gating(self):
        """Verify IPv6 [::1]:8001 provenance, empty header fallback, duplicate conn_id rejection, and 503 gating."""
        exits = []
        tok = "tok-11-ipv6-edge"
        ns = load_lifecycle_namespace(exits.append, control_token=tok)
        mgr = ns["lifecycle_manager"]
        HTTPExc = ns["HTTPException"]

        async def _run():
            # IPv6 loopback host + Origin & Referer matching
            self.assertTrue(
                mgr.is_authorized_request(
                    {"host": "[::1]:8001", "origin": "http://[::1]:8001", "sec-fetch-site": "same-origin"},
                    tok,
                )
            )
            self.assertTrue(
                mgr.is_authorized_request(
                    {"host": "[::1]:8001", "referer": "http://[::1]:8001/studio", "sec-fetch-site": "same-origin"},
                    tok,
                )
            )
            self.assertFalse(
                mgr.is_authorized_request(
                    {"host": "[::1]:8001", "origin": "http://[::1]:9999", "sec-fetch-site": "same-site"},
                    tok,
                )
            )

            # Empty x-lifecycle-token header falls back to JSON body token cleanly
            ws = FakeWebSocket(headers={"host": "[::1]:8001", "origin": "http://[::1]:8001", "sec-fetch-site": "same-origin"})
            ws_task = asyncio.create_task(ns["websocket_lifecycle_channel"](ws, "tab-v6", "conn-v6", 1, tok))
            await asyncio.sleep(0.01)
            self.assertIn("conn-v6", mgr.connections)

            # Duplicate active conn_id must be rejected cleanly before accept()
            ws_dup = FakeWebSocket(headers={"host": "127.0.0.1:8001", "origin": "http://127.0.0.1:8001", "sec-fetch-site": "same-origin"})
            await ns["websocket_lifecycle_channel"](ws_dup, "tab-v6", "conn-v6", 2, tok)
            self.assertFalse(ws_dup.accepted)
            self.assertEqual(ws_dup.closed_codes, [1008])

            leave_res = await ns["lifecycle_leave"](
                FakeRequest(
                    {"tab_id": "tab-v6", "conn_id": "conn-v6", "gen": 1, "token": tok},
                    headers={
                        "host": "[::1]:8001",
                        "origin": "http://[::1]:8001",
                        "sec-fetch-site": "same-origin",
                        "x-lifecycle-token": "",
                    },
                ),
                None,
                None,
                None,
            )
            self.assertTrue(leave_res["tab_removed"])
            await ws_task

            # Explicit shutdown with empty header + valid body token -> succeeds, subsequent downloads return 503
            res = await ns["shutdown_application"](
                FakeRequest(
                    {"token": tok},
                    headers={
                        "host": "[::1]:8001",
                        "origin": "http://[::1]:8001",
                        "sec-fetch-site": "same-origin",
                        "x-lifecycle-token": "",
                    },
                )
            )
            self.assertTrue(res["success"])
            self.assertEqual(exits, ["user_button"])

            class DummyReq:
                model_type = "whisper"

            with self.assertRaises(HTTPExc) as dl_exc:
                await ns["trigger_model_download"](DummyReq())
            self.assertEqual(dl_exc.exception.status_code, 503)
            with self.assertRaises(HTTPExc) as ff_exc:
                await ns["trigger_ffmpeg_install"]()
            self.assertEqual(ff_exc.exception.status_code, 503)

        asyncio.run(_run())

    def test_12_shutdown_cancels_setup_tasks_kills_children_and_purges_locked_temps(self):
        """Verify shutdown order: cancel jobs/setup -> terminate children -> purge unlocked temps."""
        import tempfile

        events = []
        exits = []
        tok = "tok-12-cleanup-order"

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            temp_dir = base / "temp"
            upload_dir = base / "uploads"
            bin_dir = base / "bin"
            for d in (temp_dir, upload_dir, bin_dir):
                d.mkdir(parents=True, exist_ok=True)

            locked_wav = temp_dir / "job-child_16k.wav"
            locked_wav.write_bytes(b"locked-until-child-killed")
            part_file = upload_dir / "job-child.mp4.part"
            part_file.write_bytes(b"partial-upload")
            partial_zip = bin_dir / "ffmpeg.zip"
            partial_zip.write_bytes(b"partial-zip")

            child_alive = {"flag": True}

            def _terminator():
                events.append("child_terminator")
                child_alive["flag"] = False
                return [99999]

            ns = load_lifecycle_namespace(exits.append, control_token=tok, child_terminator=_terminator)
            mgr = ns["lifecycle_manager"]
            job_mgr = ns["job_manager"]
            ns["settings"].BASE_DIR = base
            ns["settings"].TEMP_DIR = temp_dir
            ns["settings"].UPLOAD_DIR = upload_dir

            async def _yielding_cancel(jid):
                events.append(f"cancel_job:{jid}")
                job_mgr.tasks.pop(jid, None)
                if jid in job_mgr.jobs:
                    job_mgr.jobs[jid]["status"] = "cancelled"

            job_mgr.cancel_job = _yielding_cancel

            async def _hanging_ffmpeg_dl():
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    events.append("cancelled_ffmpeg_install")
                    raise

            async def _run():
                job_mgr.jobs["job-child"] = {"status": "processing"}
                mgr.spawn_tracked_task("ffmpeg_install", _hanging_ffmpeg_dl())
                await asyncio.sleep(0.01)

                res = await mgr.request_shutdown(reason="user_button")
                self.assertTrue(res["success"])
                self.assertEqual(res["cancelled_jobs"], ["job-child"])
                self.assertIn("ffmpeg_install", res["cancelled_background_tasks"])
                self.assertEqual(exits, ["user_button"])
                self.assertIn("cancelled_ffmpeg_install", events)
                self.assertIn("cancel_job:job-child", events)
                self.assertIn("child_terminator", events)
                self.assertLess(events.index("cancel_job:job-child"), events.index("child_terminator"))
                self.assertFalse(child_alive["flag"])
                self.assertFalse(locked_wav.exists())
                self.assertFalse(part_file.exists())
                self.assertFalse(partial_zip.exists())

            asyncio.run(_run())



class TestShutdownUiBehavioralRegression(unittest.TestCase):
    """Isolated Node VM behavioral verification of real ui/index.html shutdown logic."""

    @staticmethod
    def _find_node() -> str:
        import shutil
        found = shutil.which("node")
        if found:
            return found
        for cand in (
            Path.home() / "AppData" / "Local" / "hermes" / "node" / "node.exe",
            Path(r"C:\Program Files\nodejs\node.exe"),
        ):
            if cand.exists():
                return str(cand)
        raise AssertionError("Node.js runtime not found for UI behavioral regression")

    def _run_node(self, script_body: str) -> dict:
        node_exe = self._find_node()
        proc = subprocess.run(
            [node_exe, "-"],
            input=script_body,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        self.assertEqual(proc.returncode, 0, f"Node probe failed: stdout={proc.stdout} stderr={proc.stderr}")
        return json.loads(proc.stdout.strip())

    def test_01_shutdown_ui_outcomes_unknown_alive_in_progress_and_positive(self):
        script = f"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const html = fs.readFileSync({json.dumps(str(UI_PATH))}, 'utf8');
const sliceStart = html.indexOf('    let appShuttingDown = false;');
const sliceEnd = html.indexOf("window.addEventListener('DOMContentLoaded'", sliceStart);
assert.ok(sliceStart > 0 && sliceEnd > sliceStart, 'Failed locating lifecycle script block');
const baseSnippet = html.slice(sliceStart, sliceEnd);

const fnStart = html.indexOf('    async function shutdownApplication()');
assert.ok(fnStart > sliceStart && sliceEnd > fnStart, 'Failed locating shutdownApplication slice');
const fnOnlySnippet = html.slice(fnStart, sliceEnd);

function runIsolatedCase({{ shutdownBehavior, healthBehavior, simulateWsEventDuringFetch = false }}) {{
  const actions = [];
  const alerts = [];
  const btnEl = {{ disabled: false, textContent: '⏻ إيقاف البرنامج', title: '' }};
  const overlayEl = {{ style: {{ display: 'none' }} }};
  const descEl = {{ textContent: '' }};
  const domMap = {{ btnShutdownApp: btnEl, shutdownOverlay: overlayEl, shutdownDesc: descEl }};
  let triggerWsShutdown = null;
  const context = vm.createContext({{
    console,
    URLSearchParams,
    Blob,
    fetch: async (url) => {{
      if (url === '/api/shutdown') {{
        if (simulateWsEventDuringFetch && triggerWsShutdown) {{
          triggerWsShutdown();
          actions.push('ws_shutdown_event');
        }}
        if (shutdownBehavior instanceof Error) throw shutdownBehavior;
        if (typeof shutdownBehavior === 'function') return await shutdownBehavior();
        return shutdownBehavior;
      }}
      if (url === '/api/health') {{
        if (healthBehavior instanceof Error) throw healthBehavior;
        if (typeof healthBehavior === 'function') return await healthBehavior();
        return healthBehavior;
      }}
      throw new Error('Unexpected fetch url: ' + url);
    }},
    hasActiveWork: () => true,
    hasActiveUpload: false,
    hasActiveDownload: false,
    hasRunningJob: true,
    activeJobId: 'job-active-123',
    LIFECYCLE_TOKEN: 'fixture-token',
    LIFECYCLE_TAB_ID: 'tab-1',
    lifecycleReconnectTimer: null,
    lifecycleWs: null,
    activeLifecycleConnId: 'conn-1',
    lifecycleConnGen: 0,
    dlPollTimer: null,
    clearTimeout: () => {{}},
    clearInterval: () => {{}},
    confirm: () => true,
    alert: (msg) => {{ actions.push('alert'); alerts.push(String(msg)); }},
    document: {{
      querySelector: () => ({{ getAttribute: () => 'fixture-token' }}),
      getElementById: (id) => domMap[id] || {{}},
    }},
    cleanupSockets: () => actions.push('cleanup'),
    generateLifecycleUuid: () => 'conn-ws-gen',
    window: {{
      close: () => actions.push('close_window'),
      addEventListener: () => {{}},
    }},
  }});
  const origShow = [];
  const NL = String.fromCharCode(10);
  vm.runInContext(
    [
      baseSnippet,
      'const __realShow = showShutdownOverlay;',
      'showShutdownOverlay = (m) => {{ __showCalls.push(m || null); return __realShow(m); }};',
      'connectLifecycleChannel = () => __actions.push("reconnect");',
      '__setTrigger(() => {{ showShutdownOverlay(); }});',
      '__setReader(() => ({{ appShuttingDown, shutdownInProgress, activeJobId, hasRunningJob }}));',
    ].join(NL),
    Object.assign(context, {{
      __showCalls: origShow,
      __actions: actions,
      __setTrigger: (fn) => {{ triggerWsShutdown = fn; }},
      __setReader: (fn) => {{ context.__readState = fn; }},
    }})
  );
  return vm.runInContext('shutdownApplication()', context).then(() => {{
    const st = context.__readState();
    return {{
      actions,
      alerts,
      showCallsCount: origShow.length,
      appShuttingDown: st.appShuttingDown,
      shutdownInProgress: st.shutdownInProgress,
      activeJobId: st.activeJobId,
      hasRunningJob: st.hasRunningJob,
      btnDisabled: btnEl.disabled,
      btnText: btnEl.textContent,
      overlayDisplay: overlayEl.style.display,
    }};
  }});
}}

(async () => {{
  const results = {{}};

  // 1. Both POST /api/shutdown and GET /api/health reject (transport failure) -> unknown outcome, preserve session & offer retry
  results.both_reject = await runIsolatedCase({{
    shutdownBehavior: new Error('ECONNRESET shutdown'),
    healthBehavior: new Error('ECONNREFUSED health'),
  }});
  assert.equal(results.both_reject.appShuttingDown, false);
  assert.equal(results.both_reject.shutdownInProgress, false);
  assert.equal(results.both_reject.activeJobId, 'job-active-123');
  assert.equal(results.both_reject.hasRunningJob, true);
  assert.equal(results.both_reject.btnDisabled, false);
  assert.equal(results.both_reject.btnText, '⏻ إيقاف البرنامج');
  assert.equal(results.both_reject.overlayDisplay, 'none');
  assert.equal(results.both_reject.showCallsCount, 0);
  assert.deepEqual(results.both_reject.actions, ['reconnect', 'alert']);

  // 2. Transport failure + GET /api/health HTTP 500 -> arbitrary non-OK does NOT confirm shutdown
  results.health_500 = await runIsolatedCase({{
    shutdownBehavior: new Error('Network drop'),
    healthBehavior: {{ status: 500, ok: false }},
  }});
  assert.equal(results.health_500.appShuttingDown, false);
  assert.equal(results.health_500.shutdownInProgress, false);
  assert.equal(results.health_500.activeJobId, 'job-active-123');
  assert.equal(results.health_500.btnDisabled, false);
  assert.equal(results.health_500.overlayDisplay, 'none');
  assert.deepEqual(results.health_500.actions, ['reconnect', 'alert']);

  // 3. Transport failure + GET /api/health HTTP 503 WITHOUT shutting_down payload or wrong service -> unknown
  results.health_503_arbitrary = await runIsolatedCase({{
    shutdownBehavior: new Error('Network drop'),
    healthBehavior: {{ status: 503, ok: false, json: async () => ({{ status: 'error', service: 'proxy' }}) }},
  }});
  assert.equal(results.health_503_arbitrary.appShuttingDown, false);
  assert.equal(results.health_503_arbitrary.activeJobId, 'job-active-123');
  assert.deepEqual(results.health_503_arbitrary.actions, ['reconnect', 'alert']);

  // 4. Transport failure + GET /api/health HTTP 200 -> server still alive, preserve session & offer retry
  results.health_200 = await runIsolatedCase({{
    shutdownBehavior: new Error('Network drop'),
    healthBehavior: {{ status: 200, ok: true, json: async () => ({{ status: 'ok', service: 'CinematicSRT v2.0' }}) }},
  }});
  assert.equal(results.health_200.appShuttingDown, false);
  assert.equal(results.health_200.shutdownInProgress, false);
  assert.equal(results.health_200.activeJobId, 'job-active-123');
  assert.equal(results.health_200.btnDisabled, false);
  assert.equal(results.health_200.overlayDisplay, 'none');
  assert.deepEqual(results.health_200.actions, ['reconnect', 'alert']);

  // 5. Explicit POST /api/shutdown HTTP 500 and HTTP 403 -> preserve session & offer retry
  results.post_500 = await runIsolatedCase({{
    shutdownBehavior: {{ status: 500, ok: false, json: async () => ({{ detail: 'boom' }}) }},
    healthBehavior: new Error('unused'),
  }});
  assert.equal(results.post_500.appShuttingDown, false);
  assert.equal(results.post_500.activeJobId, 'job-active-123');
  assert.deepEqual(results.post_500.actions, ['reconnect', 'alert']);

  // 6. Transport failure + explicit GET /api/health 503 shutting_down -> treated as IN PROGRESS
  results.health_shutting_down = await runIsolatedCase({{
    shutdownBehavior: new Error('Network drop'),
    healthBehavior: {{ status: 503, ok: false, json: async () => ({{ status: 'shutting_down', service: 'CinematicSRT v2.0' }}) }},
  }});
  assert.equal(results.health_shutting_down.appShuttingDown, false);
  assert.equal(results.health_shutting_down.shutdownInProgress, true);
  assert.equal(results.health_shutting_down.activeJobId, 'job-active-123');
  assert.equal(results.health_shutting_down.hasRunningJob, true);
  assert.equal(results.health_shutting_down.btnDisabled, true);
  assert.equal(results.health_shutting_down.btnText, '⏳ جارٍ الإيقاف...');
  assert.equal(results.health_shutting_down.overlayDisplay, 'none');
  assert.equal(results.health_shutting_down.showCallsCount, 0);
  assert.deepEqual(results.health_shutting_down.actions, ['reconnect']);

  // 7. Positive HTTP shutdown response (200 OK with success: true) -> full shutdown overlay & close
  results.positive_http_200 = await runIsolatedCase({{
    shutdownBehavior: {{ status: 200, ok: true, json: async () => ({{ success: true, reason: 'user_button' }}) }},
    healthBehavior: new Error('unused'),
  }});
  assert.equal(results.positive_http_200.appShuttingDown, true);
  assert.equal(results.positive_http_200.shutdownInProgress, false);
  assert.equal(results.positive_http_200.activeJobId, null);
  assert.equal(results.positive_http_200.hasRunningJob, false);
  assert.equal(results.positive_http_200.btnDisabled, true);
  assert.equal(results.positive_http_200.btnText, '⏻ تم الإيقاف');
  assert.equal(results.positive_http_200.overlayDisplay, 'flex');
  assert.equal(results.positive_http_200.showCallsCount, 1);
  assert.deepEqual(results.positive_http_200.actions, ['cleanup', 'cleanup', 'close_window']);

  // 8. Positive lifecycle shutdown event arriving during in-flight transport drop -> closes window cleanly without alert
  results.positive_ws_during_drop = await runIsolatedCase({{
    shutdownBehavior: new Error('Connection closed during exit'),
    healthBehavior: new Error('Already exited'),
    simulateWsEventDuringFetch: true,
  }});
  assert.equal(results.positive_ws_during_drop.appShuttingDown, true);
  assert.equal(results.positive_ws_during_drop.shutdownInProgress, false);
  assert.equal(results.positive_ws_during_drop.activeJobId, null);
  assert.equal(results.positive_ws_during_drop.overlayDisplay, 'flex');
  assert.deepEqual(results.positive_ws_during_drop.actions, ['cleanup', 'ws_shutdown_event', 'close_window']);

  // 9. POST /api/shutdown HTTP 200 with non-JSON / invalid payload must NOT confirm shutdown
  results.post_200_bad_json = await runIsolatedCase({{
    shutdownBehavior: {{ status: 200, ok: true, json: async () => {{ throw new SyntaxError('Unexpected token <'); }} }},
    healthBehavior: new Error('unused'),
  }});
  assert.equal(results.post_200_bad_json.appShuttingDown, false);
  assert.equal(results.post_200_bad_json.shutdownInProgress, false);
  assert.equal(results.post_200_bad_json.activeJobId, 'job-active-123');
  assert.equal(results.post_200_bad_json.overlayDisplay, 'none');
  assert.deepEqual(results.post_200_bad_json.actions, ['reconnect', 'alert']);

  // 10. Minimal standalone fnOnlySnippet (v3-ui-probe style) works without outer let declarations
  const fnActions = [];
  const fnCtx = vm.createContext({{
    fetch: async () => {{ throw new Error('Network error before delivery'); }},
    hasActiveWork: () => true,
    hasActiveUpload: false,
    hasActiveDownload: false,
    pageHiddenPendingRestore: false,
    connectLifecycleChannel: () => fnActions.push('reconnect'),
    alert: () => fnActions.push('reports error'),
    confirm: () => true,
    document: {{ getElementById: () => ({{}}) }},
    LIFECYCLE_TOKEN: 'fixture',
    hasRunningJob: true,
    appShuttingDown: false,
    activeJobId: 'active',
    lifecycleReconnectTimer: null,
    lifecycleWs: null,
    activeLifecycleConnId: 'fixture',
    cleanupSockets: () => fnActions.push('cleanup'),
    showShutdownOverlay: () => fnActions.push('claims stopped'),
    window: {{ close: () => fnActions.push('closes window') }},
  }});
  await vm.runInContext([fnOnlySnippet, 'shutdownApplication()'].join(String.fromCharCode(10)), fnCtx);
  assert.deepEqual(fnActions, ['reconnect', 'reports error']);
  assert.equal(fnCtx.activeJobId, 'active');
  assert.equal(fnCtx.appShuttingDown, false);
  results.standalone_fn_slice_ok = true;

  console.log(JSON.stringify(results));
}})().catch((err) => {{ console.error(err); process.exitCode = 1; }});
"""
        res = self._run_node(script)
        self.assertIn("both_reject", res)
        self.assertIn("health_500", res)
        self.assertIn("health_200", res)
        self.assertIn("health_shutting_down", res)
        self.assertIn("positive_http_200", res)
        self.assertIn("positive_ws_during_drop", res)
        self.assertIn("post_200_bad_json", res)
        self.assertTrue(res["standalone_fn_slice_ok"])

    def test_03_shutdown_deadline_bounds_requests_and_allows_retry(self):
        script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(UI_FIXTURE, 'utf8');
const start = html.indexOf('    async function shutdownApplication()');
const end = html.indexOf("    window.addEventListener('DOMContentLoaded'", start);
const response = (status, body) => ({status, ok: status === 200, json: async () => body});
const pending = () => new Promise(() => {});

(async () => {
  for (const mode of ['post', 'post-json', 'health', 'health-json', 'in-progress']) {
    const btn = {}, timers = new Map(), events = [];
    let nextId = 0, resolveLate, signal, retry = false;
    const ctx = vm.createContext({
      AbortController, setTimeout(fn, ms) {
        assert.equal(ms, 10000);
        timers.set(++nextId, fn);
        return nextId;
      },
      clearTimeout(id) { timers.delete(id); },
      confirm: () => true, hasActiveWork: () => true, appShuttingDown: false,
      shutdownInProgress: false, hasRunningJob: true, hasActiveUpload: false,
      hasActiveDownload: false, activeJobId: 'active', lifecycleWs: null,
      lifecycleReconnectTimer: null, pageHiddenPendingRestore: false,
      activeLifecycleConnId: null, LIFECYCLE_TOKEN: 'fixture',
      document: {getElementById: () => btn},
      alert: () => events.push('alert'), connectLifecycleChannel() {},
      cleanupSockets() {}, showShutdownOverlay: () => events.push('success'),
      window: {close: () => events.push('close')},
      fetch(url, options) {
        signal = options.signal;
        assert.equal(signal.aborted, false);
        if (retry) return response(200, {success: true});
        if (url.endsWith('/shutdown')) {
          if (mode === 'post') return new Promise(resolve => { resolveLate = resolve; });
          if (mode === 'post-json') return {ok: true, json: pending};
          return Promise.reject(Error('network'));
        }
        if (mode === 'health') return pending();
        if (mode === 'health-json') return {status: 503, json: pending};
        return response(503, {status: 'shutting_down', service: 'CinematicSRT v2.0'});
      }
    });
    vm.runInContext(html.slice(start, end), ctx);
    const first = ctx.shutdownApplication();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(btn.disabled, true);
    assert.equal(timers.size, 1);
    const expire = timers.values().next().value;
    expire();
    await first;
    assert.equal(signal.aborted, true);
    assert.equal(btn.disabled, false);
    assert.equal(ctx.activeJobId, 'active');
    assert.deepEqual(events, ['alert']);
    assert.equal(timers.size, 0);
    retry = true;
    const second = ctx.shutdownApplication();
    expire();
    if (resolveLate) resolveLate(response(200, {success: true}));
    await second;
    assert.deepEqual(events, ['alert', 'success', 'close']);
    assert.equal(timers.size, 0);
  }
  console.log(JSON.stringify({cases: 5}));
})().catch(err => { console.error(err); process.exitCode = 1; });
"""
        result = self._run_node(script.replace("UI_FIXTURE", json.dumps(str(UI_PATH))))
        self.assertEqual(result["cases"], 5)

    def test_02_live_ws_shutdown_event_and_old_fallthrough_mutant_fails(self):
        script = f"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const html = fs.readFileSync({json.dumps(str(UI_PATH))}, 'utf8');
const sliceStart = html.indexOf('    let appShuttingDown = false;');
const sliceEnd = html.indexOf("window.addEventListener('DOMContentLoaded'", sliceStart);
const snippet = html.slice(sliceStart, sliceEnd);

// Part A: integrated WS shutdown message triggers showShutdownOverlay during fetch drop
(async () => {{
  const actions = [];
  const btnEl = {{ disabled: false, textContent: '⏻ إيقاف البرنامج', title: '' }};
  const overlayEl = {{ style: {{ display: 'none' }} }};
  const domMap = {{ btnShutdownApp: btnEl, shutdownOverlay: overlayEl, shutdownDesc: {{ textContent: '' }} }};
  let createdWs = null;
  function FakeWs(url) {{
    this.url = url;
    this.readyState = 1;
    this.close = () => {{ this.readyState = 3; }};
    createdWs = this;
  }}
  FakeWs.OPEN = 1;
  FakeWs.CONNECTING = 0;

  const ctx = vm.createContext({{
    console,
    URLSearchParams,
    Blob,
    WebSocket: FakeWs,
    location: {{ protocol: 'http:', host: '127.0.0.1:8001' }},
    fetch: async (url) => {{
      if (url === '/api/shutdown') {{
        createdWs.onmessage({{ data: JSON.stringify({{ type: 'shutdown', reason: 'user_button' }}) }});
        throw new Error('Server closed connection during exit');
      }}
      throw new Error('Unexpected fetch: ' + url);
    }},
    hasActiveWork: () => true,
    hasActiveUpload: false,
    hasActiveDownload: false,
    hasRunningJob: true,
    activeJobId: 'job-ws-1',
    LIFECYCLE_TOKEN: 'fixture-token',
    LIFECYCLE_TAB_ID: 'tab-ws',
    lifecycleReconnectTimer: null,
    lifecycleWs: null,
    activeLifecycleConnId: 'conn-ws',
    lifecycleConnGen: 0,
    dlPollTimer: null,
    clearTimeout: () => {{}},
    clearInterval: () => {{}},
    setTimeout: () => 1,
    confirm: () => true,
    alert: (m) => actions.push('alert:' + m),
    document: {{
      querySelector: () => ({{ getAttribute: () => 'fixture-token' }}),
      getElementById: (id) => domMap[id] || {{}},
    }},
    cleanupSockets: () => actions.push('cleanup'),
    generateLifecycleUuid: () => 'conn-ws-gen',
    window: {{
      close: () => actions.push('close_window'),
      addEventListener: () => {{}},
    }},
  }});
  const NL = String.fromCharCode(10);
  vm.runInContext(
    [
      snippet,
      '__setStateReader(() => ({{ appShuttingDown, shutdownInProgress, activeJobId }}));',
      'connectLifecycleChannel();',
    ].join(NL),
    Object.assign(ctx, {{ __setStateReader: (fn) => {{ ctx.__readState = fn; }} }})
  );
  assert.ok(createdWs, 'Expected FakeWs instance');
  await vm.runInContext('shutdownApplication()', ctx);
  const st = ctx.__readState();
  assert.equal(st.appShuttingDown, true);
  assert.equal(st.shutdownInProgress, false);
  assert.equal(st.activeJobId, null);
  assert.equal(overlayEl.style.display, 'flex');
  assert.equal(btnEl.textContent, '⏻ تم الإيقاف');
  assert.deepEqual(actions, ['cleanup', 'close_window']);

  // Part B: negative mutant reproducing the v3 stillAlive=false fallthrough bug must fail
  const mutantFn = `async function shutdownApplication() {{
    let stillAlive = false;
    try {{ await fetch('/api/shutdown'); }} catch {{
      try {{ const r = await fetch('/api/health'); if (r && r.ok) stillAlive = true; }} catch {{}}
      if (stillAlive) return;
    }}
    appShuttingDown = true;
    activeJobId = null;
    showShutdownOverlay();
  }}`;
  const mCtx = vm.createContext({{
    fetch: async () => {{ throw new Error('offline'); }},
    appShuttingDown: false,
    activeJobId: 'job-alive',
    showShutdownOverlay: () => {{ mCtx.overlayShown = true; }},
    overlayShown: false,
  }});
  await vm.runInContext([mutantFn, 'shutdownApplication()'].join(NL), mCtx);
  assert.equal(mCtx.appShuttingDown, true, 'Mutant reproduces false-positive shutdown on dual reject');
  assert.equal(mCtx.overlayShown, true);
  console.log(JSON.stringify({{ integrated_ws_ok: true, mutant_reproduced: true }}));
}})().catch((err) => {{ console.error(err); process.exitCode = 1; }});
"""
        res = self._run_node(script)
        self.assertTrue(res["integrated_ws_ok"])
        self.assertTrue(res["mutant_reproduced"])

if __name__ == "__main__":
    unittest.main()
