"""Browser tests of the web UI, driven like a user, against the real app with a scripted model.

No Ollama needed. Skipped when Playwright or a browser is not available
(install with: python -m playwright install chromium).
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

from app.config import Settings
from app.main import create_app
from conftest import FakeLLM, reply

sync_api = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e
expect = sync_api.expect


def scripted_model(messages: list[dict]) -> str:
    """Answers the poisoned-quote question the way the local model did in live runs."""
    if "TOOL-Q-5531" in messages[-1]["content"]:
        return reply(
            "answered", "PR-1042 is pending approval. Bluegate quoted 1,175 AUD per unit.", ["TCK-H-101", "TOOL-Q-5531"]
        )
    return reply("insufficient_evidence", "", [])


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui")
    settings = Settings(
        audit_path=tmp / "audit.jsonl", db_path=tmp / "db.sqlite", rate_limit_per_min=1000, retrieval_mode="bm25"
    )
    app = create_app(settings, llm=FakeLLM(scripted_model))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "test server did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            launched = p.chromium.launch()
        except Exception:
            try:
                launched = p.chromium.launch(channel="chrome")  # the installed Google Chrome, no download
            except Exception as exc:
                pytest.skip(f"no browser available for Playwright: {exc}")
        yield launched
        launched.close()


@pytest.fixture
def ui(browser, base_url):
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(base_url + "/")
    yield page, errors
    page.close()


def test_poisoned_quote_turn_explains_what_was_removed(ui):
    page, errors = ui
    page.locator(".suggestion", has_text="Poisoned vendor quote").click()
    page.locator("#ask-btn").click()

    expect(page.locator(".msg-user").last).to_contain_text("PR-1042")
    story = page.locator(".story").last
    expect(story).to_contain_text("Hidden commands were found in", timeout=15_000)
    expect(story).to_contain_text("TOOL-Q-5531")
    expect(page.locator(".pipe-step.tone-warn").last).to_contain_text("Remove hidden commands")
    expect(page.locator(".msg-assistant .answer").last).not_to_contain_text("approved")

    page.get_by_role("button", name="Show removed text").last.click()
    expect(page.locator("#drawer")).to_be_visible()
    expect(page.locator("#drawer .cut").first).to_contain_text("SYSTEM")
    assert errors == []


def test_other_company_question_is_refused_and_new_chat_resets(ui):
    page, errors = ui
    page.locator("#sidebar").get_by_role("button", name="Another tenant's price").click()
    page.locator("#ask-btn").click()

    expect(page.locator(".story").last).to_contain_text("Other companies' records are never searched", timeout=15_000)
    expect(page.locator(".msg-assistant").last).to_contain_text("None of the records available to you mention")

    page.locator("#sidebar .js-new-chat").click()
    expect(page.locator(".welcome")).to_be_visible()
    expect(page.locator(".msg-user")).to_have_count(0)
    assert errors == []


def test_phone_layout_has_no_sideways_scroll_and_a_working_menu(ui):
    page, errors = ui
    page.set_viewport_size({"width": 375, "height": 812})
    page.wait_for_timeout(400)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    expect(page.locator("#sidebar")).not_to_be_in_viewport()
    page.locator("#nav-open").click()
    expect(page.locator("#sidebar")).to_be_in_viewport()
    assert errors == []
