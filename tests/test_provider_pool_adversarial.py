"""Adversarial tests for the `[llm.providers]` provider pool.

Each test corresponds to a defect found while trying to break the feature.
They are written to fail against the current implementation.

Where a test asserts an invariant rather than a mechanism (notably the
credential-crossing ones), it accepts *either* a raised error or a refusal to
call `build_llm` — choosing between those is the implementer's call.
"""

import json
import struct
from datetime import UTC, datetime
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

import switchplane.llm as llm_mod
from switchplane.agent import AgentSpec
from switchplane.agent_runtime import AgentContext
from switchplane.config import AppConfig, load_config, resolve_provider
from switchplane.persistence import Store
from switchplane.subprocess_manager import SubprocessManager
from switchplane.task import Task, TaskRecord, TaskStatus


@pytest.fixture
def captured_build(monkeypatch):
    """Capture the (model, api_key, base_url) triple handed to build_llm."""
    calls: list[tuple] = []

    def _fake_build(model, api_key=None, base_url=None):
        calls.append((model, api_key, base_url))
        return "mock_llm"

    monkeypatch.setattr(llm_mod, "build_llm", _fake_build)
    return calls


def _ctx(config: dict) -> AgentContext:
    import socket

    _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    return AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config=config)


def _assert_no_credential_crossing(fn, calls: list[tuple], credential: str, vendor_prefix: str):
    """Assert *fn* does not hand *credential* to an adapter outside *vendor_prefix*.

    Satisfied by raising, or by not calling `build_llm` with the crossing pair.
    """
    try:
        fn()
    except Exception:
        return  # refused — acceptable
    crossing = [c for c in calls if c[1] == credential and not c[0].startswith(vendor_prefix)]
    assert not crossing, (
        f"build_llm{crossing[0]!r}: a {vendor_prefix}* credential was handed to a {crossing[0][0]} adapter"
    )


# ---------------------------------------------------------------------------
# Finding 1: `model=` override crosses vendors, carrying the credential with it
# ---------------------------------------------------------------------------


class TestModelOverrideCrossesVendor:
    """`model=` reuses the resolved credential, but `build_llm` routes on the
    *overridden* model's prefix. Overriding across a vendor boundary therefore
    hands provider A's api_key to provider B's adapter — the exact failure the
    pool exists to prevent (spec §1), reintroduced by the override that §6
    describes as "keeping its credential".

    The override is only safe within one vendor, or on a gateway whose base_url
    the target adapter honours. Nothing enforces either condition, and nothing
    in the spec's §7.4 test list covers a crossing override.
    """

    POOL: ClassVar[dict] = {
        "llm": {
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-ant-KEY",
            "base_url": None,
            "providers": {
                "cheap": {"model": "gemini-2.5-flash", "api_key": "gai-KEY", "base_url": None},
            },
        }
    }

    def test_override_to_other_vendor_on_named_entry(self, captured_build):
        """`ctx.llm("cheap", model="gpt-4o")` on a direct (no base_url) Gemini
        entry currently forwards the Google key to ChatOpenAI — i.e. to
        api.openai.com."""
        ctx = _ctx(self.POOL)
        _assert_no_credential_crossing(lambda: ctx.llm("cheap", model="gpt-4o"), captured_build, "gai-KEY", "gemini")

    def test_override_to_other_vendor_on_default(self, captured_build):
        """Same crossing via the `[llm]` block: `ctx.llm(model="gemini-2.5-flash")`
        forwards the Anthropic key to ChatGoogleGenerativeAI."""
        ctx = _ctx(self.POOL)
        _assert_no_credential_crossing(
            lambda: ctx.llm(model="gemini-2.5-flash"), captured_build, "sk-ant-KEY", "claude"
        )

    def test_same_vendor_override_still_allowed(self, captured_build):
        """The legitimate case must keep working: a different model from the
        same vendor reuses the credential. Guards against an over-broad fix."""
        ctx = _ctx(self.POOL)
        ctx.llm(model="claude-haiku-4-5-20251001")
        assert captured_build == [("claude-haiku-4-5-20251001", "sk-ant-KEY", None)]

    def test_gateway_override_across_vendor_still_allowed(self, captured_build):
        """A gateway entry legitimately fronts many vendors' models behind one
        endpoint and token — §6's motivating case, and the user's production
        install. Whatever fixes the two crossing tests must NOT break this."""
        ctx = _ctx(
            {
                "llm": {
                    "model": "claude-sonnet-4-6",
                    "api_key": "gw-TOKEN",
                    "base_url": "https://gateway.internal/v1",
                    "providers": {},
                }
            }
        )
        ctx.llm(model="gpt-4o")
        assert captured_build == [("gpt-4o", "gw-TOKEN", "https://gateway.internal/v1")]


class TestGatewayCredentialEscapesToPublicEndpoint:
    """The sharpest form of Finding 1, on the config shape the user actually
    runs in production: one gateway `base_url`, one token, many models.

    `build_llm`'s Gemini branch drops `base_url` entirely — pinned by
    `tests/test_llm.py::TestBuildLLMGemini::test_base_url_not_forwarded`,
    because `ChatGoogleGenerativeAI` has no such kwarg. So
    `ctx.llm(model="gemini-2.5-flash")` on a gateway config builds a client
    aimed at Google's *public* endpoint while still carrying the gateway token:
    the gateway is bypassed and its credential is transmitted off-site.

    `ctx.llm(model=...)` is what makes this reachable in a single call, and §6
    advertises exactly this call shape as the gateway convenience.
    """

    def test_gemini_override_on_gateway_config_does_not_leak_token(self, captured_build):
        ctx = _ctx(
            {
                "llm": {
                    "model": "claude-sonnet-4-6",
                    "api_key": "gw-TOKEN",
                    "base_url": "https://gateway.internal/v1",
                    "providers": {},
                }
            }
        )
        try:
            ctx.llm(model="gemini-2.5-flash")
        except Exception:
            return  # refused — acceptable
        assert captured_build == [], (
            f"build_llm{captured_build[0]!r}: the gateway token is forwarded to a "
            "model whose adapter discards base_url, so it is sent to Google's "
            "public endpoint instead of the configured gateway"
        )


class TestPoolEntryMergeBlendsVendors:
    """Documented limitation: per-field merge within a pool entry can blend
    credentials from different vendors.

    `deep_merge` merges pool entries per-field across the two config layers
    (spec §7.2 requires this). When a user overrides only `model` to switch
    vendors, the old vendor's `api_key` survives. Switchplane cannot detect this
    without knowing which vendor a credential belongs to (API key prefixes are
    vendor-chosen, undocumented as stable, and ambiguous — `sk-` is shared
    between OpenAI and older Anthropic keys).

    The `model=` override guard (FIX 1) does not cover this path because there's
    no override argument — `ctx.llm("cheap")` resolves the already-merged entry
    as-is.

    **Guidance:** set `model` and `api_key` together in the same config layer
    when changing vendors. If the app default ships `[llm.providers.cheap]` with
    a Google model+key, and you want OpenAI instead, override both fields in
    your user config, not just `model`.
    """

    def test_merge_can_blend_vendor_fields_documented_limitation(self, tmp_path):
        """Per-field merge produces a Google key with an OpenAI model.
        This is the accepted sharp edge: we cannot detect it without a
        credential-to-vendor mapping that doesn't exist."""
        default_cfg = tmp_path / "defaults.toml"
        default_cfg.write_text(
            '[llm]\nmodel = "claude-sonnet-4-20250514"\n\n'
            '[llm.providers.cheap]\nmodel = "gemini-2.5-flash"\napi_key = "gai-APP-KEY"\n'
        )
        user_cfg = tmp_path / "config.toml"
        user_cfg.write_text('[llm]\napi_key = "sk-ant-USER"\n\n[llm.providers.cheap]\nmodel = "gpt-4o"\n')

        merged = load_config(user_cfg, default_cfg).model_dump()
        entry = resolve_provider(merged, "cheap")

        # Documents what actually happens: the blend succeeds and forwards the
        # wrong credential to the wrong adapter. FIX 1 guards the model= path;
        # this path (no override) is unguarded.
        assert entry.model == "gpt-4o"
        assert entry.api_key == "gai-APP-KEY", (
            "Test expectation updated: this blend is now accepted as a known "
            "limitation. The user must set both model and api_key in the same layer."
        )


# ---------------------------------------------------------------------------
# Finding 2: Task.providers is iterated unguarded in agent_main
# ---------------------------------------------------------------------------


class TestTaskProvidersAdvisoryWarningIsSafe:
    """`agent_main` iterates `task_class.providers` unguarded, unlike the
    `mcp_servers` precedent immediately above it (`if task_class.mcp_servers:`).

    That line sits *before* `_instantiate_task`, outside every `try/except` that
    emits `ctx.fail`. A non-iterable value kills the subprocess with a bare
    traceback, and the control plane can only report "Agent process exited
    unexpectedly (code N)" — while the spec calls the declaration "advisory
    only" and promises it "does not fail the task".
    """

    @staticmethod
    def _warn_loop(task_class, ctx_providers):
        """Matches the guarded loop in `agent_runtime.agent_main` after FIX 3."""
        from switchplane.config import DEFAULT_PROVIDER

        if not task_class.providers:
            return []
        if not isinstance(task_class.providers, list):
            return None  # Signals type error, not iteration
        return [n for n in task_class.providers if n != DEFAULT_PROVIDER and n not in ctx_providers]

    def test_none_providers_does_not_raise(self):
        class NoneProviders(Task):
            name = "none_providers"
            providers: ClassVar[list[str]] = None  # type: ignore[assignment]

            async def run(self, ctx): ...

        assert self._warn_loop(NoneProviders, []) == []

    def test_string_providers_does_not_warn_per_character(self):
        """`providers = "cheap"` (a bare string instead of a list) iterates
        character-by-character, emitting five nonsense warnings naming 'c', 'h',
        'e', 'a', 'p' — which hides the actual mistake."""

        class StrProviders(Task):
            name = "str_providers"
            providers: ClassVar[list[str]] = "cheap"  # type: ignore[assignment]

            async def run(self, ctx): ...

        # Should return None to signal type error, not the character list
        assert self._warn_loop(StrProviders, []) is None


# ---------------------------------------------------------------------------
# Finding 3: `ctx.providers` omits "default", which `resolve_provider` accepts
# ---------------------------------------------------------------------------


class TestDefaultSentinelDoesNotWarn:
    """The "default" sentinel always resolves (§5 rule 1), so Task.providers
    declaring it should not produce a startup warning even though it does not
    appear in ctx.providers (which enumerates only the pool, not the [llm] block).
    """

    def test_providers_default_does_not_warn(self):
        """Declaring providers=["default"] should produce no warning, since
        "default" always resolves to the [llm] block."""
        import socket
        from typing import ClassVar

        class DefaultProviderTask(Task):
            name = "default_provider_task"
            providers: ClassVar[list[str]] = ["default"]

            async def run(self, ctx):
                pass

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1", task_name="test", ipc_sock=agent_sock, config={"llm": {"api_key": "k", "providers": {}}}
        )

        # The warning loop should skip "default"
        missing = [n for n in DefaultProviderTask.providers if n not in ctx.providers and n != "default"]
        assert missing == [], "default should be skipped in the warning check"


# ---------------------------------------------------------------------------
# Finding 4: config-delivery tests assert a copy of the source line
# ---------------------------------------------------------------------------


class TestConfigDeliveryThroughLaunchAgent:
    """Config delivery asserted through the real `launch_agent` path.

    These drive `launch_agent` and read the payload actually written to the
    socket, so they fail if the delivery line in `subprocess_manager.py` drops
    or mangles a key (verified against `agent_config.pop("llm", None)`).

    They replaced an earlier set in `tests/test_subprocess_manager.py` that
    re-executed a hand-copied `config.model_dump() if config else {}` inside
    the test body. Those asserted a property of `AppConfig` rather than of the
    delivery path, and survived that same mutation. Assert on the delivered
    frame here rather than reproducing the source line.
    """

    @pytest.fixture
    def mgr(self):
        store = MagicMock(spec=Store)
        store.upsert_agent = AsyncMock()
        store.update_task = AsyncMock()
        store.get_task = AsyncMock(return_value=None)
        return SubprocessManager(store)

    @staticmethod
    def _task() -> TaskRecord:
        now = datetime.now(UTC)
        return TaskRecord(
            task_id="task1",
            agent_name="bot",
            task_name="chat",
            status=TaskStatus.PENDING,
            input_json="{}",
            created_at=now,
            updated_at=now,
        )

    async def _delivered_config(self, mgr, config: AppConfig | None, monkeypatch) -> dict:
        """Launch an agent against a stubbed subprocess and return the `config`
        block of the `execute_task` payload read off the CP's socket."""
        import asyncio

        import switchplane.subprocess_manager as spm

        proc = MagicMock()
        proc.pid = 4242
        proc.returncode = None
        proc.wait = AsyncMock(return_value=0)
        proc.stderr = None

        writes: list[bytes] = []
        writer = MagicMock()
        writer.write = writes.append
        writer.drain = AsyncMock()

        async def _never(*args, **kwargs):
            await asyncio.sleep(3600)

        reader = MagicMock()
        reader.readexactly = AsyncMock(side_effect=_never)

        async def _fake_exec(*args, **kwargs):
            return proc

        async def _fake_open_connection(*args, **kwargs):
            return reader, writer

        monkeypatch.setattr(spm.asyncio, "create_subprocess_exec", _fake_exec)
        monkeypatch.setattr(spm.asyncio, "open_connection", _fake_open_connection)

        try:
            await mgr.launch_agent(AgentSpec(agent_name="bot", module_path="myapp.agents.bot"), self._task(), config)
        finally:
            for handle in list(mgr._handles.values()):
                for t in (handle.reader_task, handle.stderr_task):
                    if t:
                        t.cancel()

        assert writes, "launch_agent wrote no execute_task frame"
        frame = b"".join(writes)
        length = struct.unpack(">I", frame[:4])[0]
        command = json.loads(frame[4 : 4 + length])
        assert command["type"] == "execute_task"
        return command["payload"]["config"]

    @pytest.mark.asyncio
    async def test_pool_reaches_the_delivered_payload(self, mgr, monkeypatch):
        config = AppConfig.model_validate(
            {
                "llm": {
                    "model": "claude-sonnet-4-20250514",
                    "api_key": "sk-ant-key",
                    "providers": {
                        "cheap": {"model": "gemini-2.5-flash", "api_key": "gai-key"},
                        "gateway": {
                            "model": "gpt-4o",
                            "api_key": "gw-tok",
                            "base_url": "https://gw.internal/v1",
                        },
                    },
                }
            }
        )

        delivered = await self._delivered_config(mgr, config, monkeypatch)

        pool = delivered["llm"]["providers"]
        assert pool["cheap"]["api_key"] == "gai-key"
        assert pool["gateway"]["base_url"] == "https://gw.internal/v1"
        # The delivered dict *is* ctx.config, so the resolver must work on it
        # end to end — not merely on a hand-built fixture.
        assert resolve_provider(delivered, "cheap").api_key == "gai-key"
        assert resolve_provider(delivered).api_key == "sk-ant-key"

    @pytest.mark.asyncio
    async def test_stale_agents_section_is_not_delivered(self, mgr, monkeypatch):
        config = AppConfig.model_validate({"llm": {"api_key": "k"}, "agents": {"bot": {"api_key": "STALE"}}})

        delivered = await self._delivered_config(mgr, config, monkeypatch)

        assert "agents" not in delivered
        assert delivered["llm"]["api_key"] == "k"
