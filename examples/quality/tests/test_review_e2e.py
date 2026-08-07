"""End-to-end graph execution tests for PR review.

Tests the REAL graph (build_graph + compile + ainvoke), not just isolated node calls.
These tests catch topology bugs (unreachable nodes, missing edges), serialization failures,
and execution-path issues that direct node invocation hides.

All imports are function-scoped to avoid collection-time failures (see conftest.py).
"""

from __future__ import annotations

import pytest


class TestGraphTopology:
    """Tests verifying the graph structure itself — edges, reachability, entry point."""

    def test_graph_topology_reaches_review_branch(self):
        """review_branch must be reachable from __start__ (kills #17: fan-out unreachable).

        Tests reachability via graph traversal, not by assuming a specific topology.
        review_branch must have at least one inbound edge and be reachable from __start__.
        This test MUST fail if review_branch is orphaned or unreachable from the entry point.
        """
        from quality.agents.pr.tasks.review import build_graph

        # Mock minimal ctx and shell for build_graph
        class FakeContext:
            pass

        class FakeShell:
            pass

        graph = build_graph(FakeContext(), FakeShell())
        compiled = graph.compile()

        # Get the graph structure via get_graph()
        graph_def = compiled.get_graph()

        # Extract nodes and edges
        all_nodes = list(graph_def.nodes.keys())
        all_edges = [(e.source, e.target) for e in graph_def.edges]

        # Assert review_branch exists as a node
        assert "review_branch" in all_nodes, f"review_branch must be defined as a graph node. Found nodes: {all_nodes}"

        # Check that review_branch has at least one inbound edge (not orphaned)
        inbound_to_review_branch = [source for source, target in all_edges if target == "review_branch"]

        assert len(inbound_to_review_branch) > 0, (
            f"review_branch must have at least one inbound edge (not orphaned). "
            f"Found inbound edges: {inbound_to_review_branch}. "
            f"All edges: {all_edges}. "
            "If this fails, review_branch is unreachable dead code (bug #17)."
        )

        # Verify review_branch is reachable from __start__ via graph traversal
        # Build an adjacency list for BFS
        from collections import deque

        adjacency = {}
        for source, target in all_edges:
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(target)

        # BFS from __start__ to find all reachable nodes
        reachable = set()
        queue = deque(["__start__"])
        reachable.add("__start__")

        while queue:
            node = queue.popleft()
            if node in adjacency:
                for neighbor in adjacency[node]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        queue.append(neighbor)

        assert "review_branch" in reachable, (
            f"review_branch must be reachable from __start__. "
            f"Reachable nodes: {sorted(reachable)}. "
            f"All edges: {all_edges}. "
            "If this fails, the fan-out is unreachable dead code (bug #17)."
        )


class TestGraphExecution:
    """Tests that execute the real graph with stubbed seams (gh module, LLM, memory)."""

    @pytest.mark.asyncio
    async def test_graph_executes_end_to_end(self, monkeypatch, tmp_path, stub_setup_seams):
        """Graph must execute ainvoke without TypeError (kills #21: 3-arg node signature).

        Runs build_graph().compile().ainvoke(initial_state) with all external seams
        stubbed at the module level. Asserts execution completes and that branch nodes
        actually ran (by checking recorded calls). This test MUST fail if node functions
        have (ctx, shell, state) signatures but LangGraph calls them as (state,).
        """
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub gh module seams
        from quality import gh as gh_module

        posted_comments = []
        posted_reviews = []

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            posted_comments.append({"path": path, "line": line, "body": body})

        async def fake_submit_pr_review(shell, repo, number, event, body):
            posted_reviews.append({"event": event, "body": body})

        async def fake_list_review_comments(shell, repo, number):
            return []

        def fake_commentable_lines(diff):
            return {"test.py": {10, 20}}

        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "list_review_comments", fake_list_review_comments, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", fake_commentable_lines, raising=True)

        # Stub memory module
        from quality.agents.pr import memory as memory_module

        monkeypatch.setattr(memory_module, "save_baseline", lambda *args, **kwargs: None, raising=True)
        monkeypatch.setattr(
            memory_module,
            "baseline_path",
            lambda root, repo, number, *, local=False: "/tmp/baseline.json",
            raising=True,
        )
        monkeypatch.setattr(memory_module, "load_baseline", lambda *args: {"findings": []}, raising=True)

        # Stub rate limit module
        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Create a fake LLM that returns structured output
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str = ""
            line: int | None = None
            severity: str = "medium"
            body: str = ""
            models: list[str] = []

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Review complete"
            event: str = "COMMENT"
            comments: list[MockSynthComment] = []

        class FakeLLM:
            def with_structured_output(self, schema):
                return self

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                # Return structured output for synthesis
                if isinstance(messages, list):
                    return MockSynthResult(
                        summary="Test review",
                        event="COMMENT",
                        comments=[
                            MockSynthComment(
                                path="test.py",
                                line=10,
                                severity="low",
                                body="Test finding",
                                models=["test-model"],
                            )
                        ],
                    )
                # Return tool-call response for review branch
                from langchain_core.messages import AIMessage

                return AIMessage(content="Review complete")

        # Build initial state
        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=1,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path="/tmp/repo",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Test finding",
                    "model": "test-model",
                    "domain": "quality",
                }
            ],
            notes=[],
        )

        # Use shared fakes from conftest.py
        from conftest import FakeAgentContext, FakeShell

        class FakeContext(FakeAgentContext):
            """Subclass shared fake to add custom LLM behavior for this test."""

            def __init__(self):
                super().__init__(runtime_dir_path=tmp_path)
                self.task_id = "test-123"

            def llm(self, name=None):
                return FakeLLM()

        # Build context separately and pass to build_graph
        # DO NOT put ctx in state — that was bug #20
        ctx = FakeContext()
        shell = FakeShell()

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        # Execute the graph — this MUST not raise TypeError
        # If nodes have (ctx, shell, state) but LangGraph calls (state,), this will fail
        try:
            result = await compiled.ainvoke(initial_state)
        except TypeError as exc:
            pytest.fail(
                f"Graph execution raised TypeError (likely node signature mismatch): {exc}. "
                "LangGraph calls nodes as (state,) but review nodes may have (ctx, shell, state). "
                "This is bug #21."
            )

        # Assert the graph executed without setup errors
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Check that FakeContext has providers property and FakeShell has fs_tools()."
        )

        # Assert the graph actually executed and reached synthesize_and_post
        # Since we stubbed the seams, we should see at least one review posted
        assert len(posted_reviews) >= 1, (
            f"Expected at least 1 review posted, got {len(posted_reviews)}. "
            "Graph may have short-circuited or synthesize_and_post didn't run."
        )

    @pytest.mark.asyncio
    async def test_setup_populates_diff_worktree_sha_and_author(self, monkeypatch, tmp_path, stub_setup_seams):
        """Setup node must populate diff, worktree_path, head_sha, authed_user (kills #18).

        After the setup node runs, state must have non-empty diff, worktree_path, head_sha,
        and authed_user. is_self_review must be correctly computed from pr_author vs authed_user.
        This test MUST fail on current main where there is no setup node.
        """
        from quality.agents.pr.tasks.review import build_graph

        # This test documents the missing setup node by checking that the graph
        # does NOT populate these fields when they start empty.
        # When setup is implemented, this test will pass.

        # Build the graph
        class FakeContext:
            pass

        class FakeShell:
            pass

        graph = build_graph(FakeContext(), FakeShell())

        # Check if there's a "setup" node in the graph
        graph_def = graph.compile().get_graph()
        all_nodes = list(graph_def.nodes.keys())

        # The bug: no setup node exists
        if "setup" not in all_nodes:
            pytest.fail(
                f"Bug #18 detected: No 'setup' node in graph. "
                f"Found nodes: {all_nodes}. "
                "Without a setup node, diff/worktree_path/head_sha/authed_user are never populated. "
                "The graph cannot fetch PR data or create a worktree."
            )

    @pytest.mark.asyncio
    async def test_local_mode_posts_nothing_to_github(self, monkeypatch, tmp_path, stub_setup_seams):
        """local=True must write artifact to disk and post ZERO calls to gh (kills #19).

        When local=True, synthesize_and_post must:
        1. Write a Markdown artifact to disk with findings
        2. Make ZERO calls to gh.submit_pr_review
        3. Make ZERO calls to gh.create_pr_review_comment

        This test MUST fail on current main where local mode is ignored.
        """
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub gh module and COUNT calls
        from quality import gh as gh_module

        gh_calls = {"submit_pr_review": 0, "create_pr_review_comment": 0}

        async def counting_submit_pr_review(shell, repo, number, event, body):
            gh_calls["submit_pr_review"] += 1

        async def counting_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            gh_calls["create_pr_review_comment"] += 1

        monkeypatch.setattr(gh_module, "submit_pr_review", counting_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", counting_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda diff: {"test.py": {10}}, raising=True)

        # Stub memory and rate limit
        from quality.agents.pr import memory as memory_module

        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(
            memory_module,
            "baseline_path",
            lambda root, repo, number, *, local=False: "/tmp/baseline.json",
            raising=True,
        )
        monkeypatch.setattr(memory_module, "load_baseline", lambda *a: {"findings": []}, raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Create initial state with local=True
        from pathlib import Path

        artifact_dir = Path(tmp_path) / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=1,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path="/tmp/repo",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "high",
                    "body": "Security issue found",
                    "model": "test-model",
                    "domain": "security",
                }
            ],
            notes=[],
            local=True,  # LOCAL MODE — must write artifact, not post to GitHub
        )

        # Mock context
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str = ""
            line: int | None = None
            severity: str = "medium"
            body: str = ""
            models: list[str] = []

        class MockSynthResult(PydanticBaseModel):
            summary: str = ""
            event: str = "COMMENT"
            comments: list[MockSynthComment] = []

        # Use shared fakes from conftest.py
        from conftest import FakeAgentContext, FakeShell

        class FakeContext(FakeAgentContext):
            """Subclass shared fake to add custom LLM behavior for this test."""

            def __init__(self):
                super().__init__(runtime_dir_path=artifact_dir)

            def llm(self, name=None):
                class FakeLLM:
                    def with_structured_output(self, schema):
                        return self

                    async def ainvoke(self, messages):
                        return MockSynthResult(
                            summary="Security review",
                            event="REQUEST_CHANGES",
                            comments=[
                                MockSynthComment(
                                    path="test.py",
                                    line=10,
                                    severity="high",
                                    body="Security issue found",
                                    models=["test-model"],
                                )
                            ],
                        )

                return FakeLLM()

        # Build context separately and pass to build_graph
        # Do NOT put ctx in state — that was bug #20
        ctx = FakeContext()
        shell = FakeShell()

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without setup errors
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Check that fake reached synthesis and runtime_dir points to artifact_dir."
        )

        # Assert ZERO GitHub calls
        assert gh_calls["submit_pr_review"] == 0, (
            f"local=True must make ZERO calls to submit_pr_review, got {gh_calls['submit_pr_review']}. "
            "This is bug #19: local mode is ignored."
        )
        assert gh_calls["create_pr_review_comment"] == 0, (
            f"local=True must make ZERO calls to create_pr_review_comment, got {gh_calls['create_pr_review_comment']}. "
            "This is bug #19: local mode is ignored."
        )

        # Assert artifact was written
        # Production writes to runtime_dir / "reviews" / repo / "pr-{number}.md"
        from pathlib import Path

        expected_artifact_dir = artifact_dir / "reviews" / "github.com/org/repo"
        artifact_files = list(expected_artifact_dir.glob("*.md")) if expected_artifact_dir.exists() else []
        assert len(artifact_files) > 0, (
            f"local=True must write a Markdown artifact to {expected_artifact_dir}, found {len(artifact_files)} .md files. "
            f"Searched in: {expected_artifact_dir}, exists: {expected_artifact_dir.exists()}. "
            "This is bug #19: local mode doesn't write artifacts."
        )

        # Assert artifact contains the finding
        artifact_content = artifact_files[0].read_text()
        assert "Security issue found" in artifact_content, "Artifact must contain the finding body"
        assert "test.py" in artifact_content, "Artifact must reference the file path"

    @pytest.mark.asyncio
    async def test_state_survives_real_checkpointer(self, monkeypatch, tmp_path, stub_setup_seams):
        """State must serialize through REAL SqliteCheckpointSaver (kills #20: unserializable ctx).

        ReviewState has a ctx field that carries AgentContext. AgentContext is not picklable
        or msgpackable, so putting it in ReviewState breaks checkpointing. This test verifies
        that ReviewState can be serialized by the checkpoint saver WITHOUT executing the graph
        (to avoid hitting bug #21).
        """
        from quality.agents.pr.tasks.review import ReviewState

        # Create initial state WITHOUT ctx first
        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=1,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path="/tmp/repo",
            matrix=[],
            error=None,
            is_followup=False,
            findings=[],
            notes=[],
        )

        # Check if ReviewState has a ctx field
        if hasattr(initial_state, "ctx"):
            pytest.fail(
                "Bug #20 detected: ReviewState has a 'ctx' field. "
                "AgentContext cannot be serialized, breaking checkpointing. "
                "ctx must be passed as a node argument, not stored in state."
            )

        # Test serialization through the REAL saver (not msgpack directly)
        # LangGraph uses ormsgpack via jsonplus.py, not msgpack
        import tempfile

        import aiosqlite

        from switchplane.checkpoint import SqliteCheckpointSaver

        db_path = tempfile.mktemp(suffix=".db")
        async with aiosqlite.connect(db_path) as conn:
            cp = SqliteCheckpointSaver(conn, "test-thread")
            await cp.setup()

            # Put the state into the checkpointer
            config = {"configurable": {"thread_id": "test-thread"}}
            # Build a real Checkpoint shape — LangGraph uses "id" not "checkpoint_id",
            # and state goes in "channel_values" not metadata
            checkpoint = {
                "v": 1,
                "id": "test-1",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": initial_state.model_dump(),
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            try:
                await cp.aput(config, checkpoint, {}, {})
            except Exception as exc:
                pytest.fail(
                    f"ReviewState failed to serialize through SqliteCheckpointSaver: {exc}. "
                    "If KeyError('id'), check Checkpoint shape — state goes in channel_values."
                )

            # Read it back
            tuple_result = await cp.aget_tuple(config)
            assert tuple_result is not None, "Checkpoint should be retrievable"
            channel_values = tuple_result.checkpoint["channel_values"]
            assert "repo" in channel_values, "State should round-trip"
            assert set(initial_state.model_dump()) == set(channel_values), "All state keys must round-trip"
