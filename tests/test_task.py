from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic import Field

from switchplane.task import (
    Task,
    TaskRecord,
    TaskStatus,
    _build_command_model,
    _coerce_command_params,
    command,
)


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.INTERRUPTED == "interrupted"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"

    def test_from_string(self):
        assert TaskStatus("pending") == TaskStatus.PENDING
        assert TaskStatus("running") == TaskStatus.RUNNING
        assert TaskStatus("interrupted") == TaskStatus.INTERRUPTED


class TestTaskRecord:
    def test_defaults(self):
        now = datetime.now(UTC)
        rec = TaskRecord(
            task_id="t1",
            agent_name="worker",
            task_name="hello",
            created_at=now,
            updated_at=now,
        )
        assert rec.status == TaskStatus.PENDING
        assert rec.input_json == "{}"
        assert rec.result_json is None
        assert rec.error_json is None

    def test_full_record(self):
        now = datetime.now(UTC)
        rec = TaskRecord(
            task_id="t1",
            agent_name="worker",
            task_name="hello",
            status=TaskStatus.COMPLETED,
            input_json='{"name": "Alice"}',
            result_json='{"greeting": "Hello Alice"}',
            created_at=now,
            updated_at=now,
        )
        assert rec.status == TaskStatus.COMPLETED
        assert rec.result_json == '{"greeting": "Hello Alice"}'

    def test_serialization_round_trip(self):
        now = datetime.now(UTC)
        rec = TaskRecord(
            task_id="t1",
            agent_name="a",
            task_name="t",
            created_at=now,
            updated_at=now,
        )
        data = rec.model_dump_json()
        rec2 = TaskRecord.model_validate_json(data)
        assert rec2.task_id == rec.task_id
        assert rec2.status == rec.status


class TestCommandDecorator:
    def test_marks_function(self):
        @command
        def my_handler(self, ctx):
            pass

        assert my_handler._is_command is True

    def test_preserves_function(self):
        @command
        def my_handler(self, ctx, x: int = 5):
            return x

        assert my_handler(None, None) == 5


class TestParametersModel:
    def test_no_parameters(self):
        class SimpleTask(Task):
            name = "simple"

            async def run(self, ctx):
                pass

        assert SimpleTask.parameters_model() is None

    def test_with_parameters(self):
        class ParamTask(Task):
            name = "param"
            greeting: str = Field(description="The greeting")
            count: int = Field(default=1, description="Repeat count")

            async def run(self, ctx):
                pass

        model = ParamTask.parameters_model()
        assert model is not None
        assert "greeting" in model.model_fields
        assert "count" in model.model_fields
        assert model.model_fields["count"].default == 1

    def test_validates_params(self):
        class ParamTask(Task):
            name = "param"
            value: int = Field(description="A number")

            async def run(self, ctx):
                pass

        model = ParamTask.parameters_model()
        validated = model.model_validate({"value": 42})
        assert validated.value == 42

    def test_ignores_base_attrs(self):
        class MyTask(Task):
            name = "my"
            description = "desc"
            mode = "ephemeral"
            custom: str = Field(description="custom param")

            async def run(self, ctx):
                pass

        model = MyTask.parameters_model()
        assert "custom" in model.model_fields
        assert "name" not in model.model_fields
        assert "description" not in model.model_fields
        assert "mode" not in model.model_fields


class TestBuildCommandModel:
    def test_no_params(self):
        def handler(self, ctx):
            pass

        model = _build_command_model(handler)
        assert model is None

    def test_with_typed_params(self):
        def handler(self, ctx, count: int, name: str = "default"):
            pass

        model = _build_command_model(handler)
        assert model is not None
        assert "count" in model.model_fields
        assert "name" in model.model_fields

    def test_skips_self_and_ctx(self):
        def handler(self, ctx, x: int):
            pass

        model = _build_command_model(handler)
        assert "self" not in model.model_fields
        assert "ctx" not in model.model_fields
        assert "x" in model.model_fields


class TestCoerceCommandParams:
    def test_string_to_int(self):
        def handler(self, ctx, count: int):
            pass

        result = _coerce_command_params(handler, {"count": "42"})
        assert result == {"count": 42}

    def test_no_params(self):
        def handler(self, ctx):
            pass

        result = _coerce_command_params(handler, {})
        assert result == {}

    def test_caches_model(self):
        def handler(self, ctx, x: int):
            pass

        _coerce_command_params(handler, {"x": "1"})
        assert hasattr(handler, "_command_model")
        _coerce_command_params(handler, {"x": "2"})


class TestProcessCommands:
    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.poll_command = MagicMock(
            side_effect=[
                {"action": "set_value", "params": {"count": "10"}},
                None,
            ]
        )
        ctx.command_result = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_dispatches_known_command(self, mock_ctx):
        class MyTask(Task):
            name = "test"

            async def run(self, ctx):
                pass

            @command
            def set_value(self, ctx, count: int):
                return {"new_count": count}

        task = MyTask()
        await task.process_commands(mock_ctx)
        mock_ctx.command_result.assert_called_once_with("set_value", {"new_count": 10})

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        ctx = MagicMock()
        ctx.poll_command = MagicMock(
            side_effect=[
                {"action": "nonexistent", "params": {}},
                None,
            ]
        )
        ctx.command_result = MagicMock()

        class SimpleTask(Task):
            name = "test"

            async def run(self, ctx):
                pass

        task = SimpleTask()
        await task.process_commands(ctx)
        ctx.command_result.assert_called_once()
        args = ctx.command_result.call_args
        assert "error" in args[0][1]

    @pytest.mark.asyncio
    async def test_dispatch_command_ignores_input(self):
        """_dispatch_command returns silently for __input__ action without emitting errors."""
        ctx = MagicMock()
        ctx.command_result = MagicMock()

        class SimpleTask(Task):
            name = "test"

            async def run(self, ctx):
                pass

        task = SimpleTask()
        await task._dispatch_command(ctx, {"action": "__input__", "params": {"text": "hi"}})
        # Should NOT call command_result (no "Unknown command" error)
        ctx.command_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_command_handler(self):
        ctx = MagicMock()
        ctx.poll_command = MagicMock(
            side_effect=[
                {"action": "async_op", "params": {}},
                None,
            ]
        )
        ctx.command_result = MagicMock()

        class AsyncTask(Task):
            name = "test"

            async def run(self, ctx):
                pass

            @command
            async def async_op(self, ctx):
                return {"done": True}

        task = AsyncTask()
        await task.process_commands(ctx)
        ctx.command_result.assert_called_once_with("async_op", {"done": True})
