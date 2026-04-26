from pathlib import Path

import pytest

from switchplane.shell import Shell


@pytest.fixture
def shell(tmp_path):
    return Shell(
        allowed_paths=[tmp_path],
        allowed_commands=["echo", "true", "false", "ls", "head", "find", "grep", "cat", "sleep"],
    )


class TestValidatePath:
    def test_allowed(self, shell, tmp_path):
        result = shell.validate_path(str(tmp_path / "subdir"))
        assert result == (tmp_path / "subdir").resolve()

    def test_denied(self, shell):
        with pytest.raises(PermissionError, match="not within allowed"):
            shell.validate_path("/etc/passwd")

    def test_multiple_allowed_paths(self, tmp_path):
        path_a = tmp_path / "a"
        path_b = tmp_path / "b"
        path_a.mkdir()
        path_b.mkdir()
        s = Shell(allowed_paths=[path_a, path_b], allowed_commands=[])
        s.validate_path(str(path_a / "file.txt"))
        s.validate_path(str(path_b / "file.txt"))


class TestValidateCommand:
    def test_allowed(self, shell):
        shell._validate_command(["echo", "hello"])

    def test_denied(self, shell):
        with pytest.raises(PermissionError, match="not in allowed"):
            shell._validate_command(["rm", "-rf", "/"])

    def test_empty_command(self, shell):
        with pytest.raises(ValueError, match="Empty command"):
            shell._validate_command([])

    def test_full_path_binary(self, shell):
        shell._validate_command(["/usr/bin/echo", "hello"])

    def test_unknown_binary(self, shell):
        with pytest.raises(PermissionError):
            shell._validate_command(["curl", "http://example.com"])


class TestValidatedCwd:
    def test_none_passes_through(self, shell):
        assert shell._validated_cwd(None) is None

    def test_valid_cwd(self, shell, tmp_path):
        result = shell._validated_cwd(tmp_path)
        assert result == tmp_path.resolve()

    def test_invalid_cwd(self, shell):
        with pytest.raises(PermissionError):
            shell._validated_cwd(Path("/etc"))


class TestRun:
    @pytest.mark.asyncio
    async def test_success(self, shell):
        result = await shell.run(["echo", "hello world"])
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_non_zero_exit(self, shell):
        with pytest.raises(RuntimeError, match="failed"):
            await shell.run(["false"])

    @pytest.mark.asyncio
    async def test_with_cwd(self, shell, tmp_path):
        (tmp_path / "testfile.txt").write_text("content")
        result = await shell.run(["ls"], cwd=tmp_path)
        assert "testfile.txt" in result

    @pytest.mark.asyncio
    async def test_command_not_allowed(self, shell):
        with pytest.raises(PermissionError):
            await shell.run(["rm", "file"])


class TestRunOk:
    @pytest.mark.asyncio
    async def test_success(self, shell):
        assert await shell.run_ok(["true"]) is True

    @pytest.mark.asyncio
    async def test_failure(self, shell):
        assert await shell.run_ok(["false"]) is False


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises(self, tmp_path):
        s = Shell(allowed_paths=[tmp_path], allowed_commands=["sleep"], timeout=0.1)
        with pytest.raises(TimeoutError, match="timed out"):
            await s.run(["sleep", "10"], timeout=0.1)

    @pytest.mark.asyncio
    async def test_default_timeout(self, tmp_path):
        s = Shell(allowed_paths=[tmp_path], allowed_commands=["echo"], timeout=5.0)
        assert s.default_timeout == 5.0


class TestAsTool:
    @pytest.mark.asyncio
    async def test_creates_tool(self, shell):
        tool = shell.as_tool(
            name="echo_tool",
            cmd_template=["echo", "{message}"],
            description="Echo a message",
        )
        assert tool.name == "echo_tool"
        result = await tool.ainvoke({"message": "hello"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_path_param_validation(self, shell, tmp_path):
        tool = shell.as_tool(
            name="list_tool",
            cmd_template=["ls", "{path}"],
            description="List directory",
            path_params={"path"},
        )
        result = await tool.ainvoke({"path": "/etc"})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_path_param_resolved_to_absolute(self, shell, tmp_path):
        """Relative paths should be resolved against allowed_paths[0] before substitution."""
        (tmp_path / "marker.txt").write_text("found")
        tool = shell.as_tool(
            name="list_tool",
            cmd_template=["ls", "{path}"],
            description="List directory",
            path_params={"path"},
        )
        result = await tool.ainvoke({"path": "."})
        assert "marker.txt" in result

    @pytest.mark.asyncio
    async def test_no_placeholders(self, shell):
        tool = shell.as_tool(
            name="list_root",
            cmd_template=["echo", "static"],
            description="Static command",
        )
        result = await tool.ainvoke({})
        assert result == "static"


class TestFsTools:
    @pytest.mark.asyncio
    async def test_creates_four_tools(self, shell, tmp_path):
        tools = shell.fs_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"list_directory", "read_file", "search_files", "grep_files"}

    @pytest.mark.asyncio
    async def test_list_directory(self, shell, tmp_path):
        (tmp_path / "hello.txt").write_text("hi")
        tools = shell.fs_tools()
        list_tool = next(t for t in tools if t.name == "list_directory")
        result = await list_tool.ainvoke({"path": str(tmp_path)})
        assert "hello.txt" in result

    @pytest.mark.asyncio
    async def test_read_file(self, shell, tmp_path):
        (tmp_path / "data.txt").write_text("line1\nline2")
        tools = shell.fs_tools()
        read_tool = next(t for t in tools if t.name == "read_file")
        result = await read_tool.ainvoke({"file_path": str(tmp_path / "data.txt")})
        assert "line1" in result

    def test_missing_commands_raises(self, tmp_path):
        s = Shell(allowed_paths=[tmp_path], allowed_commands=["ls"])
        with pytest.raises(ValueError, match="fs_tools"):
            s.fs_tools()

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, shell, tmp_path):
        (tmp_path / "file.txt").write_text("nothing here")
        tools = shell.fs_tools()
        grep_tool = next(t for t in tools if t.name == "grep_files")
        result = await grep_tool.ainvoke({"directory": str(tmp_path), "pattern": "ZZZZZ"})
        assert "No matches" in result


class TestWriteTools:
    @pytest.mark.asyncio
    async def test_creates_three_tools(self, shell):
        tools = shell.write_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"write_file", "edit_file", "create_directory"}

    @pytest.mark.asyncio
    async def test_write_file(self, shell, tmp_path):
        tools = shell.write_tools()
        write_tool = next(t for t in tools if t.name == "write_file")
        result = await write_tool.ainvoke(
            {
                "file_path": str(tmp_path / "output.txt"),
                "content": "hello world",
            }
        )
        assert "Successfully wrote" in result
        assert (tmp_path / "output.txt").read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_parents(self, shell, tmp_path):
        tools = shell.write_tools()
        write_tool = next(t for t in tools if t.name == "write_file")
        result = await write_tool.ainvoke(
            {
                "file_path": str(tmp_path / "a" / "b" / "file.txt"),
                "content": "nested",
            }
        )
        assert "Successfully wrote" in result

    @pytest.mark.asyncio
    async def test_write_file_outside_allowed(self, shell):
        tools = shell.write_tools()
        write_tool = next(t for t in tools if t.name == "write_file")
        result = await write_tool.ainvoke(
            {
                "file_path": "/etc/bad.txt",
                "content": "nope",
            }
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_create_directory(self, shell, tmp_path):
        tools = shell.write_tools()
        mkdir_tool = next(t for t in tools if t.name == "create_directory")
        result = await mkdir_tool.ainvoke({"path": str(tmp_path / "newdir")})
        assert "Successfully created" in result
        assert (tmp_path / "newdir").is_dir()


class TestCodeTools:
    @pytest.mark.asyncio
    async def test_combines_fs_and_write(self, shell):
        tools = shell.code_tools()
        assert len(tools) == 7
        names = {t.name for t in tools}
        assert "list_directory" in names
        assert "write_file" in names
        assert "edit_file" in names
