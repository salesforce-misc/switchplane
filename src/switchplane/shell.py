"""Shell subprocess execution with guardrails."""

from __future__ import annotations

import asyncio
import difflib
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from switchplane.agent_runtime import AgentContext

from switchplane.llm import Tool

logger = structlog.get_logger()


def _default_render(ctx: AgentContext, name: str, args: dict) -> None:
    args_summary = " ".join((str(v).splitlines() or [""])[0] for v in args.values())
    ctx.tool_invoke(name, args_summary)


class Shell:
    """Subprocess execution wrapper with path and command allowlists."""

    def __init__(
        self,
        allowed_paths: list[Path],
        allowed_commands: list[str],
        timeout: float = 30.0,
        max_output_chars: int = 30_000,
        ctx: AgentContext | None = None,
    ):
        """Initialize Shell with security guardrails.

        Args:
            allowed_paths: Directories the shell is allowed to operate within.
            allowed_commands: Binary names that can be executed (e.g., ["git", "gh", "rg"]).
            timeout: Default timeout in seconds for each invocation.
            max_output_chars: Maximum characters returned from bash_tool output.
            ctx: Optional AgentContext for emitting file edit events.
        """
        self.allowed_paths = [p.resolve() for p in allowed_paths]
        self.allowed_commands = allowed_commands
        self.default_timeout = timeout
        self.max_output_chars = max_output_chars
        self._ctx = ctx

    def validate_path(self, path: str) -> Path:
        """Validate a path is within allowed directories.

        Relative paths are resolved against the first entry in
        ``allowed_paths`` (the primary working directory) so that LLM
        tool calls using bare filenames like ``requirements.txt`` map to
        the expected worktree rather than the process CWD.

        Args:
            path: Path to validate.

        Returns:
            Resolved Path object.

        Raises:
            PermissionError: If path is outside all allowed directories.
        """
        if not self.allowed_paths:
            raise PermissionError("No allowed paths configured")
        p = Path(path)
        if not p.is_absolute():
            p = self.allowed_paths[0] / p
        resolved = p.resolve()

        for allowed in self.allowed_paths:
            if resolved.is_relative_to(allowed):
                return resolved

        raise PermissionError(
            f"Path '{resolved}' is not within allowed directories: {', '.join(str(p) for p in self.allowed_paths)}"
        )

    def _validate_command(self, cmd: list[str]) -> None:
        """Validate command is in allowlist.

        Args:
            cmd: Command list where first element is the binary.

        Raises:
            PermissionError: If command is not in allowlist.
        """
        if not cmd:
            raise ValueError("Empty command")

        command_name = Path(cmd[0]).name  # Handle both "git" and "/usr/bin/git"
        if command_name not in self.allowed_commands:
            raise PermissionError(
                f"Command '{command_name}' is not in allowed commands: {', '.join(self.allowed_commands)}"
            )

    async def _exec(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        """Run a validated command and return (returncode, stdout, stderr).

        Raises:
            TimeoutError: If command exceeds timeout.
        """
        timeout = timeout if timeout is not None else self.default_timeout

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            return proc.returncode, stdout.decode().strip(), stderr.decode().strip()

        except TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()
            raise TimeoutError(f"Command {' '.join(cmd)} timed out after {timeout} seconds") from None

    def _validated_cwd(self, cwd: Path | None) -> Path | None:
        if cwd is not None:
            return self.validate_path(str(cwd))
        return None

    async def run(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run a command, return stdout. Raises RuntimeError on non-zero exit."""
        self._validate_command(cmd)
        returncode, stdout, stderr = await self._exec(
            cmd,
            cwd=self._validated_cwd(cwd),
            env=env,
            timeout=timeout,
        )
        if returncode != 0:
            error_msg = stderr or f"Exit code {returncode}"
            raise RuntimeError(f"Command {' '.join(cmd)} failed: {error_msg}")
        return stdout

    async def run_ok(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Run a command, return True if exit code is 0."""
        self._validate_command(cmd)
        returncode, _, _ = await self._exec(
            cmd,
            cwd=self._validated_cwd(cwd),
            env=env,
            timeout=timeout,
        )
        return returncode == 0

    def as_tool(
        self,
        name: str,
        cmd_template: list[str],
        description: str,
        cwd: Path | None = None,
        path_params: set[str] | None = None,
    ):
        """Create a LangChain StructuredTool from a command template.

        Args:
            name: Tool name.
            cmd_template: Command template with {param_name} placeholders.
            description: Tool description.
            cwd: Working directory for the command.
            path_params: Placeholder names that represent filesystem paths.
                         These are validated against allowed_paths before execution.

        Returns:
            LangChain StructuredTool instance.

        Example:
            shell = Shell(allowed_paths=[Path("/src")], allowed_commands=["rg"])
            grep_tool = shell.as_tool(
                name="grep_files",
                cmd_template=["rg", "--no-heading", "-n", "{pattern}", "{directory}"],
                description="Search file contents for a regex pattern.",
                path_params={"directory"},
            )
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain-core is required for as_tool(). Install with: pip install switchplane[mcp]"
            ) from None

        from pydantic import create_model
        from pydantic.fields import FieldInfo

        # Extract placeholders from template
        placeholder_pattern = re.compile(r"\{(\w+)\}")
        placeholders = set()
        for arg in cmd_template:
            placeholders.update(placeholder_pattern.findall(arg))

        # Build Pydantic model for parameters
        fields = {}
        for param_name in placeholders:
            fields[param_name] = (str, FieldInfo(description=f"Value for {param_name}"))

        args_schema = create_model(f"{name}_Args", **fields) if fields else None

        _path_params = path_params or set()

        async def _invoke(**kwargs) -> str:
            """Execute the command with substituted parameters."""
            for param_name in _path_params:
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = str(self.validate_path(str(kwargs[param_name])))
                    except PermissionError as e:
                        return f"Error: {e}"

            # Substitute placeholders in template
            cmd = []
            for arg in cmd_template:
                substituted = arg
                for param_name, value in kwargs.items():
                    substituted = substituted.replace(f"{{{param_name}}}", str(value))
                cmd.append(substituted)

            try:
                return await self.run(cmd, cwd=cwd)
            except (RuntimeError, PermissionError, TimeoutError) as e:
                return f"Error: {e}"

        return StructuredTool.from_function(
            coroutine=_invoke,
            name=name,
            description=description,
            args_schema=args_schema,
        )

    _FS_COMMANDS = frozenset({"ls", "find", "grep"})

    def fs_tools(self) -> list:
        """Create standard filesystem tools for LLM use.

        Returns a list of LangChain StructuredTools: list_directory, read_file,
        search_files, grep_files.

        Requires 'ls', 'find', 'grep' in allowed_commands.
        """
        missing = self._FS_COMMANDS - set(self.allowed_commands)
        if missing:
            raise ValueError(f"fs_tools() requires these commands in allowed_commands: {', '.join(sorted(missing))}")

        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain-core is required for fs_tools(). Install with: pip install switchplane[mcp]"
            ) from None

        from pydantic import create_model
        from pydantic.fields import FieldInfo

        list_directory = self.as_tool(
            name="list_directory",
            cmd_template=["ls", "-1F", "{path}"],
            description="List contents of a directory. Directories have a trailing /.",
            path_params={"path"},
        )

        _read_schema = create_model(
            "read_file_Args",
            file_path=(str, FieldInfo(description="Path to the file to read")),
            offset=(int, FieldInfo(default=1, description="1-based line number to start reading from")),
            limit=(int, FieldInfo(default=5000, description="Maximum number of lines to return")),
        )

        async def _read_invoke(file_path: str, offset: int = 1, limit: int = 5000) -> str:
            try:
                resolved = self.validate_path(file_path)
            except PermissionError as e:
                return f"Error: {e}"
            if not resolved.is_file():
                return f"Error: File not found: {resolved}"
            try:
                with open(resolved) as f:
                    lines = f.readlines()
            except Exception as e:
                return f"Error: Failed to read file: {e}"
            # offset is 1-based; clamp to valid range
            start = max(offset, 1) - 1
            selected = lines[start : start + limit]
            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")
            return "\n".join(numbered)

        read_file = StructuredTool.from_function(
            coroutine=_read_invoke,
            name="read_file",
            description=(
                "Read file contents with line numbers. "
                "Optionally pass offset (1-based start line) and limit (max lines) "
                "to read a specific range — useful after finding a location via grep."
            ),
            args_schema=_read_schema,
        )

        search_files = self.as_tool(
            name="search_files",
            cmd_template=["find", "{directory}", "-type", "f", "-name", "{pattern}"],
            description="Search for files by name pattern.",
            path_params={"directory"},
        )

        # grep needs a custom wrapper because exit code 1 means "no matches", not failure
        _grep_schema = create_model(
            "grep_files_Args",
            directory=(str, FieldInfo(description="Directory to search in")),
            pattern=(str, FieldInfo(description="Regex pattern to search for")),
        )

        async def _grep_invoke(directory: str, pattern: str) -> str:
            try:
                resolved = self.validate_path(directory)
                return await self.run(["grep", "-rn", "--include=*", "-m", "100", pattern, str(resolved)])
            except RuntimeError:
                return f"No matches found for pattern: {pattern}"
            except (PermissionError, TimeoutError) as e:
                return f"Error: {e}"

        grep_files = StructuredTool.from_function(
            coroutine=_grep_invoke,
            name="grep_files",
            description="Search file contents for a regex pattern. Returns matching lines with file:line prefix.",
            args_schema=_grep_schema,
        )

        render = _default_render if self._ctx else None
        return [
            Tool(list_directory, render_fn=render),
            Tool(read_file, render_fn=render),
            Tool(search_files, render_fn=render),
            Tool(grep_files, render_fn=render),
        ]

    def write_tools(self) -> list:
        """Create file writing tools for LLM use.

        Returns a list of LangChain StructuredTools: write_file, create_directory.

        These tools use Python file I/O directly (not shell commands) but still
        validate paths against allowed_paths.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain-core is required for write_tools(). Install with: pip install switchplane[mcp]"
            ) from None

        from pydantic import create_model
        from pydantic.fields import FieldInfo

        # write_file tool
        _write_schema = create_model(
            "write_file_Args",
            file_path=(str, FieldInfo(description="Path to file to write")),
            content=(str, FieldInfo(description="Content to write to the file")),
        )

        async def _write_invoke(file_path: str, content: str) -> str:
            try:
                resolved = self.validate_path(file_path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                try:
                    old_content = resolved.read_text() if resolved.is_file() else ""
                except (UnicodeDecodeError, ValueError):
                    old_content = ""
                resolved.write_text(content)
                if self._ctx is not None:
                    diff = "".join(difflib.unified_diff(
                        old_content.splitlines(keepends=True),
                        content.splitlines(keepends=True),
                        fromfile=str(resolved),
                        tofile=str(resolved),
                    ))
                    if diff:
                        self._ctx.file_edit(str(resolved), diff)
                return f"Successfully wrote {len(content)} bytes to {resolved}"
            except PermissionError as e:
                return f"Error: {e}"
            except Exception as e:
                return f"Error: Failed to write file: {e}"

        write_file = StructuredTool.from_function(
            coroutine=_write_invoke,
            name="write_file",
            description="Write content to a file. Creates parent directories if needed. Overwrites if file exists.",
            args_schema=_write_schema,
        )

        # edit_file tool
        _edit_schema = create_model(
            "edit_file_Args",
            file_path=(str, FieldInfo(description="Path to file to edit")),
            old_text=(str, FieldInfo(description="Exact text to find in the file")),
            new_text=(str, FieldInfo(description="Text to replace old_text with")),
        )

        async def _edit_invoke(file_path: str, old_text: str, new_text: str) -> str:
            try:
                resolved = self.validate_path(file_path)
                if not resolved.is_file():
                    return f"Error: File not found: {resolved}"
                content = resolved.read_text()
                count = content.count(old_text)
                if count == 0:
                    return "Error: old_text not found in file"
                if count > 1:
                    return f"Error: old_text matches {count} locations; must be unique"
                new_content = content.replace(old_text, new_text, 1)
                resolved.write_text(new_content)
                if self._ctx is not None:
                    diff = "".join(difflib.unified_diff(
                        content.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=str(resolved),
                        tofile=str(resolved),
                    ))
                    if diff:
                        self._ctx.file_edit(str(resolved), diff)
                return f"Successfully edited {resolved}"
            except PermissionError as e:
                return f"Error: {e}"
            except Exception as e:
                return f"Error: Failed to edit file: {e}"

        edit_file = StructuredTool.from_function(
            coroutine=_edit_invoke,
            name="edit_file",
            description=(
                "Edit a file by replacing an exact text match. Use this for targeted edits "
                "instead of rewriting the entire file with write_file. Fails if old_text is "
                "not found or matches multiple locations."
            ),
            args_schema=_edit_schema,
        )

        # create_directory tool
        _mkdir_schema = create_model(
            "create_directory_Args",
            path=(str, FieldInfo(description="Path to directory to create")),
        )

        async def _mkdir_invoke(path: str) -> str:
            try:
                resolved = self.validate_path(path)
                resolved.mkdir(parents=True, exist_ok=True)
                return f"Successfully created directory: {resolved}"
            except PermissionError as e:
                return f"Error: {e}"
            except Exception as e:
                return f"Error: Failed to create directory: {e}"

        create_directory = StructuredTool.from_function(
            coroutine=_mkdir_invoke,
            name="create_directory",
            description="Create a directory (with parent directories if needed).",
            args_schema=_mkdir_schema,
        )

        render = _default_render if self._ctx else None
        return [
            Tool(write_file, render_fn=None),
            Tool(edit_file, render_fn=None),
            Tool(create_directory, render_fn=render),
        ]

    def code_tools(self) -> list:
        """Create combined filesystem read/write tools for LLM coding tasks.

        Returns fs_tools() + write_tools() - a complete set for code manipulation.
        """
        return self.fs_tools() + self.write_tools()

    def bash_tool(self):
        """Create a general-purpose bash tool for LLM use.

        Returns a single LangChain StructuredTool that executes shell commands
        with validation against the command allowlist. CWD is locked to
        allowed_paths[0]. Output is truncated to max_output_chars.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain-core is required for bash_tool(). Install with: pip install switchplane[mcp]"
            ) from None

        from pydantic import create_model
        from pydantic.fields import FieldInfo

        _bash_schema = create_model(
            "bash_Args",
            command=(str, FieldInfo(description="Shell command to execute")),
            timeout=(
                int,
                FieldInfo(
                    default=int(self.default_timeout),
                    description="Timeout in seconds",
                ),
            ),
        )

        async def _bash_invoke(command: str, timeout: int | None = None) -> str:
            try:
                cmd = shlex.split(command)
            except ValueError as e:
                return f"Error: Failed to parse command: {e}"

            if not cmd:
                return "Error: Empty command"

            try:
                self._validate_command(cmd)
            except (PermissionError, ValueError) as e:
                return f"Error: {e}"

            effective_timeout = float(timeout) if timeout is not None else self.default_timeout
            cwd = self.allowed_paths[0]

            try:
                returncode, stdout, stderr = await self._exec(
                    cmd,
                    cwd=cwd,
                    timeout=effective_timeout,
                )
            except TimeoutError as e:
                return f"Error: {e}"

            parts = []
            if returncode != 0:
                parts.append(f"Exit code: {returncode}")
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"\nSTDERR:\n{stderr}")

            output = "\n".join(parts) if parts else ""
            if len(output) > self.max_output_chars:
                output = output[: self.max_output_chars] + "\n... [truncated]"
            return output

        raw = StructuredTool.from_function(
            coroutine=_bash_invoke,
            name="bash",
            description=(
                "Execute a shell command. The command is parsed with shlex and "
                "validated against the allowed command list. Working directory "
                f"is {self.allowed_paths[0]}."
            ),
            args_schema=_bash_schema,
        )
        render = _default_render if self._ctx else None
        return Tool(raw, render_fn=render)

    def agent_tools(self) -> list:
        """Minimal tool surface for LLM coding agents: bash + write_file + edit_file."""
        write_tools = self.write_tools()
        # write_tools() returns [write_file, edit_file, create_directory];
        # agent_tools only needs the first two.
        return [self.bash_tool(), *write_tools[:2]]
