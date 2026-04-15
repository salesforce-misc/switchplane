"""Shell subprocess execution with guardrails."""

import asyncio
import re
from pathlib import Path

import structlog

logger = structlog.get_logger()


class Shell:
    """Subprocess execution wrapper with path and command allowlists."""

    def __init__(
        self,
        allowed_paths: list[Path],
        allowed_commands: list[str],
        timeout: float = 30.0,
    ):
        """Initialize Shell with security guardrails.

        Args:
            allowed_paths: Directories the shell is allowed to operate within.
            allowed_commands: Binary names that can be executed (e.g., ["git", "gh", "rg"]).
            timeout: Default timeout in seconds for each invocation.
        """
        self.allowed_paths = [p.resolve() for p in allowed_paths]
        self.allowed_commands = allowed_commands
        self.default_timeout = timeout

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

    _FS_COMMANDS = frozenset({"ls", "head", "find", "grep"})

    def fs_tools(self) -> list:
        """Create standard filesystem tools for LLM use.

        Returns a list of LangChain StructuredTools: list_directory, read_file,
        search_files, grep_files.

        Requires 'ls', 'head', 'find', 'grep' in allowed_commands.
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

        read_file = self.as_tool(
            name="read_file",
            cmd_template=["head", "-n", "5000", "{file_path}"],
            description="Read file contents (first 5000 lines).",
            path_params={"file_path"},
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

        return [list_directory, read_file, search_files, grep_files]

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
                # Create parent directories if needed
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content)
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

        return [write_file, create_directory]

    def code_tools(self) -> list:
        """Create combined filesystem read/write tools for LLM coding tasks.

        Returns fs_tools() + write_tools() - a complete set for code manipulation.
        """
        return self.fs_tools() + self.write_tools()
