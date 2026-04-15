"""Click CLI factory for Switchplane applications."""

import asyncio
import json
import queue
import sys
import threading
import time

import click

from switchplane import fmt
from switchplane.daemon import RuntimePaths, start_daemon, stop_daemon
from switchplane.protocol import CliRequest, CliResponse
from switchplane.transport import ControlPlaneClient, is_alive
from switchplane.tui import _parse_kv_args, run_tui


def build_cli(app) -> click.Group:
    """Build a Click CLI for an application.

    Args:
        app: Application instance with runtime_dir set

    Returns:
        Click group configured for this application
    """

    paths = RuntimePaths(app.runtime_dir)

    def ensure_daemon() -> None:
        """Ensure the control plane daemon is running, starting it if needed."""
        if not is_alive(paths.sock_path):
            start_daemon(paths, app)
            # Wait briefly for socket to be ready
            for _ in range(20):
                time.sleep(0.1)
                if is_alive(paths.sock_path):
                    return
            click.echo("Error: Could not connect to control plane", err=True)
            sys.exit(1)

    def send_request(method: str, params: dict | None = None) -> CliResponse:
        """Send a request to the control plane and return the response."""
        with ControlPlaneClient(paths.sock_path) as client:
            request = CliRequest(method=method, params=params or {})
            return client.send(request)

    @click.group(invoke_without_command=True)
    @click.pass_context
    def cli(ctx):
        # {app.name} - powered by Switchplane.
        ctx.ensure_object(dict)
        if ctx.invoked_subcommand is None:
            if sys.stdin.isatty():
                # Interactive: enter TUI dashboard (auto-discovers running tasks)
                ensure_daemon()
                asyncio.run(run_tui(paths.sock_path, initial_tasks=None))
            else:
                click.echo(ctx.get_help())

    # Build all the same subcommands: runtime (start/stop/status), task (list/show/cancel/follow/_dispatch), run
    # But using the closured ensure_daemon and send_request instead of globals

    @cli.group()
    def runtime():
        """Manage the runtime."""
        pass

    @runtime.command("start")
    def runtime_start():
        """Start the control plane daemon."""
        start_daemon(paths, app)

    @runtime.command("stop")
    def runtime_stop():
        """Stop the control plane daemon."""
        stop_daemon(paths)

    @runtime.command("status")
    def runtime_status():
        """Show runtime status."""
        if not is_alive(paths.sock_path):
            click.echo("Control plane is not running")
            return
        resp = send_request("status")
        if resp.ok:
            status = resp.result
            click.echo("Status: running")
            click.echo(f"Active agents: {status['active_agents']}")
            click.echo(f"Active connections: {status['active_connections']}")
            click.echo(f"Running tasks: {status['running_tasks']}")
        else:
            click.echo(f"Error: {resp.error}", err=True)

    # Agent commands
    @cli.group()
    def agent():
        """Manage agents."""
        pass

    @agent.command("list")
    def agent_list():
        """List available agents and their tasks."""
        ensure_daemon()
        resp = send_request("list_agents")
        if not resp.ok:
            click.echo(f"Error: {resp.error}", err=True)
            return
        agents = resp.result
        if not agents:
            click.echo("No agents registered")
            return
        for ag in agents:
            click.echo(f"\n{ag['name']}")
            tasks = ag.get("tasks", {})
            if not tasks:
                click.echo("  (no tasks)")
                continue
            for task_name, info in tasks.items():
                mode_tag = f" [{info['mode']}]" if info.get("mode") == "long_running" else ""
                desc = f" - {info['description']}" if info.get("description") else ""
                click.echo(f"  {task_name}{mode_tag}{desc}")
                params = info.get("parameters", {})
                if params:
                    for pname, pinfo in params.items():
                        req = "required" if pinfo.get("required") else f"default: {pinfo.get('default')}"
                        pdesc = f"  {pinfo['description']}" if pinfo.get("description") else ""
                        click.echo(f"    --{pname.replace('_', '-')}  ({req}){pdesc}")
                commands = info.get("commands", {})
                if commands:
                    click.echo(f"    commands: {', '.join(commands)}")

    # Auth commands (does not require the daemon)
    @cli.group()
    def auth():
        """Manage OAuth authentication for MCP servers."""
        pass

    def _resolve_oauth_config(name):
        """Resolve a server name or oauth_group to an McpServerConfig."""
        config = app.mcp_servers.get(name)
        if config:
            return config
        # Check if name matches an oauth_group
        for cfg in app.mcp_servers.values():
            if cfg.oauth_group == name and cfg.oauth:
                return cfg
        return None

    @auth.command("login")
    @click.argument("server_name")
    def auth_login(server_name):
        """Authenticate with an OAuth-enabled MCP server.

        Opens a browser for the consent flow and stores tokens locally.
        Does not require the daemon to be running.
        """
        config = _resolve_oauth_config(server_name)
        if not config:
            available = ", ".join(app.mcp_servers) or "(none)"
            click.echo(f"Error: MCP server '{server_name}' not registered. Available: {available}", err=True)
            sys.exit(1)
        if not config.oauth:
            click.echo(f"Error: MCP server '{server_name}' does not use OAuth", err=True)
            sys.exit(1)

        async def _run_auth():
            from switchplane.oauth import FileTokenStorage, build_oauth_http_client, run_direct_oidc_login

            click.echo(f"Authenticating with '{server_name}'...")

            if config.oauth.is_direct:
                # Direct OIDC: run the PKCE flow against explicit endpoints
                storage_dir = app.runtime_dir / "oauth" / config.oauth_storage_key
                storage = FileTokenStorage(storage_dir, config.oauth)
                await run_direct_oidc_login(config.oauth, storage, runtime_dir=app.runtime_dir)
            else:
                # MCP-spec OAuth: trigger the flow via a probe request
                client = await build_oauth_http_client(config, app.runtime_dir, interactive=True)
                async with client:
                    try:
                        await client.post(config.url, content=b"")
                    except Exception:
                        pass  # Tokens are stored by the auth flow regardless of response.

            click.echo(f"Authenticated with '{server_name}'. Tokens stored in {app.runtime_dir / 'oauth' / config.oauth_storage_key}/")

        try:
            asyncio.run(_run_auth())
        except KeyboardInterrupt:
            click.echo("\nAuthentication cancelled.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    @auth.command("status")
    def auth_status():
        """Show OAuth token status for registered MCP servers."""
        oauth_dir = app.runtime_dir / "oauth"
        found_any = False
        groups: dict[str, list[str]] = {}
        for name, config in app.mcp_servers.items():
            if not config.oauth:
                continue
            found_any = True
            label = config.oauth_group or name
            groups.setdefault(label, []).append(name)
        for label, members in groups.items():
            config = _resolve_oauth_config(label)
            key = config.oauth_storage_key if config else label
            suffix = f" ({', '.join(members)})" if config and config.oauth_group else ""
            token_file = oauth_dir / key / "tokens.json"
            if token_file.exists():
                click.echo(f"  {label}: authenticated{suffix}")
            else:
                click.echo(f"  {label}: not authenticated{suffix} — run '{app.name} auth login {label}'")
        if not found_any:
            click.echo("No OAuth-enabled MCP servers registered")

    @auth.command("logout")
    @click.argument("server_name")
    def auth_logout(server_name):
        """Remove stored OAuth tokens for an MCP server."""
        import shutil

        config = _resolve_oauth_config(server_name)
        key = config.oauth_storage_key if config else server_name
        server_dir = app.runtime_dir / "oauth" / key
        if not server_dir.exists():
            click.echo(f"No stored tokens for '{server_name}'")
            return
        shutil.rmtree(server_dir)
        click.echo(f"Removed tokens for '{server_name}'")

    # TaskGroup with same dispatch pattern
    class TaskGroup(click.Group):
        def parse_args(self, ctx, args):
            if args and args[0] not in self.commands:
                ctx.ensure_object(dict)
                ctx.obj["task_command_args"] = list(args)
                args = ["_dispatch"]
            return super().parse_args(ctx, args)

    @cli.group(cls=TaskGroup)
    def task():
        """Manage tasks."""
        pass

    # All task subcommands: list, show, cancel, follow, _dispatch
    # Same implementation as current cli.py but using closured send_request/ensure_daemon

    @task.command("list")
    @click.option("--status", "status_filter", default=None, help="Filter by status")
    def task_list(status_filter):
        """List tasks."""
        ensure_daemon()
        params = {}
        if status_filter:
            params["status"] = status_filter
        resp = send_request("list_tasks", params)
        if resp.ok:
            tasks = resp.result
            if not tasks:
                click.echo("No tasks found")
                return
            rows = []
            for t in tasks:
                rows.append((t["task_id"], f"{t['agent_name']}/{t['task_name']}", t["status"], t["created_at"]))
            widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
            for row in rows:
                line = "  ".join(val.ljust(w) for val, w in zip(row, widths, strict=True))
                click.echo(f"  {line}")
        else:
            click.echo(f"Error: {resp.error}", err=True)

    @task.command("show")
    @click.argument("task_id")
    def task_show(task_id):
        """Show task details."""
        ensure_daemon()
        resp = send_request("get_task", {"task_id": task_id})
        if resp.ok:
            data = resp.result
            t = data["task"]
            click.echo(f"Task ID:  {t['task_id']}")
            click.echo(f"Agent:    {t['agent_name']}")
            click.echo(f"Task:     {t['task_name']}")
            click.echo(f"Status:   {t['status']}")
            click.echo(f"Created:  {t['created_at']}")
            click.echo(f"Updated:  {t['updated_at']}")
            if t.get("input_json"):
                click.echo(f"Input:    {t['input_json']}")
            if t.get("result_json"):
                for line in fmt.format_result(t["result_json"]):
                    click.echo(f"  {line}")
            if t.get("error_json"):
                click.echo(f"Error:    {t['error_json']}")
            events = data.get("events", [])
            if events:
                click.echo(f"\nEvents ({len(events)}):")
                for e in events:
                    click.echo(f"  {e['timestamp']}  {e['event_type']}  {json.dumps(e['payload'])}")
        else:
            click.echo(f"Error: {resp.error}", err=True)

    @task.command("cancel")
    @click.argument("task_id")
    def task_cancel(task_id):
        """Cancel a running task."""
        ensure_daemon()
        resp = send_request("cancel_task", {"task_id": task_id})
        if resp.ok:
            click.echo(f"Task {task_id} cancelled" if resp.result.get("cancelled") else "Task not found or not running")
        else:
            click.echo(f"Error: {resp.error}", err=True)

    @task.command("retry")
    @click.argument("task_id")
    @click.option("--detach", "-d", is_flag=True, help="Retry and exit without following")
    def task_retry(task_id, detach):
        """Retry a failed or cancelled task from its last checkpoint."""
        ensure_daemon()
        resp = send_request("retry_from_checkpoint", {"task_id": task_id})
        if not resp.ok:
            click.echo(f"Error: {resp.error}", err=True)
            sys.exit(1)
        if detach:
            click.echo(f"Task {task_id} retried")
            click.echo(f"Follow with: {app.name} task follow {task_id}")
            return
        click.echo(f"Task {task_id} retried")
        _follow_task(task_id, send_request)

    @task.command("clear")
    def task_clear():
        """Purge completed, failed, and cancelled task history."""
        ensure_daemon()
        resp = send_request("clear_tasks")
        if resp.ok:
            count = resp.result.get("deleted", 0)
            click.echo(f"Deleted {count} task(s)")
        else:
            click.echo(f"Error: {resp.error}", err=True)

    @task.command("purge")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
    def task_purge(yes):
        """Purge all terminal tasks: database records, data files, and logs."""
        ensure_daemon()
        if not yes:
            click.confirm("This will delete all completed, failed, and cancelled tasks and their data. Continue?", abort=True)
        resp = send_request("purge_tasks")
        if resp.ok:
            count = resp.result.get("purged", 0)
            click.echo(f"Purged {count} task(s)")
        else:
            click.echo(f"Error: {resp.error}", err=True)

    @task.command("follow")
    @click.argument("task_id")
    def task_follow(task_id):
        """Follow events from a running task."""
        ensure_daemon()
        _follow_task(task_id, send_request)

    @task.command("_dispatch", hidden=True)
    @click.pass_context
    def task_dispatch(ctx):
        """Hidden command for: <app> task <task_id> <command> [--key=value ...]"""
        args = ctx.obj.get("task_command_args", [])
        if len(args) < 2:
            click.echo("Error: Usage: task <task_id> <command> [--key=value ...]", err=True)
            sys.exit(1)
        task_id = args[0]
        try:
            command_name, params = _parse_kv_args(args[1:])
        except (IndexError, ValueError) as exc:
            click.echo(f"Error: Invalid command: {exc}", err=True)
            sys.exit(1)
        ensure_daemon()
        resp = send_request("task_command", {"task_id": task_id, "action": command_name, "params": params})
        if resp.ok:
            click.echo(f"Command '{command_name}' sent to task {task_id}")
            click.echo("Use 'task follow' to see results")
        else:
            click.echo(f"Error: {resp.error}", err=True)
            sys.exit(1)

    # Run command
    @cli.command("run", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
    @click.argument("agent_name")
    @click.argument("task_name")
    @click.option("--detach", "-d", is_flag=True, help="Submit and exit without following")
    @click.pass_context
    def run_task(ctx, agent_name, task_name, detach):
        """Run a task: <app> run <agent> <task> [--param value ...] [-d]"""
        ensure_daemon()
        _, params = _parse_kv_args(ctx.args, start_index=0)
        resp = send_request("submit_task", {"agent_name": agent_name, "task_name": task_name, "input": params})
        if not resp.ok:
            click.echo(f"Error: {resp.error}", err=True)
            sys.exit(1)
        task_id = resp.result["task_id"]
        if detach:
            click.echo(f"Task submitted: {task_id}")
            click.echo(f"Follow with: {app.name} task follow {task_id}")
            click.echo(f"Cancel with: {app.name} task cancel {task_id}")
            return
        click.echo(f"Task submitted: {task_id}")
        _follow_task(task_id, send_request)

    return cli


def _stdin_reader(cmd_queue: queue.Queue) -> None:
    try:
        for line in sys.stdin:
            stripped = line.strip()
            if stripped:
                cmd_queue.put(stripped)
    except (EOFError, OSError):
        pass


def _print_task_help(task_id: str, send_request) -> None:
    """Print available commands for the task currently being followed."""
    # Look up task metadata
    task_resp = send_request("get_task", {"task_id": task_id})
    if not task_resp.ok:
        click.echo(f"  Error: {task_resp.error}", err=True)
        return
    t = task_resp.result["task"]
    agent_name, task_name = t["agent_name"], t["task_name"]

    agents_resp = send_request("list_agents")
    if not agents_resp.ok:
        click.echo(f"  Error: {agents_resp.error}", err=True)
        return

    task_info = None
    for ag in agents_resp.result:
        if ag["name"] == agent_name:
            task_info = ag.get("tasks", {}).get(task_name)
            break

    if not task_info:
        click.echo(f"  {agent_name}/{task_name} — no metadata available")
        return

    desc = task_info.get("description", "")
    click.echo(f"  {agent_name}/{task_name}" + (f" — {desc}" if desc else ""))

    commands = task_info.get("commands", {})
    if not commands:
        click.echo("  No commands available for this task.")
        return

    click.echo("  Commands:")
    for cmd_name, cmd_info in commands.items():
        params = cmd_info.get("parameters", {})
        if params:
            flags = " ".join(f"--{p.replace('_', '-')}" for p in params)
            click.echo(f"    /{cmd_name} {flags}")
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "str")
                if pinfo.get("required"):
                    req = "required"
                elif "default" in pinfo:
                    req = f"default: {pinfo['default']}"
                else:
                    req = "optional"
                click.echo(f"      --{pname.replace('_', '-')}  ({ptype}, {req})")
        else:
            click.echo(f"    /{cmd_name}")


def _dispatch_inline_command(line: str, task_id: str, send_request) -> None:
    try:
        command_name, params = _parse_kv_args(line.split())
    except (IndexError, ValueError) as exc:
        click.echo(f"  Invalid command: {exc}", err=True)
        return
    resp = send_request("task_command", {"task_id": task_id, "action": command_name, "params": params})
    if resp.ok:
        click.echo(f"  Command '{command_name}' sent")
    else:
        click.echo(f"  Command error: {resp.error}", err=True)


def _follow_task(task_id: str, send_request) -> None:
    """Stream events from a task until it completes."""
    last_event_id = 0
    last_system_seq = 0
    terminal_statuses = {"completed", "failed", "cancelled"}

    cmd_queue: queue.Queue = queue.Queue()
    reader_thread = threading.Thread(target=_stdin_reader, args=(cmd_queue,), daemon=True)
    reader_thread.start()

    task_status = "running"

    if sys.stdin.isatty():
        click.echo("(use /command for task commands, Ctrl+C to detach)")

    try:
        while True:
            while not cmd_queue.empty():
                try:
                    line = cmd_queue.get_nowait()
                    if line.startswith("/"):
                        stripped = line[1:].strip()
                        if stripped.lower() in ("help", "?"):
                            _print_task_help(task_id, send_request)
                        else:
                            _dispatch_inline_command(stripped, task_id, send_request)
                    elif task_status == "interrupted":
                        resp = send_request(
                            "task_command",
                            {"task_id": task_id, "action": "__input__", "params": {"text": line}},
                        )
                        if not resp.ok:
                            click.echo(f"  Error: {resp.error}", err=True)
                    else:
                        click.echo("  Task is not waiting for input. Use /command for task commands.")
                except queue.Empty:
                    break

            resp = send_request(
                "get_events_since",
                {
                    "task_id": task_id,
                    "after_event_id": last_event_id,
                },
            )
            if not resp.ok:
                click.echo(f"Error: {resp.error}", err=True)
                return

            events = resp.result.get("events", [])
            status = resp.result.get("status", "unknown")
            task_status = status

            for e in events:
                last_event_id = e["event_id"]
                _print_event(e)

            # Poll CP system logs
            try:
                sys_resp = send_request("get_system_logs", {"after_seq": last_system_seq})
                if sys_resp.ok:
                    for rec in sys_resp.result.get("logs", []):
                        last_system_seq = rec["seq"]
                        click.echo(click.style(f"  [cp/{rec['level']}] {rec['logger']}: {rec['message']}", dim=True))
            except Exception:
                pass  # non-critical; don't disrupt the follow loop

            if status in terminal_statuses:
                check = send_request("get_task", {"task_id": task_id})
                if check.ok:
                    t = check.result["task"]
                    if status == "completed" and t.get("result_json"):
                        for line in fmt.format_result(t["result_json"]):
                            click.echo(f"  {line}")
                    elif status == "failed" and t.get("error_json"):
                        try:
                            error_data = json.loads(t["error_json"])
                            if isinstance(error_data, dict):
                                click.echo(f"Error: {error_data.get('error', error_data)}", err=True)
                                if "traceback" in error_data:
                                    for line in error_data["traceback"].splitlines():
                                        click.echo(f"  {line}", err=True)
                            else:
                                click.echo(f"Error: {t['error_json']}", err=True)
                        except (json.JSONDecodeError, TypeError):
                            click.echo(f"Error: {t['error_json']}", err=True)
                    click.echo(f"Task {status}.")
                return

            time.sleep(0.5)

    except KeyboardInterrupt:
        click.echo("\nDetached. Task continues in background.")


def _print_event(event: dict) -> None:
    """Format and print a single event using the shared renderer."""
    for line in fmt.render_event(event):
        text = "".join(seg[1] for seg in line.segments)
        is_dim = all(s in (fmt.DIM, fmt.TS) for s, _ in line.segments)
        click.echo(click.style(f"  {text}", dim=True) if is_dim else f"  {text}")
