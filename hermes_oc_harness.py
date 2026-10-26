from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

OPENCODE_BIN_ENV = "OPENCODE_BIN"
DEFAULT_PROVIDER = "opencode"
DEFAULT_MODEL = "mimo-v2.6-flash-free"
FREE_MODEL_SUFFIX = "-free"
DEFAULT_TIMEOUT = 300.0
LOGICAL_BASE_URL = "https://opencode.ai/zen/v1/chat/completions"
_SYSTEM_PREAMBLE = "Follow these standing instructions for the whole session:\n\n{}\n\n---\n"

# Tool interception (PoC): run OpenCode with every permission set to "ask". A
# non-interactive run auto-rejects the call, but still emits the full tool_use
# event (name + input), which we translate into OpenAI tool_calls for Hermes to
# execute locally. The tool schema stays untouched, which is what keeps the
# free-tier request valid (removing or changing tools returns 429).
TOOL_INTERCEPT_ENV = "HERMES_OC_TOOL_INTERCEPT"
_INTERCEPT_CONFIG = '{"permission":{"*":"ask"}}'
_INTERCEPT_GUIDANCE = (
    "When a tool call comes back with an error saying that a non-interactive run "
    "cannot ask for permission, never retry the call inside this turn and never "
    "invent or guess the tool output. End your turn immediately: the host runs the "
    "call and sends the real result back in the next message.\n"
    "Tool calls shown in earlier assistant turns were already executed by the host. "
    "Every TOOL RESULT block is authoritative and final: never repeat a call whose "
    "result is already in the transcript, and never fabricate output for a tool you "
    "have not received a result for."
)
_MAX_TOOL_CALLS = 8
# OpenCode tool name -> Hermes tool name (mirrors COMPAT_TOOL_TARGETS).
_OC_TO_HERMES_TOOL = {
    "bash": "terminal",
    "shell": "terminal",
    "edit": "patch",
    "glob": "search_files",
    "grep": "search_files",
    "list": "list_directory",
    "read": "read_file",
    "skill": "skill_view",
    "task": "delegate_task",
    "todowrite": "todo_list",
    "webfetch": "web_extract",
    "websearch": "web_search",
    "write": "write_file",
}


def tool_intercept_enabled() -> bool:
    return os.environ.get(TOOL_INTERCEPT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class HarnessError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HarnessUnavailable(HarnessError):
    pass


def opencode_binary() -> str | None:
    configured = os.environ.get(OPENCODE_BIN_ENV)
    if configured:
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        return None
    return shutil.which("opencode")


def model_reference(model: str) -> str:
    return model if "/" in model else f"{DEFAULT_PROVIDER}/{model}"


def _content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    system = [text for text in (_content(m) for m in messages if m.get("role") == "system") if text]
    turns = [text for text in (_content(m) for m in messages if m.get("role") == "user") if text]
    latest = turns[-1] if turns else (_content(messages[-1]) if messages else "")
    if system:
        return f"{''.join(_SYSTEM_PREAMBLE.format(part) for part in system)}{latest}"
    return latest


def _tool_call_block(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    name = str(function.get("name") or "tool")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return f"[tool call {call.get('id') or ''}] {name}({arguments})"


def render_transcript(messages: list[dict[str, Any]], tools: Any = None) -> str:
    """Render the whole OpenAI conversation as one OpenCode prompt.

    The harness is stateless (no --session reuse), so every turn must carry the
    tool results that Hermes executed locally.
    """
    blocks: list[str] = []
    for message in messages:
        role = str(message.get("role") or "")
        text = _content(message).strip()
        if role == "system":
            if text:
                blocks.append(_SYSTEM_PREAMBLE.format(text).rstrip())
        elif role == "user":
            blocks.append(f"USER:\n{text}" if text else "USER:\n(empty)")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                lines = ["ASSISTANT (previous turn, already handled by the host):"]
                if text:
                    lines.append(text)
                lines.append("TOOL CALLS the host already executed:")
                lines.extend(_tool_call_block(call) for call in calls)
                blocks.append("\n".join(lines))
            else:
                blocks.append(f"ASSISTANT:\n{text}" if text else "ASSISTANT:")
        elif role == "tool":
            label = message.get("tool_call_id") or message.get("name") or "tool"
            blocks.append(f"TOOL RESULT [{label}] (authoritative, already executed):\n{text}")
        elif text:
            blocks.append(f"{role.upper()}:\n{text}")
    tool_names = []
    if isinstance(tools, list):
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name"):
                tool_names.append(str(function["name"]))
    if tool_names:
        blocks.append(
            "AVAILABLE HOST TOOLS: " + ", ".join(tool_names) + "\n" + _INTERCEPT_GUIDANCE
        )
    else:
        blocks.append(_INTERCEPT_GUIDANCE)
    return "\n\n".join(blocks)


def _parse_events(stdout: str) -> tuple[str, list[dict[str, Any]], str | None]:
    """Return (text, raw tool calls, session id) from `opencode run --format json`."""
    chunks: list[str] = []
    calls: list[dict[str, Any]] = []
    session: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        session = event.get("sessionID") or session
        part = event.get("part") or {}
        kind = event.get("type")
        if kind == "text" and part.get("type") == "text":
            chunks.append(part.get("text", ""))
        elif kind in {"tool_use", "tool"} and part.get("type") == "tool":
            state = part.get("state") or {}
            arguments = state.get("input")
            if not isinstance(arguments, dict) or not arguments:
                continue
            name = str(part.get("tool") or part.get("name") or "").strip()
            if not name:
                continue
            calls.append(
                {
                    "id": str(part.get("id") or f"call_{len(calls)}"),
                    "name": name,
                    "arguments": arguments,
                }
            )
    return "".join(chunks).strip(), calls, session



def _usage(total: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=total,
        completion_tokens=0,
        total_tokens=total,
    )


def _response(text: str, model: str, session: str, total: int = 0) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=text, tool_calls=[])
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        id=session,
        model=model,
        choices=[choice],
        usage=_usage(total),
    )


class HarnessClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, **_: Any) -> None:
        self._session: str | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def close(self) -> None:
        return None

    def _run(
        self, reference: str, prompt: str, timeout: float, *, intercept: bool = False
    ) -> str:
        binary = opencode_binary()
        if binary is None:
            raise HarnessUnavailable(
                "opencode CLI not found; set OPENCODE_BIN or install opencode to use free models"
            )
        command = [
            binary,
            "run",
            "--format",
            "json",
        ]
        if intercept:
            # Without --standalone the client delegates to the shared `opencode
            # serve --service` daemon, which ignores our OPENCODE_CONFIG_CONTENT
            # and happily runs every tool. Standalone applies the deny/ask
            # permission config, so the call is rejected before execution while
            # the tool_use event still carries name + input.
            command.append("--standalone")
        else:
            command.append("--auto")
        command.extend(["-m", reference])
        if self._session and not intercept:
            command.extend(["--session", self._session])
        env = os.environ.copy()
        env.pop("PWD", None)
        env.pop("OLDPWD", None)
        if intercept:
            env["OPENCODE_CONFIG_CONTENT"] = _INTERCEPT_CONFIG
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
                cwd=os.path.expanduser("~"),
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessError(
                f"opencode harness timed out after {timeout}s", status_code=504
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-500:]
            raise HarnessError(
                f"opencode harness failed with exit {completed.returncode}: {detail}",
                status_code=502,
            )
        return completed.stdout

    def _parse(self, stdout: str) -> tuple[str, str | None]:
        text, _calls, session = _parse_events(stdout)
        return text, session

    def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float = DEFAULT_TIMEOUT,
        tools: Any = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> SimpleNamespace:
        reference = model_reference(model)
        intercept = tool_intercept_enabled()
        if intercept:
            prompt = render_transcript(messages, tools)
            stdout = self._run(
                reference, prompt, timeout or DEFAULT_TIMEOUT, intercept=True
            )
            return self._tool_response(stdout, model)
        prompt = prompt_from_messages(messages)
        text, session = self._parse(
            self._run(reference, prompt, timeout or DEFAULT_TIMEOUT)
        )
        if session:
            self._session = session
        return _response(text, model, self._session or "", len(text))

    def _tool_response(self, stdout: str, model: str) -> SimpleNamespace:
        text, raw_calls, session = _parse_events(stdout)
        if session:
            self._session = session
        tool_calls: list[SimpleNamespace] = []
        seen: set[str] = set()
        for call in raw_calls:
            name = _OC_TO_HERMES_TOOL.get(call["name"], call["name"])
            arguments = json.dumps(call["arguments"], ensure_ascii=False)
            fingerprint = f"{name}:{arguments}"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            tool_calls.append(
                SimpleNamespace(
                    id=call["id"],
                    type="function",
                    function=SimpleNamespace(name=name, arguments=arguments),
                )
            )
            if len(tool_calls) >= _MAX_TOOL_CALLS:
                break
        finish_reason = "tool_calls" if tool_calls else "stop"
        message = SimpleNamespace(
            role="assistant",
            content=text or None,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
        return SimpleNamespace(
            id=self._session or "",
            model=model,
            choices=[choice],
            usage=_usage(len(text)),
        )

    def list_models(self, timeout: float = 30.0) -> list[str]:
        binary = opencode_binary()
        if binary is None:
            raise HarnessUnavailable(
                "opencode CLI not found; set OPENCODE_BIN or install opencode to use free models"
            )
        env = os.environ.copy()
        env.pop("PWD", None)
        env.pop("OLDPWD", None)
        completed = subprocess.run(
            [binary, "models"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
            cwd=os.path.expanduser("~"),
        )
        if completed.returncode != 0:
            raise HarnessError(
                f"opencode models failed with exit {completed.returncode}",
                status_code=502,
            )
        prefix = f"{DEFAULT_PROVIDER}/"
        found = [
            line[len(prefix):]
            for line in completed.stdout.splitlines()
            if line.startswith(prefix) and line.endswith(FREE_MODEL_SUFFIX)
        ]
        return sorted(set(found))
