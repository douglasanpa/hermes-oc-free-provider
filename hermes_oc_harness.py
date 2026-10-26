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

    def _run(self, reference: str, prompt: str, timeout: float) -> tuple[str, str | None]:
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
            "--auto",
            "-m",
            reference,
        ]
        if self._session:
            command.extend(["--session", self._session])
        env = os.environ.copy()
        env.pop("PWD", None)
        env.pop("OLDPWD", None)
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
        return self._parse(completed.stdout)

    def _parse(self, stdout: str) -> tuple[str, str | None]:
        chunks: list[str] = []
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
            if event.get("type") == "text" and part.get("type") == "text":
                chunks.append(part.get("text", ""))
        return "".join(chunks), session

    def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float = DEFAULT_TIMEOUT,
        **_: Any,
    ) -> SimpleNamespace:
        reference = model_reference(model)
        prompt = prompt_from_messages(messages)
        text, session = self._run(reference, prompt, timeout or DEFAULT_TIMEOUT)
        if session:
            self._session = session
        return _response(text, model, self._session or "", len(text))

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
