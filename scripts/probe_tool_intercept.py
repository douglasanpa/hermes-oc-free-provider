"""Live probe for the tool-interception mode of hermes_oc_harness.

Usage (run from the plugin root or anywhere):

    HERMES_OC_TOOL_INTERCEPT=1 python scripts/probe_tool_intercept.py
    HERMES_OC_TOOL_INTERCEPT=0 python scripts/probe_tool_intercept.py   # legacy

Step 1 asks for a command that must be run; the probe asserts the harness
returns it as an OpenAI tool_call mapped to Hermes' `terminal`. Step 2 feeds the
tool result back and asserts the model answers without calling the tool again.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_oc_harness import HarnessClient, tool_intercept_enabled  # noqa: E402

MODEL = os.environ.get("POC_MODEL", "deepseek-v4.1-flash-free")
TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Execute a shell command on the host and return stdout.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def main() -> int:
    print(f"binary           : {shutil.which('opencode')}")
    print(f"intercept enabled: {tool_intercept_enabled()}")
    client = HarnessClient()

    messages: list[dict] = [
        {"role": "system", "content": "You are a Hermes agent. Use tools when needed."},
        {
            "role": "user",
            "content": (
                "Preciso do valor exato que o comando `echo POC_42` imprime. "
                "Use a tool terminal e nao invente o resultado."
            ),
        },
    ]

    started = time.time()
    first = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[TERMINAL_TOOL], timeout=180
    )
    message = first.choices[0].message
    calls = list(getattr(message, "tool_calls", []) or [])
    print(
        f"\n[1] {time.time() - started:.1f}s finish_reason={first.choices[0].finish_reason}"
    )
    print(f"    text: {(message.content or '')[:160]!r}")
    for call in calls:
        print(f"    tool_call: {call.function.name} {call.function.arguments}")

    if not calls:
        print("\nRESULTADO: FALHOU (nenhum tool_call devolvido)")
        return 1

    call = calls[0]
    arguments = json.loads(call.function.arguments)
    if call.function.name != "terminal" or "command" not in arguments:
        print("\nRESULTADO: FALHOU (nome/args nao mapeados)")
        return 1

    messages.append(
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "name": "terminal",
            "content": "POC_42\n",
        }
    )

    started = time.time()
    second = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[TERMINAL_TOOL], timeout=180
    )
    final = second.choices[0].message
    final_text = (final.content or "").strip()
    again = list(getattr(final, "tool_calls", []) or [])
    print(f"\n[2] {time.time() - started:.1f}s finish_reason={second.choices[0].finish_reason}")
    print(f"    text: {final_text[:220]!r}")
    print(f"    tool_calls: {len(again)}")
    for call2 in again:
        print(f"      repeat: {call2.function.name} {call2.function.arguments}")

    ok = "POC_42" in final_text and not again
    print("\nRESULTADO:", "OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
