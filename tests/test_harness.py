from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hermes_oc_harness as harness  # type: ignore[import-not-found]


class ModelReferenceTests(unittest.TestCase):
    def test_bare_model_gets_default_provider(self):
        self.assertEqual(harness.model_reference("beam-free"), "opencode/beam-free")

    def test_qualified_model_is_untouched(self):
        self.assertEqual(
            harness.model_reference("opencode/beam-free"), "opencode/beam-free"
        )


class PromptTests(unittest.TestCase):
    def test_system_prompt_becomes_a_preamble(self):
        prompt = harness.prompt_from_messages(
            [
                {"role": "system", "content": "You are Hermes"},
                {"role": "user", "content": "hello"},
            ]
        )
        self.assertIn("You are Hermes", prompt)
        self.assertTrue(prompt.endswith("hello"))

    def test_only_the_latest_user_turn_is_sent(self):
        prompt = harness.prompt_from_messages(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        )
        self.assertEqual(prompt, "second")

    def test_structured_content_is_flattened(self):
        prompt = harness.prompt_from_messages(
            [{"role": "user", "content": [{"type": "text", "text": "part one"}]}]
        )
        self.assertEqual(prompt, "part one")


class ParseTests(unittest.TestCase):
    def test_collects_text_parts_and_session(self):
        stdout = "\n".join(
            [
                '{"type":"step_start","sessionID":"ses_1","part":{"type":"step-start"}}',
                '{"type":"text","sessionID":"ses_1","part":{"type":"text","text":"he"}}',
                '{"type":"text","sessionID":"ses_1","part":{"type":"text","text":"llo"}}',
                "not json",
            ]
        )
        text, session = harness.HarnessClient()._parse(stdout)
        self.assertEqual(text, "hello")
        self.assertEqual(session, "ses_1")

    def test_ignores_non_text_events(self):
        text, session = harness.HarnessClient()._parse(
            '{"type":"tool","sessionID":"ses_2","part":{"type":"tool-use"}}'
        )
        self.assertEqual(text, "")
        self.assertEqual(session, "ses_2")


class ClientTests(unittest.TestCase):
    def setUp(self):
        # The legacy path must be deterministic even when the gateway/shell
        # exports HERMES_OC_TOOL_INTERCEPT=1 for real runs.
        patcher = mock.patch.dict(os.environ, {harness.TOOL_INTERCEPT_ENV: "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_session_is_reused_across_calls(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"type":"text","sessionID":"ses_keep","part":{"type":"text","text":"hi"}}',
            stderr="",
        )
        client = harness.HarnessClient()
        with (
            mock.patch.object(harness, "opencode_binary", return_value="/usr/bin/opencode"),
            mock.patch.object(subprocess, "run", return_value=completed) as run,
        ):
            client.chat.completions.create(model="beam-free", messages=[{"role": "user", "content": "a"}])
            client.chat.completions.create(model="beam-free", messages=[{"role": "user", "content": "b"}])

        first, second = run.call_args_list
        self.assertNotIn("--session", first.args[0])
        self.assertIn("--session", second.args[0])
        self.assertEqual(second.args[0][second.args[0].index("--session") + 1], "ses_keep")

    def test_missing_binary_raises_harness_unavailable(self):
        client = harness.HarnessClient()
        with mock.patch.object(harness, "opencode_binary", return_value=None):
            with self.assertRaises(harness.HarnessUnavailable):
                client.chat.completions.create(
                    model="beam-free", messages=[{"role": "user", "content": "a"}]
                )

    def test_nonzero_exit_surfaces_stderr(self):
        completed = SimpleNamespace(returncode=1, stdout="", stderr="Model unavailable")
        client = harness.HarnessClient()
        with (
            mock.patch.object(harness, "opencode_binary", return_value="/usr/bin/opencode"),
            mock.patch.object(subprocess, "run", return_value=completed),
        ):
            with self.assertRaises(harness.HarnessError) as caught:
                client.chat.completions.create(
                    model="beam-free", messages=[{"role": "user", "content": "a"}]
                )
        self.assertIn("Model unavailable", str(caught.exception))

    def test_response_exposes_openai_shaped_result(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"type":"text","sessionID":"ses_x","part":{"type":"text","text":"OK"}}',
            stderr="",
        )
        client = harness.HarnessClient()
        with (
            mock.patch.object(harness, "opencode_binary", return_value="/usr/bin/opencode"),
            mock.patch.object(subprocess, "run", return_value=completed),
        ):
            result = client.chat.completions.create(
                model="beam-free", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(result.choices[0].message.content, "OK")
        self.assertEqual(result.choices[0].message.role, "assistant")
        self.assertEqual(result.choices[0].message.tool_calls, [])
        self.assertEqual(result.choices[0].finish_reason, "stop")
        self.assertEqual(result.model, "beam-free")

    def test_binary_env_override(self):
        with mock.patch.dict(os.environ, {harness.OPENCODE_BIN_ENV: "/opt/opencode"}):
            with mock.patch.object(harness.os.path, "isfile", return_value=True):
                with mock.patch.object(harness.os, "access", return_value=True):
                    self.assertEqual(harness.opencode_binary(), "/opt/opencode")


class ToolInterceptTests(unittest.TestCase):
    STDOUT = "\n".join(
        [
            '{"type":"step_start","sessionID":"ses_t","part":{"type":"step-start"}}',
            '{"type":"text","sessionID":"ses_t","part":{"type":"text","text":"vou listar"}}',
            (
                '{"type":"tool_use","sessionID":"ses_t","part":{"type":"tool",'
                '"id":"call_1","tool":"shell","state":{"status":"error",'
                '"input":{"command":"echo hi"},'
                '"error":"This non-interactive run cannot ask the user for permission"}}}'
            ),
            (
                '{"type":"tool_use","sessionID":"ses_t","part":{"type":"tool",'
                '"id":"call_2","tool":"shell","state":{"status":"error",'
                '"input":{"command":"echo hi"}}}}'
            ),
            (
                '{"type":"tool_use","sessionID":"ses_t","part":{"type":"tool",'
                '"id":"call_3","tool":"read","state":{"status":"error",'
                '"input":{"filePath":"a.txt"}}}}'
            ),
        ]
    )

    def _run(self, stdout: str = STDOUT, **create_kwargs):
        completed = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        client = harness.HarnessClient()
        with (
            mock.patch.object(harness, "opencode_binary", return_value="/usr/bin/opencode"),
            mock.patch.object(subprocess, "run", return_value=completed) as run,
            mock.patch.dict(os.environ, {harness.TOOL_INTERCEPT_ENV: "1"}),
        ):
            result = client.chat.completions.create(
                model="beam-free",
                messages=[{"role": "user", "content": "a"}],
                **create_kwargs,
            )
        return result, run

    def test_tool_events_become_openai_tool_calls(self):
        result, _ = self._run()
        message = result.choices[0].message
        self.assertEqual(result.choices[0].finish_reason, "tool_calls")
        names = [call.function.name for call in message.tool_calls]
        self.assertEqual(names, ["terminal", "read_file"])
        self.assertIn('"command": "echo hi"', message.tool_calls[0].function.arguments)
        self.assertEqual(message.content, "vou listar")

    def test_duplicate_calls_are_collapsed(self):
        result, _ = self._run()
        self.assertEqual(len(result.choices[0].message.tool_calls), 2)

    def test_intercept_uses_permission_config_and_no_auto(self):
        _, run = self._run()
        command = run.call_args.args[0]
        self.assertNotIn("--auto", command)
        # --standalone is what makes OPENCODE_CONFIG_CONTENT actually apply;
        # without it the shared `serve --service` daemon ignores permissions.
        self.assertIn("--standalone", command)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["OPENCODE_CONFIG_CONTENT"], '{"permission":{"*":"ask"}}')

    def test_plain_text_keeps_legacy_path(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"type":"text","sessionID":"ses_l","part":{"type":"text","text":"OK"}}',
            stderr="",
        )
        client = harness.HarnessClient()
        with (
            mock.patch.object(harness, "opencode_binary", return_value="/usr/bin/opencode"),
            mock.patch.object(subprocess, "run", return_value=completed) as run,
            mock.patch.dict(os.environ, {harness.TOOL_INTERCEPT_ENV: "0"}),
        ):
            result = client.chat.completions.create(
                model="beam-free", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertIn("--auto", run.call_args.args[0])
        self.assertEqual(result.choices[0].message.content, "OK")
        self.assertEqual(result.choices[0].message.tool_calls, [])


class TranscriptTests(unittest.TestCase):
    def test_tool_results_and_prior_calls_are_rendered(self):
        prompt = harness.render_transcript(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "run it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "terminal", "parameters": {"type": "object"}},
                }
            ],
        )
        self.assertIn("USER:\nrun it", prompt)
        self.assertIn("terminal({\"command\":\"ls\"})", prompt)
        self.assertIn("TOOL CALLS the host already executed:", prompt)
        self.assertIn("TOOL RESULT [c1] (authoritative, already executed):\na.txt", prompt)
        self.assertIn("AVAILABLE HOST TOOLS: terminal", prompt)
        self.assertIn(harness._INTERCEPT_GUIDANCE, prompt)


class ListModelsTests(unittest.TestCase):
    def test_filters_to_free_opencode_models(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "opencode/beam-free\n"
                "opencode/gpt-6-sol\n"
                "opencode/mimo-v2.6-flash-free\n"
                "novita-ai/some/model\n"
            ),
            stderr="",
        )
        with (
            mock.patch.object(harness, "opencode_binary", return_value="/usr/bin/opencode"),
            mock.patch.object(subprocess, "run", return_value=completed),
        ):
            self.assertEqual(
                harness.HarnessClient().list_models(),
                ["beam-free", "mimo-v2.6-flash-free"],
            )
