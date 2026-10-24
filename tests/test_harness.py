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
