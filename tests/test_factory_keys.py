#    Copyright (c) Cohesity 2026. All rights reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Tests for env-only API key resolution in factory."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import openai

from scalablerag.llm.factory import make_embedder, make_llm


class FactoryKeyTests(unittest.TestCase):
    def test_gemini_requires_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_KEY environment variable"):
                make_llm(provider="gemini")

    def test_openai_uses_env_not_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "OPENAI_KEY").write_text("file-only-key", encoding="utf-8")
            env = {"OPENAI_API_KEY": "env-key"}
            with mock.patch.dict(os.environ, env, clear=True):
                llm = make_llm(provider="openai", repo_root=root, api_key=None)
            self.assertEqual(llm.client.api_key, "env-key")

    def test_openai_key_file_is_not_used_when_env_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "OPENAI_KEY").write_text("file-only-key", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(openai.OpenAIError):
                    make_llm(provider="openai", repo_root=root, api_key=None)

    def test_embedder_uses_env_not_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "OPENAI_KEY").write_text("file-only-key", encoding="utf-8")
            env = {"OPENAI_API_KEY": "env-key"}
            with mock.patch.dict(os.environ, env, clear=True):
                embedder = make_embedder(
                    provider="openai", repo_root=root, api_key=None
                )
            self.assertEqual(embedder.client.api_key, "env-key")


if __name__ == "__main__":
    unittest.main()
