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

"""Tests for local LLM endpoint transport warnings."""

from __future__ import annotations

import unittest

from scalablerag.llm.factory import _warn_insecure_local_http


class FactoryTransportTests(unittest.TestCase):
    def test_warns_on_remote_http(self):
        import io
        import sys

        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            _warn_insecure_local_http("http://gpu-server.internal:11434", "Ollama")
        finally:
            sys.stderr = old
        msg = err.getvalue()
        self.assertIn("non-loopback", msg)
        self.assertIn("gpu-server.internal", msg)

    def test_silent_for_loopback_http(self):
        import io
        import sys

        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            _warn_insecure_local_http("http://127.0.0.1:11434", "Ollama")
            _warn_insecure_local_http("http://localhost:8000/v1", "vLLM")
        finally:
            sys.stderr = old
        self.assertEqual(err.getvalue(), "")

    def test_silent_for_https(self):
        import io
        import sys

        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            _warn_insecure_local_http("https://gpu.example.com:8000/v1", "vLLM")
        finally:
            sys.stderr = old
        self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
