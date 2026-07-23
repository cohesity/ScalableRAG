#    Copyright (c) Cohesity 2026. All rights reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
 
#        http://www.apache.org/licenses/LICENSE-2.0
 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""CLI + run_config tests."""

from __future__ import annotations

import argparse
import unittest

from scalablerag.llm.factory import KNOWN_LLM_PROVIDERS
from scalablerag.run_config import detect_explicit_flags, resolve_run_config


def _eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["zero", "limited"], default=None)
    p.add_argument("--embeddings", action=argparse.BooleanOptionalAction)
    p.add_argument("--discover-patterns", action=argparse.BooleanOptionalAction)
    p.add_argument("--curate-patterns", action=argparse.BooleanOptionalAction)
    p.add_argument("--agg-tools", action=argparse.BooleanOptionalAction)
    p.add_argument("--branch-select", action=argparse.BooleanOptionalAction)
    return p


class CliIntegrationTests(unittest.TestCase):
    def test_limited_with_no_embeddings_override(self):
        argv = [
            "eval.py",
            "--regime",
            "limited",
            "--no-embeddings",
            "--no-discover-patterns",
        ]
        args = _eval_parser().parse_args(argv[1:])
        cfg = resolve_run_config(args, detect_explicit_flags(argv))
        self.assertTrue(cfg.skip_embeddings)
        self.assertFalse(cfg.discover_patterns)
        self.assertFalse(cfg.require_embeddings)

    def test_zero_regime_via_parser(self):
        argv = ["eval.py", "--regime", "zero"]
        args = _eval_parser().parse_args(argv[1:])
        cfg = resolve_run_config(args, detect_explicit_flags(argv))
        self.assertTrue(cfg.skip_embeddings)
        self.assertTrue(cfg.branch_select)

    def test_explicit_embeddings_flag(self):
        argv = ["eval.py", "--embeddings"]
        args = _eval_parser().parse_args(argv[1:])
        cfg = resolve_run_config(args, detect_explicit_flags(argv))
        self.assertFalse(cfg.skip_embeddings)
        self.assertTrue(cfg.require_embeddings)


class JudgeCliTests(unittest.TestCase):
    def test_judge_accepts_known_llm_providers(self):
        p = argparse.ArgumentParser()
        p.add_argument(
            "--llm-provider",
            default=None,
            choices=list(KNOWN_LLM_PROVIDERS),
        )
        for prov in KNOWN_LLM_PROVIDERS:
            args = p.parse_args(["--llm-provider", prov])
            self.assertEqual(args.llm_provider, prov)


class IngestFlagTests(unittest.TestCase):
    def test_ingest_embeddings_default_on(self):
        p = argparse.ArgumentParser()
        p.add_argument(
            "--embeddings", action=argparse.BooleanOptionalAction, default=True
        )
        args = p.parse_args([])
        self.assertTrue(args.embeddings)

    def test_ingest_no_embeddings(self):
        p = argparse.ArgumentParser()
        p.add_argument(
            "--embeddings", action=argparse.BooleanOptionalAction, default=True
        )
        args = p.parse_args(["--no-embeddings"])
        self.assertFalse(args.embeddings)


if __name__ == "__main__":
    unittest.main()
