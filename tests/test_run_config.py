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

"""Tests for run_config."""

from __future__ import annotations

import os
import unittest
from argparse import Namespace
from unittest.mock import patch

from scalablerag.run_config import detect_explicit_flags, resolve_run_config


class RunConfigTests(unittest.TestCase):
    def test_zero_regime_defaults(self):
        cfg = resolve_run_config(Namespace(regime="zero"))
        self.assertTrue(cfg.skip_embeddings)
        self.assertFalse(cfg.discover_patterns)
        self.assertFalse(cfg.curate_patterns)
        self.assertTrue(cfg.use_agg_tools)
        self.assertTrue(cfg.branch_select)
        self.assertFalse(cfg.require_embeddings)

    def test_limited_regime_defaults(self):
        cfg = resolve_run_config(Namespace(regime="limited"))
        self.assertTrue(cfg.skip_embeddings)
        self.assertTrue(cfg.discover_patterns)
        self.assertTrue(cfg.curate_patterns)
        self.assertFalse(cfg.require_embeddings)
        self.assertEqual(cfg.embed_provider, "openai")
        self.assertEqual(cfg.embed_model, "text-embedding-3-small")
        self.assertTrue(cfg.use_tiered_filters)
        self.assertTrue(cfg.require_lexical_index)

    def test_explicit_flags_override_regime(self):
        argv = [
            "eval.py",
            "--regime",
            "limited",
            "--no-discover-patterns",
            "--with-embeddings",
        ]
        explicit = detect_explicit_flags(argv)
        cfg = resolve_run_config(
            Namespace(
                regime="limited",
                with_embeddings=True,
                discover_patterns=False,
            ),
            explicit,
        )
        self.assertFalse(cfg.skip_embeddings)
        self.assertFalse(cfg.discover_patterns)
        self.assertTrue(cfg.require_embeddings)
        self.assertTrue(cfg.use_tiered_filters)

    def test_env_overrides_when_no_explicit_flag(self):
        with patch.dict(
            os.environ,
            {"SR_BRANCH_SELECT": "0", "SR_DISCOVER_PATTERNS": "1"},
            clear=False,
        ):
            cfg = resolve_run_config(Namespace(regime=None))
        self.assertFalse(cfg.branch_select)
        self.assertTrue(cfg.discover_patterns)

    def test_explicit_with_embeddings_requires_store(self):
        explicit = detect_explicit_flags(["eval.py", "--with-embeddings"])
        cfg = resolve_run_config(Namespace(with_embeddings=True), explicit)
        self.assertFalse(cfg.skip_embeddings)
        self.assertTrue(cfg.require_embeddings)

    def test_detect_explicit_flags(self):
        argv = ["eval.py", "--with-embeddings", "--branch-select"]
        explicit = detect_explicit_flags(argv)
        self.assertEqual(explicit, {"with_embeddings", "branch_select"})

    def test_branch_select_default_on_without_regime(self):
        cfg = resolve_run_config(Namespace(), set())
        self.assertTrue(cfg.branch_select)

    def test_explicit_no_tiered_filters_overrides_limited(self):
        argv = ["query.py", "--regime", "limited", "--no-tiered-filters"]
        cfg = resolve_run_config(
            Namespace(regime="limited", tiered_filters=False),
            detect_explicit_flags(argv),
        )
        self.assertFalse(cfg.use_tiered_filters)
        self.assertFalse(cfg.require_lexical_index)

    def test_explicit_tiered_filters_requires_index(self):
        argv = ["query.py", "--tiered-filters"]
        cfg = resolve_run_config(
            Namespace(regime=None, tiered_filters=True),
            detect_explicit_flags(argv),
        )
        self.assertTrue(cfg.use_tiered_filters)
        self.assertTrue(cfg.require_lexical_index)


if __name__ == "__main__":
    unittest.main()
