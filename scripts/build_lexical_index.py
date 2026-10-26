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

"""Build or rebuild the portable lexical index for an existing store."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalablerag.storage.doc_store import DocStore
from scalablerag.storage.lexical_index import LexicalIndex


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the SQLite FTS5 lexical index for a ScalableRAG store"
    )
    parser.add_argument("--store", required=True, help="Existing corpus store")
    args = parser.parse_args()

    store_path = Path(args.store)
    doc_store = DocStore(store_path / "docs")
    if doc_store.doc_count() == 0:
        raise SystemExit(f"No documents found under {store_path / 'docs'}")

    started = time.perf_counter()
    index = LexicalIndex.build(store_path, doc_store)
    elapsed = time.perf_counter() - started
    print(
        f"Built {index.path} for {doc_store.doc_count()} documents "
        f"({doc_store.total_bytes():,} bytes) in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
