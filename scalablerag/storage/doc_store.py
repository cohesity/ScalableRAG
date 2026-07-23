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

"""Document store: persistent storage for corpus documents and metadata.

Designed for up to 100k+ documents. Uses a JSON-lines metadata index
and stores original text files on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator


@dataclass
class DocMeta:
    """Metadata for a single document."""

    doc_id: str
    filename: str
    char_count: int = 0
    word_count: int = 0
    estimated_tokens: int = 0
    extra: dict = field(default_factory=dict)


class DocStore:
    """On-disk document store.

    Layout::

        store_dir/
            meta.jsonl          # one DocMeta JSON per line
            texts/
                <doc_id>.txt    # raw text
    """

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.texts_dir = self.store_dir / "texts"
        self._meta: dict[str, DocMeta] = {}
        self._doc_ids: list[str] = []
        if self.store_dir.exists():
            self._load_index()

    def _load_index(self) -> None:
        idx_path = self.store_dir / "meta.jsonl"
        if not idx_path.exists():
            return
        with open(idx_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                dm = DocMeta(**obj)
                self._meta[dm.doc_id] = dm
                self._doc_ids.append(dm.doc_id)
        self._build_filename_index()

    def _build_filename_index(self) -> None:
        """Build reverse lookup: filename → doc_id."""
        self._fname_to_id: dict[str, str] = {}
        for did, meta in self._meta.items():
            self._fname_to_id[meta.filename] = did
            base = meta.filename.rsplit(".", 1)[0] if "." in meta.filename else meta.filename
            if base != meta.filename:
                self._fname_to_id[base] = did

    def resolve_doc_id(self, key: str) -> str | None:
        """Resolve a key to a valid doc_id.

        Accepts a doc_id, a filename, or a filename without extension.
        Returns the canonical doc_id, or None if unresolvable.
        """
        if key in self._meta:
            return key
        fname_map = getattr(self, "_fname_to_id", {})
        if key in fname_map:
            return fname_map[key]
        stripped = key.rstrip(".txt") if key.endswith(".txt") else key
        if stripped in self._meta:
            return stripped
        if stripped in fname_map:
            return fname_map[stripped]
        return None

    def _save_index(self) -> None:
        idx_path = self.store_dir / "meta.jsonl"
        with open(idx_path, "w", encoding="utf-8") as f:
            for doc_id in self._doc_ids:
                f.write(json.dumps(asdict(self._meta[doc_id])) + "\n")

    def init_store(self) -> None:
        """Create the store directory structure."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.texts_dir.mkdir(exist_ok=True)

    def add_document(
        self,
        doc_id: str,
        text: str,
        filename: str = "",
        extra: dict | None = None,
    ) -> DocMeta:
        """Add or replace a document."""
        self.init_store()
        words = text.split()
        meta = DocMeta(
            doc_id=doc_id,
            filename=filename,
            char_count=len(text),
            word_count=len(words),
            estimated_tokens=max(1, len(text) * 10 // 35),
            extra=extra or {},
        )
        text_path = self.texts_dir / f"{doc_id}.txt"
        text_path.write_text(text, encoding="utf-8")

        if doc_id not in self._meta:
            self._doc_ids.append(doc_id)
        self._meta[doc_id] = meta
        self._save_index()
        return meta

    def get_text(self, doc_id: str) -> str:
        """Read document text. Raises KeyError if not found."""
        if doc_id not in self._meta:
            raise KeyError(f"Unknown doc_id: {doc_id}")
        p = self.texts_dir / f"{doc_id}.txt"
        return p.read_text(encoding="utf-8")

    def get_meta(self, doc_id: str) -> DocMeta:
        if doc_id not in self._meta:
            raise KeyError(f"Unknown doc_id: {doc_id}")
        return self._meta[doc_id]

    def all_doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    def doc_count(self) -> int:
        return len(self._doc_ids)

    def iter_docs(self) -> Iterator[tuple[str, DocMeta]]:
        """Iterate (doc_id, meta) pairs."""
        for did in self._doc_ids:
            yield did, self._meta[did]

    def total_chars(self) -> int:
        return sum(m.char_count for m in self._meta.values())

    def total_estimated_tokens(self) -> int:
        return sum(m.estimated_tokens for m in self._meta.values())
