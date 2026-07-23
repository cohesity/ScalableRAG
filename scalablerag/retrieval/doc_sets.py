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

"""Named document set management for the retrieval agent.

The agent creates and references named sets of documents as it narrows
down candidates through classifiers, regex filters, and labeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib


@dataclass
class DocSet:
    """A named set of document IDs with provenance."""

    name: str
    doc_ids: list[str]
    source: str = ""
    description: str = ""
    filter_regex: str = ""


class DocSetRegistry:
    """Registry of named document sets.

    Provides 'all' as a built-in set referencing the full corpus.
    Supports set algebra: intersect, union, difference.
    """

    def __init__(self, all_doc_ids: list[str]):
        self._all_ids = list(all_doc_ids)
        self._sets: dict[str, DocSet] = {
            "all": DocSet(
                name="all",
                doc_ids=list(all_doc_ids),
                source="corpus",
                description="All documents in the corpus",
            )
        }

    def register(
        self,
        name: str,
        doc_ids: list[str],
        source: str = "",
        description: str = "",
        filter_regex: str = "",
    ) -> DocSet:
        ds = DocSet(
            name=name, doc_ids=list(doc_ids), source=source,
            description=description, filter_regex=filter_regex,
        )
        self._sets[name] = ds
        return ds

    def get_filter_regex(self, name: str) -> str:
        """Return the regex that produced a set, or empty string."""
        if name in self._sets:
            return self._sets[name].filter_regex
        return ""

    def get(self, name: str) -> DocSet:
        if name not in self._sets:
            candidates = difflib.get_close_matches(
                name, self._sets.keys(), n=3, cutoff=0.0
            )
            hint = f" Did you mean: {candidates}?" if candidates else ""
            raise KeyError(
                f"Unknown set: {name!r}.{hint} Available: {list(self._sets.keys())}"
            )
        return self._sets[name]

    def resolve_ids(self, set_name: str) -> list[str]:
        # Allow set expressions in names (deterministic parse).
        #
        # Supported forms (single binary op, recursively):
        #   A_and_B   -> intersect(A, B)
        #   A_or_B    -> union(A, B)
        #   A_minus_B -> difference(A, B)
        #
        # If the expression cannot be resolved, fall back to normal lookup.
        if set_name in self._sets:
            return self._sets[set_name].doc_ids

        def _try_split(op: str) -> tuple[str, str] | None:
            token = f"_{op}_"
            if token not in set_name:
                return None
            left, right = set_name.split(token, 1)
            if not left or not right:
                return None
            return left, right

        # Try recursively resolving expressions
        for op in ("and", "or", "minus"):
            parts = _try_split(op)
            if not parts:
                continue
            a, b = parts
            try:
                if op == "and":
                    a_ids = set(self.resolve_ids(a))
                    b_ordered = self.resolve_ids(b)
                    return [d for d in b_ordered if d in a_ids]
                if op == "or":
                    a_ordered = self.resolve_ids(a)
                    b_ordered = self.resolve_ids(b)
                    seen = set(a_ordered)
                    out = list(a_ordered)
                    for d in b_ordered:
                        if d not in seen:
                            out.append(d)
                            seen.add(d)
                    return out
                if op == "minus":
                    b_ids = set(self.resolve_ids(b))
                    return [d for d in self.resolve_ids(a) if d not in b_ids]
            except KeyError:
                # If sub-expression fails, try next op / fallback.
                pass

        return self.get(set_name).doc_ids

    def intersect(self, name_a: str, name_b: str) -> list[str]:
        a = set(self.resolve_ids(name_a))
        b_ordered = self.resolve_ids(name_b)
        return [d for d in b_ordered if d in a]

    def union(self, name_a: str, name_b: str) -> list[str]:
        a_ordered = self.resolve_ids(name_a)
        b_ordered = self.resolve_ids(name_b)
        seen = set(a_ordered)
        result = list(a_ordered)
        for d in b_ordered:
            if d not in seen:
                result.append(d)
                seen.add(d)
        return result

    def difference(self, name_a: str, name_b: str) -> list[str]:
        b = set(self.resolve_ids(name_b))
        return [d for d in self.resolve_ids(name_a) if d not in b]

    def list_sets(self) -> list[dict]:
        return [
            {
                "name": ds.name,
                "count": len(ds.doc_ids),
                "source": ds.source,
                "description": ds.description,
            }
            for ds in self._sets.values()
        ]

    def set_names(self) -> list[str]:
        return list(self._sets.keys())

    @property
    def all_ids(self) -> list[str]:
        return self._all_ids
