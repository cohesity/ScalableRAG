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

"""Pattern library: stores validated parameterized regex templates.

A PatternTemplate is a parameterized regex that answers a specific
class of questions about a corpus.  For example:

    PatternTemplate(
        name="section_heading",
        description="Matches a numbered section heading with title",
        parameters=["section_number"],
        regex_template=r"(?:^|\\n)\\s*{section_number}[.)]\\s+([^\\n]+)",
        scope_regex="",
        output_type="extract",
    )

A PatternLibrary is a collection of validated templates that can be
serialized and loaded at inference time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class PatternTemplate:
    """A parameterized regex template for a class of corpus queries."""

    name: str
    description: str
    parameters: list[str]
    regex_template: str
    scope_regex: str = ""
    output_type: str = "extract"
    validation_score: float = 0.0
    validation_details: dict[str, Any] = field(default_factory=dict)

    def instantiate(self, **kwargs: str) -> str:
        """Fill parameters into the regex template.

        Parameter values are regex-escaped by default.  Pass raw regex
        by prefixing the key with ``raw_``.
        """
        regex = self.regex_template
        for param in self.parameters:
            raw_key = f"raw_{param}"
            if raw_key in kwargs:
                value = str(kwargs[raw_key] or "")
            elif param in kwargs and kwargs[param] is not None:
                value = re.escape(str(kwargs[param]))
            else:
                value = r".+?"
            regex = regex.replace(f"{{{param}}}", value)
        return regex


class PatternLibrary:
    """Collection of validated pattern templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PatternTemplate] = {}

    def add(self, template: PatternTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> PatternTemplate | None:
        return self._templates.get(name)

    def all_templates(self) -> list[PatternTemplate]:
        return list(self._templates.values())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(t) for t in self._templates.values()]
        path.write_text(json.dumps(data, indent=2))
        print(f"Saved {len(data)} patterns to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PatternLibrary":
        path = Path(path)
        data = json.loads(path.read_text())
        lib = cls()
        for d in data:
            lib.add(PatternTemplate(**d))
        return lib

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return f"PatternLibrary({len(self._templates)} patterns)"
