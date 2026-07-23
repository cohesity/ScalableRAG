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

"""Auto-pattern discovery: learn parameterized regex patterns from a corpus.

Preprocessing step that reads documents, discovers structural patterns,
builds parameterized regex templates, and validates them against the corpus.
The resulting pattern library is used at inference time by the retrieval agent.
"""

from .discovery import PatternDiscovery
from .extraction_index import ExtractionIndex, build_extraction_index
from .patterns import PatternLibrary, PatternTemplate
from .curation import (
    CurationResult,
    curate_patterns,
    full_curation_pipeline,
    gather_pattern_stats,
    discover_gaps,
    deduplicate_patterns,
    estimate_concept_coverage,
)

__all__ = [
    "PatternDiscovery", "PatternLibrary", "PatternTemplate",
    "ExtractionIndex", "build_extraction_index",
    "CurationResult", "curate_patterns", "full_curation_pipeline",
    "gather_pattern_stats",
    "discover_gaps", "deduplicate_patterns", "estimate_concept_coverage",
]
