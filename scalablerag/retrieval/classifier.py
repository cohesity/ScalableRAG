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

"""Classifier training and inference for document filtering.

Supports XGBoost, logistic regression, and random forest on document
embeddings with labels produced by the labeling/regex tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClassifierResult:
    """Result of training a classifier."""

    classifier_id: str
    classifier_type: str
    accuracy: float
    num_train: int
    num_positive: int
    num_negative: int
    feature_dim: int
    extra: dict


class ClassifierStore:
    """Train, store, and apply classifiers on document embeddings."""

    def __init__(self):
        self._classifiers: dict[str, Any] = {}
        self._results: dict[str, ClassifierResult] = {}
        self._counter = 0

    def train(
        self,
        classifier_type: str,
        embeddings: np.ndarray,
        labels: np.ndarray,
        doc_ids: list[str],
        classifier_id: str | None = None,
    ) -> ClassifierResult:
        """Train a binary classifier on embeddings + labels.

        Args:
            classifier_type: 'xgboost', 'logistic', or 'random_forest'.
            embeddings: (N, D) feature matrix.
            labels: (N,) binary labels (0/1).
            doc_ids: Document IDs corresponding to each row.
            classifier_id: Optional explicit ID.

        Returns:
            ClassifierResult with accuracy and metadata.
        """
        assert embeddings.shape[0] == len(labels) == len(doc_ids)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if classifier_id is None:
            self._counter += 1
            classifier_id = f"clf_{self._counter}"

        ct = classifier_type.lower().strip()

        if ct == "xgboost":
            clf, acc, extra = self._train_xgboost(embeddings, labels)
        elif ct in {"logistic", "logistic_regression"}:
            clf, acc, extra = self._train_logistic(embeddings, labels)
        elif ct in {"random_forest", "rf"}:
            clf, acc, extra = self._train_random_forest(embeddings, labels)
        else:
            raise ValueError(
                f"Unknown classifier type: {ct}. "
                "Use 'xgboost', 'logistic', or 'random_forest'."
            )

        self._classifiers[classifier_id] = clf
        result = ClassifierResult(
            classifier_id=classifier_id,
            classifier_type=ct,
            accuracy=acc,
            num_train=len(labels),
            num_positive=n_pos,
            num_negative=n_neg,
            feature_dim=embeddings.shape[1],
            extra=extra,
        )
        self._results[classifier_id] = result
        return result

    def predict(
        self,
        classifier_id: str,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Predict binary labels for new embeddings. Returns (N,) array of 0/1."""
        if classifier_id not in self._classifiers:
            raise KeyError(f"Unknown classifier: {classifier_id}")
        clf = self._classifiers[classifier_id]
        return (clf.predict(embeddings) >= 0.5).astype(int)

    def get_result(self, classifier_id: str) -> ClassifierResult:
        if classifier_id not in self._results:
            raise KeyError(f"Unknown classifier: {classifier_id}")
        return self._results[classifier_id]

    def list_classifiers(self) -> list[dict]:
        return [
            {
                "id": r.classifier_id,
                "type": r.classifier_type,
                "accuracy": round(r.accuracy, 4),
                "train_size": r.num_train,
                "pos": r.num_positive,
                "neg": r.num_negative,
            }
            for r in self._results.values()
        ]

    # -- private training methods ----------------------------------------

    def _train_xgboost(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[Any, float, dict]:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "xgboost required for XGBoost classifier. Install: pip install xgboost"
            )

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = float((preds == y).mean())
        return clf, acc, {"n_estimators": 100, "max_depth": 4}

    def _train_logistic(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[Any, float, dict]:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = float((preds == y).mean())
        return clf, acc, {"C": 1.0}

    def _train_random_forest(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[Any, float, dict]:
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = float((preds == y).mean())
        return clf, acc, {"n_estimators": 100, "max_depth": 6}
