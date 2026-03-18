import os

import joblib
import pandas as pd
from spiral import ronin

from src.lm_based_tagger.distilbert_tagger import DistilBertTagger
from src.tree_based_tagger.create_models import createModel, mutable_feature_list
from src.tree_based_tagger.feature_generator import createFeatures, custom_to_numeric, universal_to_custom


def context_to_number(context):
    if context == "ATTRIBUTE":
        return 1
    if context == "CLASS":
        return 2
    if context == "DECLARATION":
        return 3
    if context == "FUNCTION":
        return 4
    if context == "PARAMETER":
        return 5
    raise ValueError(f"Unsupported context: {context}")


def annotate_identifier(classifier, data):
    data = data.drop(columns=["WORD", "SPLIT_IDENTIFIER"], errors="ignore")

    trained_features = classifier.feature_names_in_
    missing_features = set(trained_features) - set(data.columns)
    extra_features = set(data.columns) - set(trained_features)

    if missing_features:
        raise ValueError(f"The following expected features are missing: {missing_features}")
    if extra_features:
        data = data[trained_features]

    return classifier.predict(data[trained_features])


class TaggingBackend:
    def __init__(
        self,
        source_dir: str,
        model_type: str = "tree_based",
        model_path: str | None = None,
        local: bool = False,
        pattern_postprocessing: bool | None = None,
    ):
        self.source_dir = source_dir
        self.model_type = model_type
        self.pattern_postprocessing = pattern_postprocessing
        self.lm_model = None
        self.model_gensim_english = None
        self.tree_classifier = None

        if self.model_type == "tree_based":
            _, _, self.model_gensim_english = createModel(rootDir=self.source_dir)
            classifier_path = os.path.join(
                self.source_dir,
                "..",
                "models",
                "model_GradientBoostingClassifier.pkl",
            )
            self.tree_classifier = joblib.load(classifier_path)
        elif self.model_type == "lm_based":
            if not model_path:
                raise ValueError("LM tagging requires a model path or HuggingFace repo id.")
            self.lm_model = DistilBertTagger(
                model_path,
                local=local,
                pattern_postprocessing=pattern_postprocessing,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def tag_identifier(
        self,
        identifier_name: str,
        context: str,
        type_str: str = "",
        language: str = "",
        system_name: str = "",
        pattern_postprocessing: bool | None = None,
    ) -> dict:
        words = ronin.split(identifier_name)
        if self.model_type == "lm_based":
            tags = self.lm_model.tag_identifier(
                tokens=words,
                context=context,
                type_str=type_str,
                language=language,
                system_name=system_name,
                pattern_postprocessing=pattern_postprocessing,
            )
        else:
            tags = self._tag_tree_identifier(words, context)

        return {"tokens": words, "tags": list(tags)}

    def tag_identifier_batch(self, records, batch_size: int = 64):
        prepared_records = []
        for record in records:
            prepared_records.append(
                {
                    "identifier_name": record["identifier_name"],
                    "tokens": ronin.split(record["identifier_name"]),
                    "context": record["context"],
                    "type_str": record.get("type_str", ""),
                    "language": record.get("language", ""),
                    "system_name": record.get("system_name", ""),
                    "pattern_postprocessing": record.get("pattern_postprocessing"),
                }
            )

        if self.model_type != "lm_based":
            return [
                {
                    "tokens": item["tokens"],
                    "tags": list(self._tag_tree_identifier(item["tokens"], item["context"])),
                }
                for item in prepared_records
            ]

        rows = [
            {
                "tokens": item["tokens"],
                "context": item["context"],
                "type_str": item["type_str"],
                "language": item["language"],
                "system_name": item["system_name"],
                "pattern_postprocessing": item["pattern_postprocessing"],
            }
            for item in prepared_records
        ]
        batch_predictions = self.lm_model.tag_identifiers(rows, batch_size=batch_size)
        return [
            {"tokens": item["tokens"], "tags": tags}
            for item, tags in zip(prepared_records, batch_predictions)
        ]

    def _tag_tree_identifier(self, words, context):
        data = pd.DataFrame(
            {
                "WORD": words,
                "SPLIT_IDENTIFIER": " ".join(words),
                "CONTEXT_NUMBER": context_to_number(context),
            }
        )

        data = createFeatures(
            data,
            mutable_feature_list,
            modelGensimEnglish=self.model_gensim_english,
        )

        categorical_features = ["NLTK_POS", "PREV_POS", "NEXT_POS"]
        for category_column in categorical_features:
            if category_column in data.columns:
                data[category_column] = data[category_column].astype(str)
                unique_vals = data[category_column].unique()
                category_map = {}
                for val in unique_vals:
                    if val in universal_to_custom:
                        category_map[val] = custom_to_numeric[universal_to_custom[val]]
                    else:
                        category_map[val] = custom_to_numeric["NOUN"]
                data[category_column] = data[category_column].map(category_map)

        return annotate_identifier(self.tree_classifier, data)