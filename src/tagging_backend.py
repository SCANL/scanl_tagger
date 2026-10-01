from spiral import ronin

from src.lm_based_tagger.distilbert_tagger import DistilBertTagger


class TaggingBackend:
    def __init__(
        self,
        model_path: str,
        local: bool = False,
        pattern_postprocessing: bool | None = None,
    ):
        if not model_path:
            raise ValueError("Tagging requires a model path or HuggingFace repo id.")

        self.pattern_postprocessing = pattern_postprocessing
        self.lm_model = DistilBertTagger(
            model_path,
            local=local,
            pattern_postprocessing=pattern_postprocessing,
        )

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
        tags = self.lm_model.tag_identifier(
            tokens=words,
            context=context,
            type_str=type_str,
            language=language,
            system_name=system_name,
            pattern_postprocessing=pattern_postprocessing,
        )
        return {"tokens": words, "tags": list(tags)}

    def tag_identifier_batch(self, records, batch_size: int = 64):
        rows = [
            {
                "tokens": ronin.split(record["identifier_name"]),
                "context": record["context"],
                "type_str": record.get("type_str", ""),
                "language": record.get("language", ""),
                "system_name": record.get("system_name", ""),
                "pattern_postprocessing": record.get("pattern_postprocessing"),
            }
            for record in records
        ]
        batch_predictions = self.lm_model.tag_identifiers(rows, batch_size=batch_size)
        return [
            {"tokens": row["tokens"], "tags": tags}
            for row, tags in zip(rows, batch_predictions)
        ]
