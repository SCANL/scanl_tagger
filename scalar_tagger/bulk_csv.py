from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


csv.field_size_limit(sys.maxsize)

CLASS_TYPE_VALUES = {"class", "struct", "enum class", "interface", "union", "enum"}
CLASS_CATEGORY_VALUES = {"class", "struct", "interface", "union", "enum"}
FUNCTION_CATEGORY_VALUES = {"function", "constructor", "destructor"}
PARAMETER_CATEGORY_VALUES = {"parameter", "function-parameter"}
ATTRIBUTE_CATEGORY_VALUES = {"field"}
DECLARATION_CATEGORY_VALUES = {"global", "local", "typedef"}


@dataclass(frozen=True)
class IdentifierRequest:
    identifier_name: str
    context: str
    type_str: str = ""
    language: str = ""
    system_name: str = ""
    pattern_postprocessing: bool | None = None


@dataclass(frozen=True)
class TaggedIdentifier:
    identifier_name: str
    context: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    type_str: str = ""
    language: str = ""
    system_name: str = ""


def iter_csv_input_paths(input_path: Path, selected_files: list[str] | None = None) -> list[Path]:
    resolved_input = input_path.resolve()
    if resolved_input.is_file():
        return [resolved_input]

    if not resolved_input.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    if selected_files:
        csv_paths = []
        for file_name in selected_files:
            candidate = resolved_input / file_name
            if not candidate.exists():
                raise FileNotFoundError(f"Input file not found: {candidate}")
            csv_paths.append(candidate.resolve())
        return csv_paths

    return sorted(path.resolve() for path in resolved_input.glob("*.csv"))


def infer_context(raw_type: str, raw_category: str) -> str | None:
    normalized_type = raw_type.strip().casefold()
    normalized_category = raw_category.strip().casefold()

    if normalized_type in CLASS_TYPE_VALUES:
        return "CLASS"
    if normalized_category in CLASS_CATEGORY_VALUES:
        return "CLASS"
    if normalized_category in ATTRIBUTE_CATEGORY_VALUES:
        return "ATTRIBUTE"
    if normalized_category in FUNCTION_CATEGORY_VALUES:
        return "FUNCTION"
    if normalized_category in PARAMETER_CATEGORY_VALUES:
        return "PARAMETER"
    if normalized_category in DECLARATION_CATEGORY_VALUES:
        return "DECLARATION"
    return None


def load_json_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}

    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Cache file must contain a JSON object: {cache_path}")
    return payload


def save_json_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_path.parent, delete=False) as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name

    Path(temp_name).replace(cache_path)


def default_cache_key(request: IdentifierRequest) -> str:
    return "\t".join(
        [
            request.identifier_name,
            request.context,
            request.type_str,
            request.language,
            request.system_name,
            "" if request.pattern_postprocessing is None else str(bool(request.pattern_postprocessing)),
        ]
    )


def serialize_tagged_identifier(tagged: TaggedIdentifier) -> dict[str, object]:
    return {
        "identifier_name": tagged.identifier_name,
        "context": tagged.context,
        "tokens": list(tagged.tokens),
        "tags": list(tagged.tags),
        "type_str": tagged.type_str,
        "language": tagged.language,
        "system_name": tagged.system_name,
    }


def deserialize_tagged_identifier(payload: dict[str, object]) -> TaggedIdentifier:
    return TaggedIdentifier(
        identifier_name=str(payload.get("identifier_name", "")),
        context=str(payload.get("context", "")),
        tokens=tuple(str(token) for token in payload.get("tokens", [])),
        tags=tuple(str(tag) for tag in payload.get("tags", [])),
        type_str=str(payload.get("type_str", "")),
        language=str(payload.get("language", "")),
        system_name=str(payload.get("system_name", "")),
    )


def tag_requests(
    backend,
    requests: list[IdentifierRequest],
    cache: dict | None = None,
    batch_size: int = 64,
    cache_key_fn: Callable[[IdentifierRequest], str] = default_cache_key,
    request_stats: Counter[str] | None = None,
) -> list[TaggedIdentifier]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    cache = cache if cache is not None else {}
    request_stats = request_stats if request_stats is not None else Counter()
    results: list[TaggedIdentifier | None] = [None] * len(requests)
    missing_groups: dict[str, list[int]] = {}
    unique_requests: list[IdentifierRequest] = []

    for index, request in enumerate(requests):
        cache_key = cache_key_fn(request)
        cached_value = cache.get(cache_key)
        if cached_value is not None:
            request_stats["cache_hit"] += 1
            if isinstance(cached_value, TaggedIdentifier):
                results[index] = cached_value
            else:
                results[index] = deserialize_tagged_identifier(cached_value)
            continue

        if cache_key not in missing_groups:
            missing_groups[cache_key] = []
            unique_requests.append(request)
        missing_groups[cache_key].append(index)

    if unique_requests:
        backend_records = [
            {
                "identifier_name": request.identifier_name,
                "context": request.context,
                "type_str": request.type_str,
                "language": request.language,
                "system_name": request.system_name,
                "pattern_postprocessing": request.pattern_postprocessing,
            }
            for request in unique_requests
        ]
        backend_results = backend.tag_identifier_batch(backend_records, batch_size=batch_size)

        for request, backend_result in zip(unique_requests, backend_results):
            tagged = TaggedIdentifier(
                identifier_name=request.identifier_name,
                context=request.context,
                tokens=tuple(str(token) for token in backend_result["tokens"]),
                tags=tuple(str(tag) for tag in backend_result["tags"]),
                type_str=request.type_str,
                language=request.language,
                system_name=request.system_name,
            )
            cache_key = cache_key_fn(request)
            cache[cache_key] = serialize_tagged_identifier(tagged)
            request_stats["request_made"] += 1
            for index in missing_groups[cache_key]:
                results[index] = tagged

    return [result for result in results if result is not None]


def process_csv_with_backend(
    source_path: Path,
    output_path: Path,
    backend,
    row_parser: Callable[[dict[str, str], str], tuple[IdentifierRequest | None, str | None]],
    row_formatter: Callable[[dict[str, str], TaggedIdentifier], dict[str, str]],
    fieldnames: list[str],
    cache: dict | None = None,
    batch_size: int = 64,
    flush_multiplier: int = 4,
    cache_key_fn: Callable[[IdentifierRequest], str] = default_cache_key,
    request_stats: Counter[str] | None = None,
) -> tuple[int, Counter[str]]:
    cache = cache if cache is not None else {}
    request_stats = request_stats if request_stats is not None else Counter()
    skip_counts: Counter[str] = Counter()
    written_rows = 0
    pending_rows: list[tuple[dict[str, str], IdentifierRequest]] = []
    flush_threshold = max(batch_size, 1) * max(flush_multiplier, 1)
    source_name = source_path.stem

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", newline="") as source_handle, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            request, skip_reason = row_parser(row, source_name)
            if skip_reason is not None:
                skip_counts[skip_reason] += 1
                continue
            if request is None:
                skip_counts["parser_rejected"] += 1
                continue

            pending_rows.append((row, request))
            if len(pending_rows) >= flush_threshold:
                written_rows += _flush_pending_rows(
                    pending_rows,
                    writer,
                    backend,
                    row_formatter,
                    cache,
                    batch_size,
                    cache_key_fn,
                    request_stats,
                )

        written_rows += _flush_pending_rows(
            pending_rows,
            writer,
            backend,
            row_formatter,
            cache,
            batch_size,
            cache_key_fn,
            request_stats,
        )

    return written_rows, skip_counts


def _flush_pending_rows(
    pending_rows: list[tuple[dict[str, str], IdentifierRequest]],
    writer: csv.DictWriter,
    backend,
    row_formatter: Callable[[dict[str, str], TaggedIdentifier], dict[str, str]],
    cache: dict,
    batch_size: int,
    cache_key_fn: Callable[[IdentifierRequest], str],
    request_stats: Counter[str],
) -> int:
    if not pending_rows:
        return 0

    rows = [row for row, _ in pending_rows]
    requests = [request for _, request in pending_rows]
    tagged_rows = tag_requests(
        backend,
        requests,
        cache=cache,
        batch_size=batch_size,
        cache_key_fn=cache_key_fn,
        request_stats=request_stats,
    )

    written_rows = 0
    for row, tagged in zip(rows, tagged_rows):
        writer.writerow(row_formatter(row, tagged))
        written_rows += 1

    pending_rows.clear()
    return written_rows