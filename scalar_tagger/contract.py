"""
The JSON request/response contract shared by every transport (stdio, HTTP).

This module validates requests, splits identifiers, and computes token offsets. It has no
model dependencies, so it can be tested without loading the tagger.

Request:
    {"id": 1,
     "options": {"postprocess": true, "confidence": false},
     "identifiers": [{"key": "a1", "name": "getUserToken", "context": "FUNCTION",
                      "type": "string", "language": "C++", "system": "myproj",
                      "tokens": null}]}

Response:
    {"id": 1,
     "model": {...},
     "results": [{"key": "a1", "tokens": [{"text": "get", "start": 0, "end": 3,
                                           "tag": "V", "dictionary": true}, ...]},
                 {"key": "a2", "error": {"code": "...", "message": "..."}}]}

A request that can't be read at all gets {"id": ..., "model": {...}, "error": {...}} instead
of "results".
"""

import re
from dataclasses import dataclass, field

from spiral import ronin

CONTEXTS = ("ATTRIBUTE", "CLASS", "DECLARATION", "FUNCTION", "PARAMETER")

# Request-level error codes
INVALID_JSON = "INVALID_JSON"
INVALID_REQUEST = "INVALID_REQUEST"
UNKNOWN_COMMAND = "UNKNOWN_COMMAND"

# Identifier-level error codes
INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
INVALID_CONTEXT = "INVALID_CONTEXT"
INVALID_TOKENS = "INVALID_TOKENS"
EMPTY_IDENTIFIER = "EMPTY_IDENTIFIER"
NO_TOKENS = "NO_TOKENS"
UNSUPPORTED_IDENTIFIER = "UNSUPPORTED_IDENTIFIER"
IDENTIFIER_TOO_LONG = "IDENTIFIER_TOO_LONG"
INTERNAL_ERROR = "INTERNAL_ERROR"

# `operator==`, `operator()`, `operator new`, but not `operatorCount`
_OPERATOR_NAME = re.compile(r"^operator(?![A-Za-z0-9_])")
_QUALIFIED_NAME = re.compile(r"::|\.|->")


class ContractError(ValueError):
    """An error that applies to a whole request or a single identifier."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class Options:
    postprocess: bool | None = None
    confidence: bool = False


@dataclass
class IdentifierInput:
    """A validated identifier, ready to tag."""
    name: str
    context: str
    tokens: list[str]
    type_str: str = ""
    language: str = ""
    system: str = ""


@dataclass
class ParsedRequest:
    id: object
    options: Options
    # One (key, item) pair per identifier, in request order. The item is an IdentifierInput,
    # or the ContractError that kept it from being tagged.
    items: list = field(default_factory=list)


def _optional_string(entry: dict, field_name: str) -> str:
    value = entry.get(field_name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ContractError(INVALID_IDENTIFIER, f"'{field_name}' must be a string")
    return value


def parse_options(raw) -> Options:
    if raw is None:
        return Options()
    if not isinstance(raw, dict):
        raise ContractError(INVALID_REQUEST, "'options' must be an object")

    postprocess = raw.get("postprocess")
    if postprocess is not None and not isinstance(postprocess, bool):
        raise ContractError(INVALID_REQUEST, "'options.postprocess' must be true, false, or null")
    confidence = raw.get("confidence", False)
    if not isinstance(confidence, bool):
        raise ContractError(INVALID_REQUEST, "'options.confidence' must be true or false")
    return Options(postprocess=postprocess, confidence=confidence)


def split_identifier(name: str) -> list[str]:
    """Split an identifier into words, rejecting names the tagger isn't trained on."""
    if not name.strip():
        raise ContractError(EMPTY_IDENTIFIER, "identifier name is empty")
    if _OPERATOR_NAME.match(name):
        raise ContractError(UNSUPPORTED_IDENTIFIER, "operator names are not tagged")
    if name.startswith("~"):
        raise ContractError(UNSUPPORTED_IDENTIFIER, "destructor names are not tagged")
    if _QUALIFIED_NAME.search(name):
        raise ContractError(
            UNSUPPORTED_IDENTIFIER,
            "qualified names are not tagged; send the unqualified name",
        )

    tokens = ronin.split(name)
    if not tokens:
        raise ContractError(NO_TOKENS, "identifier has no words to tag")
    return tokens


def parse_identifier(entry) -> IdentifierInput:
    if not isinstance(entry, dict):
        raise ContractError(INVALID_IDENTIFIER, "each identifier must be an object")

    name = entry.get("name")
    if not isinstance(name, str):
        raise ContractError(INVALID_IDENTIFIER, "'name' must be a string")

    context = entry.get("context")
    if not isinstance(context, str) or context.strip().upper() not in CONTEXTS:
        raise ContractError(
            INVALID_CONTEXT,
            f"'context' must be one of {', '.join(CONTEXTS)}",
        )

    raw_tokens = entry.get("tokens")
    if raw_tokens is None:
        tokens = split_identifier(name)
    else:
        if (
            not isinstance(raw_tokens, list)
            or not raw_tokens
            or not all(isinstance(t, str) and t.strip() for t in raw_tokens)
        ):
            raise ContractError(INVALID_TOKENS, "'tokens' must be null or a non-empty list of non-empty strings")
        tokens = list(raw_tokens)

    return IdentifierInput(
        name=name,
        context=context.strip().upper(),
        tokens=tokens,
        type_str=_optional_string(entry, "type"),
        language=_optional_string(entry, "language"),
        system=_optional_string(entry, "system"),
    )


def parse_request(raw) -> ParsedRequest:
    """
    Validate a tagging request.

    Raises:
        ContractError: if the request as a whole is malformed. Problems with a single
        identifier are returned in `items` instead, so the rest of the batch still runs.
    """
    if not isinstance(raw, dict):
        raise ContractError(INVALID_REQUEST, "request must be a JSON object")

    identifiers = raw.get("identifiers")
    if not isinstance(identifiers, list):
        raise ContractError(INVALID_REQUEST, "'identifiers' must be a list")

    parsed = ParsedRequest(id=raw.get("id"), options=parse_options(raw.get("options")))
    for entry in identifiers:
        key = entry.get("key") if isinstance(entry, dict) else None
        try:
            parsed.items.append((key, parse_identifier(entry)))
        except ContractError as exc:
            parsed.items.append((key, exc))
    return parsed


def token_offsets(name: str, tokens: list[str]) -> list[tuple[int | None, int | None]]:
    """
    Locate each token in `name`, in order and ignoring case.

    Returns (start, end) pairs such that name[start:end] matches the token. A token that
    can't be found (for example, a caller-supplied token that isn't in the name) gets
    (None, None), and the search for the next token continues from the last match.
    """
    offsets = []
    cursor = 0
    for token in tokens:
        match = re.compile(re.escape(token), re.IGNORECASE).search(name, cursor)
        if match is None:
            offsets.append((None, None))
            continue
        offsets.append((match.start(), match.end()))
        cursor = match.end()
    return offsets

