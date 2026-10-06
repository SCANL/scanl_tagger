import pytest

from scalar_tagger import contract
from scalar_tagger.tagging_backend import TaggingBackend
from tests.conftest import FakeTagger


@pytest.fixture
def backend():
    return TaggingBackend("fake/model", tagger=FakeTagger(), english_words={"get", "user", "token"})


def request(*identifiers, **extra):
    return {"id": 7, "identifiers": list(identifiers), **extra}


def ident(name, context="FUNCTION", **extra):
    return {"key": name, "name": name, "context": context, **extra}


@pytest.mark.parametrize(
    "name, tokens, expected",
    [
        ("getUserToken", ["get", "User", "Token"], [(0, 3), (3, 7), (7, 12)]),
        ("m_userName", ["m", "user", "Name"], [(0, 1), (2, 6), (6, 10)]),
        ("__init__", ["init"], [(2, 6)]),
        ("host1", ["host", "1"], [(0, 4), (4, 5)]),
        ("XMLHttpRequest", ["xml", "http", "request"], [(0, 3), (3, 7), (7, 14)]),
        ("aaa_aaa", ["aaa", "aaa"], [(0, 3), (4, 7)]),
        ("getFoo", ["get", "Bar", "Foo"], [(0, 3), (None, None), (3, 6)]),
        ("naïveCount", ["naïve", "Count"], [(0, 5), (5, 10)]),
    ],
)
def test_token_offsets(name, tokens, expected):
    offsets = contract.token_offsets(name, tokens)
    assert offsets == expected
    for token, (start, end) in zip(tokens, offsets):
        if start is not None:
            assert name[start:end].lower() == token.lower()


def test_response_shape(backend):
    response = backend.tag_batch(request(ident("getUserToken", type="string", language="C++", system="proj")))
    assert response["id"] == 7
    assert response["model"] == {
        "name": "fake/model",
        "revision": None,
        "features": ["context"],
        "postprocess": False,
        "device": "cpu",
        "scalar_version": response["model"]["scalar_version"],
    }
    assert response["results"] == [{
        "key": "getUserToken",
        "tokens": [
            {"text": "get", "start": 0, "end": 3, "tag": "N", "dictionary": True},
            {"text": "User", "start": 3, "end": 7, "tag": "N", "dictionary": True},
            {"text": "Token", "start": 7, "end": 12, "tag": "N", "dictionary": True},
        ],
    }]


def test_caller_supplied_tokens_skip_splitting(backend):
    response = backend.tag_batch(request(ident("IPv4Address", tokens=["IPv4", "Address"])))
    tokens = response["results"][0]["tokens"]
    assert [(t["text"], t["start"], t["end"]) for t in tokens] == [("IPv4", 0, 4), ("Address", 4, 11)]


def test_context_is_case_insensitive(backend):
    response = backend.tag_batch(request(ident("count", context="declaration")))
    assert "tokens" in response["results"][0]
    assert backend.lm_model.calls[0][0]["context"] == "DECLARATION"


@pytest.mark.parametrize(
    "entry, code",
    [
        (ident(""), contract.EMPTY_IDENTIFIER),
        (ident("   "), contract.EMPTY_IDENTIFIER),
        (ident("operator=="), contract.UNSUPPORTED_IDENTIFIER),
        (ident("operator()"), contract.UNSUPPORTED_IDENTIFIER),
        (ident("~Foo"), contract.UNSUPPORTED_IDENTIFIER),
        (ident("ns::name"), contract.UNSUPPORTED_IDENTIFIER),
        (ident("self.name"), contract.UNSUPPORTED_IDENTIFIER),
        (ident("___"), contract.NO_TOKENS),
        (ident("foo", context="LAMBDA"), contract.INVALID_CONTEXT),
        (ident("foo", context=None), contract.INVALID_CONTEXT),
        ({"key": "k", "name": 5, "context": "FUNCTION"}, contract.INVALID_IDENTIFIER),
        (ident("foo", type=3), contract.INVALID_IDENTIFIER),
        (ident("foo", tokens=[]), contract.INVALID_TOKENS),
        (ident("foo", tokens=["fo", ""]), contract.INVALID_TOKENS),
        (ident("foo", tokens="foo"), contract.INVALID_TOKENS),
        ("not an object", contract.INVALID_IDENTIFIER),
    ],
)
def test_bad_identifier_gets_its_own_error(backend, entry, code):
    response = backend.tag_batch(request(ident("getUserToken"), entry, ident("userToken")))
    good_before, bad, good_after = response["results"]
    assert "tokens" in good_before and "tokens" in good_after
    assert bad["error"]["code"] == code
    assert bad["error"]["message"]
    assert bad["key"] == (entry.get("key") if isinstance(entry, dict) else None)


@pytest.mark.parametrize("name", ["operatorCount", "operator_name", "tildeFoo", "count1", "用户名", "naïveCount"])
def test_names_that_look_unusual_are_still_tagged(backend, name):
    response = backend.tag_batch(request(ident(name)))
    assert "tokens" in response["results"][0], response


def test_too_long_identifier():
    backend = TaggingBackend("fake/model", tagger=FakeTagger(max_words=3), english_words=set())
    response = backend.tag_batch(request(ident("a_b_c_d"), ident("a_b")))
    assert response["results"][0]["error"]["code"] == contract.IDENTIFIER_TOO_LONG
    assert "tokens" in response["results"][1]


def test_model_failure_is_isolated_to_one_identifier():
    backend = TaggingBackend("fake/model", tagger=FakeTagger(fail_on="boom"), english_words=set())
    response = backend.tag_batch(request(ident("getUser"), ident("boom_now"), ident("setUser")))
    assert "tokens" in response["results"][0]
    assert response["results"][1]["error"]["code"] == contract.INTERNAL_ERROR
    assert "tokens" in response["results"][2]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {"id": 1},
        {"id": 1, "identifiers": "x"},
        {"id": 1, "identifiers": [], "options": "x"},
        {"id": 1, "identifiers": [], "options": {"postprocess": "yes"}},
        {"id": 1, "identifiers": [], "options": {"confidence": 1}},
    ],
)
def test_malformed_request(backend, raw):
    response = backend.tag_batch(raw)
    assert response["error"]["code"] == contract.INVALID_REQUEST
    assert "results" not in response
    assert response["id"] == (raw.get("id") if isinstance(raw, dict) else None)


def test_empty_batch(backend):
    assert backend.tag_batch(request())["results"] == []


def test_postprocess_option_is_passed_through_and_reported(backend):
    response = backend.tag_batch(request(ident("getUser"), options={"postprocess": True}))
    assert response["model"]["postprocess"] is True
    assert backend.lm_model.calls[0][0]["pattern_postprocessing"] is True


def test_confidence_request_warns_until_supported(backend):
    response = backend.tag_batch(request(ident("getUser"), options={"confidence": True}))
    assert response["warnings"][0]["code"] == "CONFIDENCE_UNAVAILABLE"
    assert "p" not in response["results"][0]["tokens"][0]
