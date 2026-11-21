import io
import json

import pytest

from scalar_tagger import stdio_server
from scalar_tagger.tagging_backend import TaggingBackend
from tests.conftest import FakeTagger


@pytest.fixture
def backend():
    return TaggingBackend("fake/model", tagger=FakeTagger(), english_words={"get", "user"})


def tag_request(request_id=1, name="getUser"):
    return {"id": request_id, "identifiers": [{"key": "k", "name": name, "context": "FUNCTION"}]}


def run_stdio(backend, *lines):
    stdin = io.BytesIO(b"".join(lines))
    stdout = io.BytesIO()
    status = stdio_server.serve_stdio(lambda: backend, stdin=stdin, stdout=stdout)
    messages = [json.loads(line) for line in stdout.getvalue().decode("utf-8").splitlines()]
    return status, messages


def line(message) -> bytes:
    return json.dumps(message).encode("utf-8") + b"\n"


# --- stdio ---------------------------------------------------------------------------

def test_stdio_ready_line_then_one_response_per_request(backend):
    status, messages = run_stdio(backend, line(tag_request(1)), b"\n", line(tag_request(2, "setUser")))
    assert status == 0
    ready, first, second = messages
    assert ready == {"ready": True, "model": backend.model_info()}
    assert first["id"] == 1 and first["results"][0]["tokens"][0]["text"] == "get"
    assert second["id"] == 2 and second["results"][0]["tokens"][0]["text"] == "set"


def test_stdio_exits_cleanly_when_stdin_is_empty(backend):
    status, messages = run_stdio(backend)
    assert status == 0
    assert messages == [{"ready": True, "model": backend.model_info()}]


def test_stdio_info_command(backend):
    _, messages = run_stdio(backend, line({"id": "x", "command": "info"}))
    assert messages[1] == {"id": "x", "model": backend.model_info()}


@pytest.mark.parametrize(
    "raw, code",
    [
        (b"{not json\n", "INVALID_JSON"),
        (b"\xff\xfe\n", "INVALID_JSON"),
        (line({"id": 3, "command": "shutdown"}), "UNKNOWN_COMMAND"),
        (line({"id": 4}), "INVALID_REQUEST"),
        (line([1, 2]), "INVALID_REQUEST"),
    ],
)
def test_stdio_bad_message_gets_an_error_and_the_session_continues(backend, raw, code):
    _, messages = run_stdio(backend, raw, line(tag_request(9)))
    assert messages[1]["error"]["code"] == code
    assert messages[2]["id"] == 9 and "results" in messages[2]


def test_stdio_non_ascii_round_trip(backend):
    _, messages = run_stdio(backend, line(tag_request(1, "naïveCount")))
    assert messages[1]["results"][0]["tokens"][0]["text"] == "naïve"


def test_stdio_model_load_failure():
    def fail():
        raise OSError("no such model")

    stdout = io.BytesIO()
    status = stdio_server.serve_stdio(fail, stdin=io.BytesIO(), stdout=stdout)
    assert status == 1
    assert json.loads(stdout.getvalue()) == {
        "ready": False,
        "error": {"code": "MODEL_LOAD_FAILED", "message": "no such model"},
    }


# --- HTTP ----------------------------------------------------------------------------

@pytest.fixture
def client(backend):
    from scalar_tagger import tag_identifier

    tag_identifier.app.backend = backend
    tag_identifier.app.words = tag_identifier.WordList("")
    return tag_identifier.app.test_client()


def test_http_post_tag(client, backend):
    response = client.post("/tag", json=tag_request(5))
    assert response.status_code == 200
    body = response.get_json()
    assert list(body) == ["id", "model", "results"]
    assert body == backend.tag_batch(tag_request(5))


def test_http_post_tag_with_type_that_would_break_a_url(client):
    request = tag_request(6)
    request["identifiers"][0]["type"] = "std::map<std::string, int>"
    assert client.post("/tag", json=request).status_code == 200


def test_http_post_tag_without_content_type(client):
    response = client.post("/tag", data=json.dumps(tag_request(7)))
    assert response.status_code == 200


def test_http_post_tag_bad_json(client):
    response = client.post("/tag", data="{nope", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_JSON"


def test_http_post_tag_invalid_request(client):
    response = client.post("/tag", json={"id": 8})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_REQUEST"


def test_http_info(client, backend):
    response = client.get("/info")
    assert response.status_code == 200
    assert response.get_json() == {"id": None, "model": backend.model_info()}


def test_http_legacy_get_route_still_works(client):
    response = client.get("/getUser/FUNCTION")
    assert response.status_code == 200
    assert response.get_json() == {
        "words": [{"get": {"tag": "N", "dictionary": "DW"}}, {"User": {"tag": "N", "dictionary": "DW"}}]
    }
