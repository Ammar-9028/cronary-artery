import io, pytest
from PIL import Image
from app import create_app


@pytest.fixture()
def client():
    return create_app(model_path="models/does_not_exist.keras").test_client()


def png(color=(120, 40, 40)):
    buf = io.BytesIO(); Image.new("RGB", (64, 64), color).save(buf, "PNG"); buf.seek(0); return buf


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_index(client):
    assert b"CardioScan" in client.get("/").data


def test_predict_ok(client):
    r = client.post("/predict", data={"image": (png(), "a.png")}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["label"] in ("Positive", "Negative") and 0 <= j["probability"] <= 1


def test_predict_is_deterministic(client):
    a = client.post("/predict", data={"image": (png(), "a.png")}).get_json()
    b = client.post("/predict", data={"image": (png(), "a.png")}).get_json()
    assert a["probability"] == b["probability"]


def test_rejects_missing_and_bad(client):
    assert client.post("/predict").status_code == 400
    assert client.post("/predict", data={"image": (io.BytesIO(b"hello"), "x.png")}).status_code == 400
    assert client.post("/predict", data={"image": (io.BytesIO(b"hello"), "x.txt")}).status_code == 400
