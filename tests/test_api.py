from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)
RICE = {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82, "ph": 6.5, "rainfall": 203}


def test_health():
    assert c.get("/health").json()["ok"] is True


def test_predict_rice_row():
    r = c.post("/predict", json=RICE).json()
    assert r["recommended_crop"] == "rice"
    assert len(r["top_crops"]) == 3
    assert r["assessment"]["crop"] == "rice"


def test_target_crop_assessment_flags_issues():
    r = c.post("/predict", json={**RICE, "target_crop": "chickpea"}).json()
    assert r["assessment"]["crop"] == "chickpea"
    assert r["assessment"]["out_of_range"] > 0


def test_validation_rejects_bad_input():
    assert c.post("/predict", json={**RICE, "ph": 20}).status_code == 422
    assert c.post("/predict", json={"N": 1}).status_code == 422
    assert c.post("/predict", json={**RICE, "target_crop": "banana-split"}).status_code == 422


def test_metrics_exposed():
    m = c.get("/metrics").json()
    assert m["models"]["random_forest"]["test_accuracy"] > 0.9
