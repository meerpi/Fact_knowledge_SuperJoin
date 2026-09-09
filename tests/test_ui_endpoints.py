import os
import tempfile
import pytest
from starlette.testclient import TestClient
import app.server as server_module
from app.server import app
from app.storage import SQLiteStorage


@pytest.fixture(autouse=True)
def isolated_storage():
    orig_storage = server_module.storage
    temp_fd, temp_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_fd)
    server_module.storage = SQLiteStorage(temp_path)
    yield
    server_module.storage = orig_storage
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def client():
    return TestClient(app)


def test_capabilities_endpoint(client):
    response = client.get("/system/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "gemini_api_key_configured" in data
    assert "voyage_api_key_configured" in data
    assert "docling_available" in data
    assert "nli_model_available" in data
    assert "loaded_documents" in data
    assert "extracted_documents" in data
    assert "has_reconciliation" in data
    assert isinstance(data["loaded_documents"], int)


def test_seed_demo_and_cases(client):
    response = client.post("/system/seed-demo")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "seeded"
    assert data["total_facts"] > 0
    assert data["clusters_count"] > 0

    # Verify /reconcile/cases returns the 4 graded cases
    cases_resp = client.get("/reconcile/cases")
    assert cases_resp.status_code == 200
    cases = cases_resp.json()
    assert "case_1_corroborated" in cases
    assert "case_2_contradicted" in cases
    assert "case_3_reconciled" in cases
    assert "case_4_extraction_failure" in cases

    # Check case 1
    c1 = cases["case_1_corroborated"]
    assert c1 is not None
    assert "evidence" in c1
    assert "explanation" in c1
    assert "dispute_code" in c1

    # Check case 2
    c2 = cases["case_2_contradicted"]
    assert c2 is not None
    assert "evidence" in c2
    assert "consensus_value" in c2
    assert "dispute_code" in c2

    # Check case 3
    c3 = cases["case_3_reconciled"]
    assert c3 is not None
    assert "reconciliation_dimension" in c3
    assert "dispute_code" in c3

    # Check case 4
    c4 = cases["case_4_extraction_failure"]
    assert c4 is not None
    assert "failure_type" in c4
    assert "description" in c4
    assert "mitigation" in c4

    # Verify /reconcile/latest returns the full ClaimGraph
    graph_resp = client.get("/reconcile/latest")
    assert graph_resp.status_code == 200
    graph = graph_resp.json()
    assert "clusters" in graph
    assert len(graph["clusters"]) > 0
    assert "extraction_failures" in graph


def test_document_file_and_word_bboxes(client):
    # Ensure starter doc is seeded
    seed_res = client.post("/system/seed-demo")
    assert seed_res.status_code == 200
    docs_map = seed_res.json().get("documents", {})
    doc_id = list(docs_map.keys())[0] if docs_map else "0bcab8c03589"

    file_resp = client.get(f"/documents/{doc_id}/file")
    assert file_resp.status_code == 200
    assert file_resp.headers["content-type"] == "application/pdf"

    # Nonexistent doc
    missing_file = client.get("/documents/nonexistent_doc_id/file")
    assert missing_file.status_code == 404

    # Word bboxes for page 0
    bbox_resp = client.get(f"/documents/{doc_id}/page/0/word-bboxes")
    assert bbox_resp.status_code == 200
    bbox_data = bbox_resp.json()
    assert bbox_data["doc_id"] == doc_id
    assert bbox_data["page_num"] == 0
    assert isinstance(bbox_data["words"], list)
    assert len(bbox_data["words"]) > 0
    first_word = bbox_data["words"][0]
    assert "word" in first_word
    assert "bbox" in first_word
    assert "x0" in first_word["bbox"]
    assert "y0" in first_word["bbox"]

    # Out of range page
    oor_resp = client.get(f"/documents/{doc_id}/page/9999/word-bboxes")
    assert oor_resp.status_code == 400


def test_pipeline_status_lifecycle(client):
    seed_res = client.post("/system/seed-demo")
    assert seed_res.status_code == 200
    docs_map = seed_res.json().get("documents", {})
    doc_id = list(docs_map.keys())[0] if docs_map else "0bcab8c03589"

    start_resp = client.post("/pipeline/start", json={"doc_ids": [doc_id]})
    assert start_resp.status_code == 200
    data = start_resp.json()
    assert "job_id" in data
    job_id = data["job_id"]
    assert data["status"] == "running"

    status_resp = client.get(f"/pipeline/status/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert "current_step" in status_data
    assert "facts_extracted_so_far" in status_data
    assert "stages_completed" in status_data

    # Nonexistent job
    missing_job = client.get("/pipeline/status/invalid_job_999")
    assert missing_job.status_code == 404
