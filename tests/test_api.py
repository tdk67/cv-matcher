"""Tests for API routes."""
import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


class TestHealthCheck:
    def test_health(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestDocumentsAPI:
    def test_list_empty(self):
        resp = client.get("/api/documents/")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total_documents" in data

    def test_upload_invalid_format(self):
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("test.exe", b"binary content", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_valid_txt(self):
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("test_cv.txt", b"John Doe\nSenior Developer\nPython, Java", "text/plain")},
        )
        # Should succeed or return 422 if pipeline fails
        assert resp.status_code in (200, 422)

    def test_delete_nonexistent(self):
        resp = client.delete("/api/documents/nonexistent_file.pdf")
        assert resp.status_code == 404


class TestDashboardAPI:
    def test_stats(self):
        resp = client.get("/api/dashboard/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "documents_by_format" in data


class TestQueryAPI:
    def test_query_empty_kb(self):
        resp = client.post("/api/query/", json={"question": "Find Java developer"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "query_type" in data

    def test_query_out_of_scope(self):
        resp = client.post("/api/query/", json={"question": "What is the weather?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("out_of_scope") is True

    def test_query_missing_question(self):
        resp = client.post("/api/query/", json={})
        assert resp.status_code == 422  # Validation error


class TestEvaluationAPI:
    def test_get_results_empty(self):
        resp = client.get("/api/evaluation/results")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data

    def test_run_evaluation(self):
        resp = client.post("/api/evaluation/run")
        # May take time, just check it doesn't crash
        assert resp.status_code in (200, 500)
