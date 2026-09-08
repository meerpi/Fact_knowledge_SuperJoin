"""Persistent storage layer for Fact Knowledge Layer using SQLite.

Features:
- Zero external dependencies (uses Python standard library sqlite3)
- WAL (Write-Ahead Logging) mode for concurrent read/write transactions
- Dual hybrid schema: relational index fields + native JSON storage
- Complete serialization/deserialization for Pydantic v2 models:
    DocumentData, ExtractedFacts, ContradictionReport, ClaimGraph
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from app.models import (
    ClaimGraph,
    ContradictionReport,
    DocumentData,
    ExtractedFacts,
    Fact,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get("FACT_LAYER_DB_PATH", "fact_layer.db")


class SQLiteStorage:
    """Thread-safe SQLite storage engine for the fact knowledge layer."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or initialize thread-safe SQLite connection with WAL mode."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode and performance pragmas
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def _init_db(self) -> None:
        """Create database tables and indices if they do not exist."""
        conn = self._get_connection()
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS extracted_facts (
                    doc_id TEXT PRIMARY KEY,
                    model_used TEXT,
                    fallback_attempts INTEGER,
                    facts_count INTEGER,
                    data JSON NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS contradiction_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    cross_doc_id TEXT,
                    data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS claim_graphs (
                    graph_id TEXT PRIMARY KEY,
                    doc_count INTEGER,
                    total_facts INTEGER,
                    clusters_count INTEGER,
                    data JSON NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
                CREATE INDEX IF NOT EXISTS idx_contra_doc ON contradiction_reports(doc_id);
            """)
        logger.info("SQLite storage initialized at: %s (WAL mode enabled)", self.db_path)

    # -----------------------------------------------------------------------
    # Document Operations
    # -----------------------------------------------------------------------

    def save_document(self, doc: DocumentData) -> None:
        """Insert or replace a parsed DocumentData object."""
        conn = self._get_connection()
        doc_json = doc.model_dump_json()
        with conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, page_count, data, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    page_count = excluded.page_count,
                    data = excluded.data,
                    created_at = CURRENT_TIMESTAMP;
                """,
                (doc.doc_id, doc.filename, doc.page_count, doc_json),
            )

    def get_document(self, doc_id: str) -> DocumentData | None:
        """Retrieve a DocumentData by doc_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return DocumentData.model_validate_json(row["data"])

    def list_documents(self) -> list[dict[str, Any]]:
        """List summary metadata for all stored documents."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, filename, page_count, created_at FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [
            {
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "page_count": r["page_count"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in storage."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,))
        return cursor.fetchone() is not None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and cascade associated records."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Extracted Facts Operations
    # -----------------------------------------------------------------------

    def save_extracted_facts(self, doc_id: str, facts: ExtractedFacts) -> None:
        """Insert or replace ExtractedFacts for a document."""
        conn = self._get_connection()
        facts_json = facts.model_dump_json()
        with conn:
            conn.execute(
                """
                INSERT INTO extracted_facts (doc_id, model_used, fallback_attempts, facts_count, data, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(doc_id) DO UPDATE SET
                    model_used = excluded.model_used,
                    fallback_attempts = excluded.fallback_attempts,
                    facts_count = excluded.facts_count,
                    data = excluded.data,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (doc_id, facts.model_used, facts.fallback_attempts, len(facts.facts), facts_json),
            )

    def get_extracted_facts(self, doc_id: str) -> ExtractedFacts | None:
        """Retrieve ExtractedFacts for a document."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM extracted_facts WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return ExtractedFacts.model_validate_json(row["data"])

    def has_extracted_facts(self, doc_id: str) -> bool:
        """Check if extracted facts exist for a document."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM extracted_facts WHERE doc_id = ?", (doc_id,))
        return cursor.fetchone() is not None

    def list_extracted_doc_ids(self) -> list[str]:
        """List all doc_ids that have extracted facts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id FROM extracted_facts ORDER BY updated_at ASC")
        return [r["doc_id"] for r in cursor.fetchall()]

    def get_all_extracted_facts(self) -> dict[str, list[Fact]]:
        """Retrieve all extracted Fact lists grouped by doc_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, data FROM extracted_facts")
        results: dict[str, list[Fact]] = {}
        for row in cursor.fetchall():
            ef = ExtractedFacts.model_validate_json(row["data"])
            results[row["doc_id"]] = ef.facts
        return results

    # -----------------------------------------------------------------------
    # Contradiction Report Operations
    # -----------------------------------------------------------------------

    def save_contradiction_report(self, report: ContradictionReport) -> None:
        """Save a ContradictionReport."""
        conn = self._get_connection()
        report_json = report.model_dump_json()
        with conn:
            conn.execute(
                """
                INSERT INTO contradiction_reports (doc_id, cross_doc_id, data, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (report.doc_id, report.cross_doc_id, report_json),
            )

    def get_contradiction_report(self, doc_id: str) -> ContradictionReport | None:
        """Retrieve the most recent ContradictionReport for a document."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT data FROM contradiction_reports
            WHERE doc_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (doc_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return ContradictionReport.model_validate_json(row["data"])

    # -----------------------------------------------------------------------
    # Claim Graph Operations
    # -----------------------------------------------------------------------

    def save_claim_graph(self, graph: ClaimGraph, graph_id: str = "latest") -> None:
        """Insert or replace a ClaimGraph."""
        conn = self._get_connection()
        graph_json = graph.model_dump_json()
        doc_count = len(graph.documents)
        total_facts = graph.total_facts
        clusters_count = len(graph.clusters)

        with conn:
            conn.execute(
                """
                INSERT INTO claim_graphs (graph_id, doc_count, total_facts, clusters_count, data, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(graph_id) DO UPDATE SET
                    doc_count = excluded.doc_count,
                    total_facts = excluded.total_facts,
                    clusters_count = excluded.clusters_count,
                    data = excluded.data,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (graph_id, doc_count, total_facts, clusters_count, graph_json),
            )

    def get_claim_graph(self, graph_id: str = "latest") -> ClaimGraph | None:
        """Retrieve a ClaimGraph by graph_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM claim_graphs WHERE graph_id = ?", (graph_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return ClaimGraph.model_validate_json(row["data"])

    def has_claim_graph(self, graph_id: str = "latest") -> bool:
        """Check if a claim graph exists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM claim_graphs WHERE graph_id = ?", (graph_id,))
        return cursor.fetchone() is not None

    # -----------------------------------------------------------------------
    # Maintenance & Cleanup
    # -----------------------------------------------------------------------

    def clear_all(self) -> None:
        """Clear all stored data (useful for test fixtures)."""
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM claim_graphs;")
            conn.execute("DELETE FROM contradiction_reports;")
            conn.execute("DELETE FROM extracted_facts;")
            conn.execute("DELETE FROM documents;")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
