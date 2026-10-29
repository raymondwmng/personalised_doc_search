"""Unit and integration tests for Stage 2 extended features (Section 4 of PLAN.md)."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

import config
from src.embedder import DocumentEmbedder
from src.indexer import FaissIndexer
from src.memory import UserMemoryStore
from src.parser import LegalDocParser
from src.searcher import SearchEngine
from src.tagger_api import AzureDocIntelligenceClient
from src.tagger_bert import LegalBertTagger


class TestStage2Deliverables(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "test_documents.db"
        self.memory_json_path = self.temp_path / "test_prompt_context.json"
        self.history_db_path = self.temp_path / "test_user_history.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_meso_chunking_provenance_and_granularity(self):
        """Verify Meso chunking (~2,000 tokens) retains document provenance and structures chunks."""
        parser = LegalDocParser(db_path=self.db_path)
        sample_legal_text = (
            "Article 1 Subject Matter. This Regulation establishes rules on high-risk AI systems. " * 50 + "\n\n"
            "Article 2 Scope. This Regulation applies to deployers within the Union. " * 50 + "\n\n"
            "Article 3 Definitions. 'High-risk AI system' means an AI system intended to be used as a safety component. " * 60
        )
        celex = "32024R1689"
        title = "Artificial Intelligence Act"

        chunks = parser.chunk_text(
            text=sample_legal_text,
            celex=celex,
            title=title,
            target_tokens=2000,
            overlap_tokens=200,
            regime="meso",
        )

        self.assertGreater(len(chunks), 0)
        for idx, chunk_text, tok_count in chunks:
            # Verify provenance header is present in every Meso chunk
            self.assertTrue(chunk_text.startswith(f"Document {celex} — {title}"))
            self.assertGreater(tok_count, 50)
            self.assertLessEqual(tok_count, 3500)

    def test_dual_model_spec_resolution(self):
        """Verify config resolution for BAAI/bge-m3 and ModernBERT-base."""
        bge_spec = config.resolve_embedding_spec("bge-m3")
        self.assertEqual(bge_spec["id"], "bge-m3")
        self.assertEqual(bge_spec["dim"], 1024)
        self.assertEqual(bge_spec["name"], "BAAI/bge-m3")

        bert_spec = config.resolve_embedding_spec("modernbert")
        self.assertEqual(bert_spec["id"], "modernbert")
        self.assertEqual(bert_spec["dim"], 768)
        self.assertEqual(bert_spec["name"], "nomic-ai/modernbert-embed-base")

        default_spec = config.resolve_embedding_spec()
        self.assertEqual(default_spec["id"], "bge-m3")

    def test_query_expansion_legal_thesaurus(self):
        """Verify EuroVoc and statutory synonym expansion."""
        search_engine = SearchEngine(
            model_name_or_key="bge-m3",
            db_path=self.db_path,
            memory_store=UserMemoryStore(json_path=self.memory_json_path, history_db_path=self.history_db_path),
        )

        transformed_ai = search_engine.transform_query("AI liability and penalties")
        self.assertIn("artificial intelligence", transformed_ai["lexical_query"])

        transformed_privacy = search_engine.transform_query("privacy breaches")
        self.assertIn("personal data", transformed_privacy["lexical_query"])

        transformed_cars = search_engine.transform_query("passenger cars emissions")
        self.assertIn("co2 emissions", transformed_cars["lexical_query"])

    def test_memory_interaction_accumulation_and_export(self):
        """Verify user interaction accumulation, topic tracking, and JSON/Markdown export."""
        mem_store = UserMemoryStore(json_path=self.memory_json_path, history_db_path=self.history_db_path)

        mock_results = [
            {
                "celex": "32024R1689",
                "tags": {
                    "Regulatory Topics": [{"category": "Digital Services"}],
                    "Industry / Sector": [{"category": "Technology and Digital"}],
                },
            }
        ]

        mem_store.accumulate_interaction(
            query="AI Act compliance",
            retrieved_docs=mock_results,
            summary="Synthesis of AI Act",
            model_used="mistral-small-2603",
        )

        loaded = mem_store.load_memory()
        self.assertEqual(len(loaded["interactions"]), 1)
        self.assertEqual(loaded["interactions"][0]["query"], "AI Act compliance")
        self.assertIn("Digital Services", loaded["interactions"][0]["key_topics"])

        # Test Export
        json_export = mem_store.export_memory_json()
        self.assertIn("AI Act compliance", json_export)

        md_export = mem_store.export_memory_markdown()
        self.assertIn("# Personalised EUR-Lex Legal Search Profile", md_export)
        self.assertIn("AI Act compliance", md_export)

    def test_azure_doc_intelligence_complexity_and_budget(self):
        """Verify Document Intelligence layout complexity classifier and budget guardrails."""
        client = AzureDocIntelligenceClient(budget_pages=10)

        simple_html = "<html><body><p>Standard regulatory recital text.</p></body></html>"
        simple_res = client.assess_complexity(simple_html)
        self.assertFalse(simple_res["is_complex"])
        self.assertEqual(simple_res["selected_model"], "read")

        complex_html = (
            "<html><body>"
            "<table>" + "".join([f"<tr><td>Item {i}</td><td>Value</td></tr>" for i in range(15)]) + "</table>"
            "<table><tr><td>Second table</td></tr></table>"
            "<h3>ANNEX I</h3>"
            "</body></html>"
        )
        complex_res = client.assess_complexity(complex_html)
        self.assertTrue(complex_res["is_complex"])
        self.assertEqual(complex_res["selected_model"], "layout")

        analysis = client.analyze_document(complex_html, celex="32020R0852")
        self.assertGreater(analysis["table_count"], 0)
        self.assertIn("budget_stats", analysis)
        self.assertGreaterEqual(analysis["budget_stats"]["total_pages_processed"], 1)

    def test_faiss_and_bm25_hybrid_indexing(self):
        """Verify FAISS and BM25 indexing and combined search."""
        idx_path = self.temp_path / "test.index"
        meta_path = self.temp_path / "test.jsonl"
        bm25_path = self.temp_path / "test_bm25.pkl"

        indexer = FaissIndexer(
            model_name_or_key="bge-m3",
            index_path=idx_path,
            meta_path=meta_path,
            bm25_path=bm25_path,
            dimension=4,
        )

        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        metadata = [
            {"chunk_id": 1, "doc_id": 101, "text": "artificial intelligence safety systems and deployers"},
            {"chunk_id": 2, "doc_id": 102, "text": "marine fisheries sustainable catch quotas"},
            {"chunk_id": 3, "doc_id": 103, "text": "aviation flight safety bilateral agreements"},
        ]

        indexer.build_index(embeddings, metadata, build_bm25=True)
        indexer.save()

        # Reload
        reloaded = FaissIndexer(
            model_name_or_key="bge-m3",
            index_path=idx_path,
            meta_path=meta_path,
            bm25_path=bm25_path,
            dimension=4,
        )
        self.assertTrue(reloaded.load())

        dense_res = reloaded.search_dense(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)
        self.assertEqual(len(dense_res), 1)
        self.assertEqual(dense_res[0]["chunk_id"], 1)

        bm25_res = reloaded.search_bm25("fisheries quotas", top_k=1)
        self.assertEqual(len(bm25_res), 1)
        self.assertEqual(bm25_res[0]["chunk_id"], 2)


if __name__ == "__main__":
    unittest.main()

