"""Analytics engine for aggregating search results and preparing radar & histogram data."""

from collections import Counter, defaultdict
from typing import Any, Dict, List
import pandas as pd

import config


class SearchAnalytics:
    """Aggregates and formats statistical metrics across search results for visual charts."""

    def __init__(self, taxonomy: Dict[str, List[str]] = config.TAXONOMY):
        self.taxonomy = taxonomy

    def aggregate_tags(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Counts frequency of categories per dimension across retrieved documents."""
        counts = {dim: Counter() for dim in self.taxonomy}

        for doc in results:
            doc_tags = doc.get("tags", {})
            for dim in self.taxonomy:
                cat_list = doc_tags.get(dim, [])
                for item in cat_list:
                    cat_name = item.get("category")
                    if cat_name:
                        counts[dim][cat_name] += 1

        return {dim: dict(counter) for dim, counter in counts.items()}

    def prepare_histogram_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Constructs a DataFrame suitable for Plotly bar chart histograms."""
        tag_counts = self.aggregate_tags(results)
        rows = []
        for dim, cats in tag_counts.items():
            for cat, count in cats.items():
                rows.append({"Dimension": dim, "Category": cat, "Count": count})

        if not rows:
            return pd.DataFrame(columns=["Dimension", "Category", "Count"])
        return pd.DataFrame(rows).sort_values(by=["Dimension", "Count"], ascending=[True, False])

    def prepare_radar_data(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Prepares radar chart vertices and normalized values per dimension.
        
        Returns:
            Dict mapping dimension -> {'categories': [...], 'values': [...]}
        """
        tag_counts = self.aggregate_tags(results)
        total_docs = max(1, len(results))
        radar_data = {}

        for dim, all_cats in self.taxonomy.items():
            cat_counts = tag_counts.get(dim, {})
            # Only include top 8 categories if dimension has many items (for readability)
            sorted_cats = sorted(all_cats, key=lambda c: cat_counts.get(c, 0), reverse=True)
            active_cats = sorted_cats[:8] if len(sorted_cats) > 8 else sorted_cats

            values = [cat_counts.get(c, 0) for c in active_cats]
            # Normalise to 0-1 scale or raw count
            radar_data[dim] = {
                "categories": active_cats,
                "values": values,
                "normalized": [round(v / total_docs, 3) for v in values],
            }

        return radar_data
