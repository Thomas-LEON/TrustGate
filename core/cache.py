"""
TrustGate — In-Memory Deterministic & Semantic Cache.

Features:
- Dual-tier caching:
  1. Tier 1: Exact Normalized Hash (O(1) lookup, < 0.2ms latency).
  2. Tier 2: Semantic Similarity Matcher (RapidFuzz token-set & token-sort ratio, < 5ms).
- Zero external dependencies: works out-of-the-box on Windows/Linux without Redis.
- Tracks FinOps metrics: exact tokens saved, USD saved, hit count.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional, Any
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from core.config import settings


class CacheEntry(BaseModel):
    """Cached response entry with FinOps savings metadata."""
    key: str
    prompt: str
    response_content: str
    model_origin: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    created_at: float
    last_accessed_at: float
    hits: int = 0


class SemanticCache:
    """High-performance in-memory semantic cache with LRU eviction."""

    def __init__(
        self,
        similarity_threshold: float = settings.cache_similarity_threshold,
        max_entries: int = settings.cache_max_entries,
        ttl_seconds: int = settings.cache_ttl_seconds,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        # Storage: key -> CacheEntry
        self._store: dict[str, CacheEntry] = {}

        # Global FinOps counters
        self.total_queries: int = 0
        self.total_hits: int = 0
        self.total_tokens_saved: int = 0
        self.total_cost_saved_usd: float = 0.0

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Strip whitespace, lowercase, and remove redundant whitespace."""
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _hash_prompt(normalized: str) -> str:
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(
        self,
        prompt: str,
        threshold: Optional[float] = None,
    ) -> Optional[tuple[CacheEntry, float, str]]:
        """
        Lookup prompt in cache.
        Returns: (CacheEntry, similarity_score, match_type: 'EXACT' | 'SEMANTIC') or None.
        """
        self.total_queries += 1
        now = time.time()
        norm_prompt = self._normalize_text(prompt)
        exact_key = self._hash_prompt(norm_prompt)

        # 1. Level 1: Exact Hash Match (O(1))
        if exact_key in self._store:
            entry = self._store[exact_key]
            # Check TTL
            if now - entry.created_at <= self.ttl_seconds:
                entry.hits += 1
                entry.last_accessed_at = now
                self.total_hits += 1
                self.total_tokens_saved += entry.total_tokens
                self.total_cost_saved_usd += entry.estimated_cost_usd
                return entry, 1.0, "EXACT"

        # 2. Level 2: Semantic Similarity Match
        thresh = threshold if threshold is not None else self.similarity_threshold
        best_entry: Optional[CacheEntry] = None
        best_score: float = 0.0

        for key, entry in list(self._store.items()):
            # Check TTL expiration
            if now - entry.created_at > self.ttl_seconds:
                del self._store[key]
                continue

            entry_norm = self._normalize_text(entry.prompt)

            # RapidFuzz token_set_ratio & token_sort_ratio
            score_set = fuzz.token_set_ratio(norm_prompt, entry_norm) / 100.0
            score_sort = fuzz.token_sort_ratio(norm_prompt, entry_norm) / 100.0
            # Weighted average favoring token_set
            combined_score = round(max(score_set, score_sort), 3)

            if combined_score > best_score:
                best_score = combined_score
                best_entry = entry

        if best_entry and best_score >= thresh:
            best_entry.hits += 1
            best_entry.last_accessed_at = now
            self.total_hits += 1
            self.total_tokens_saved += best_entry.total_tokens
            self.total_cost_saved_usd += best_entry.estimated_cost_usd
            return best_entry, best_score, "SEMANTIC"

        return None

    def set(
        self,
        prompt: str,
        response_content: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> CacheEntry:
        """Store a successful LLM response in cache."""
        now = time.time()
        norm_prompt = self._normalize_text(prompt)
        key = self._hash_prompt(norm_prompt)

        # LRU Eviction if capacity reached
        if len(self._store) >= self.max_entries:
            # Remove least recently accessed
            oldest_key = min(self._store.keys(), key=lambda k: self._store[k].last_accessed_at)
            del self._store[oldest_key]

        entry = CacheEntry(
            key=key,
            prompt=prompt,
            response_content=response_content,
            model_origin=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            estimated_cost_usd=round(cost_usd, 6),
            created_at=now,
            last_accessed_at=now,
            hits=0,
        )
        self._store[key] = entry
        return entry

    def stats(self) -> dict[str, Any]:
        """Return cache analytics and FinOps savings."""
        hit_rate = (
            round((self.total_hits / self.total_queries) * 100, 2)
            if self.total_queries > 0
            else 0.0
        )
        return {
            "entries_count": len(self._store),
            "max_entries": self.max_entries,
            "total_queries": self.total_queries,
            "total_hits": self.total_hits,
            "total_misses": self.total_queries - self.total_hits,
            "hit_rate_pct": hit_rate,
            "total_tokens_saved": self.total_tokens_saved,
            "total_cost_saved_usd": round(self.total_cost_saved_usd, 6),
        }

    def clear(self) -> None:
        """Clear all cache entries and counters."""
        self._store.clear()
        self.total_queries = 0
        self.total_hits = 0
        self.total_tokens_saved = 0
        self.total_cost_saved_usd = 0.0


# Global singleton instance
semantic_cache = SemanticCache()
