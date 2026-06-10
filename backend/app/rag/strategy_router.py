from app.schemas.rag import RagState, RagStrategyConfig


LOW_CONFIDENCE_RERANK_THRESHOLD = 0.65


class StrategyRouter:
    def select(self, state: RagState) -> RagStrategyConfig:
        intent = state.rag_intent or "unknown"
        strategy = self._base_strategy(intent, state)
        if state.router_confidence < LOW_CONFIDENCE_RERANK_THRESHOLD and not strategy.use_reranker:
            strategy.strategy_name = "low_confidence_hybrid_reranker"
            strategy.use_reranker = True
            strategy.final_top_k = max(strategy.final_top_k, 8)
            strategy.fusion_top_k = max(strategy.fusion_top_k, 60)
            strategy.fallback_policy = "insufficient_evidence"
        return strategy

    def _base_strategy(self, intent: str, state: RagState) -> RagStrategyConfig:
        if intent == "fact_lookup":
            return RagStrategyConfig(
                strategy_name="dense_bm25_rrf",
                retrieval_mode="hybrid",
                top_k_dense=40,
                top_k_bm25=40,
                fusion_top_k=40,
                final_top_k=5,
                use_reranker=False,
                use_query_rewrite=False,
                use_decompose=False,
                fallback_policy="insufficient_evidence",
            )
        if intent == "semantic_query":
            return RagStrategyConfig(
                strategy_name="dense_bm25_rrf_reranker",
                retrieval_mode="hybrid",
                top_k_dense=40,
                top_k_bm25=40,
                fusion_top_k=40,
                final_top_k=5,
                use_reranker=True,
                use_query_rewrite=False,
                use_decompose=False,
                fallback_policy="insufficient_evidence",
            )
        if intent in {"multi_hop", "comparison"}:
            return RagStrategyConfig(
                strategy_name="dense_bm25_rrf_reranker_decompose",
                retrieval_mode="hybrid",
                top_k_dense=80,
                top_k_bm25=80,
                fusion_top_k=80,
                final_top_k=10,
                use_reranker=True,
                use_query_rewrite=False,
                use_decompose=True,
                fallback_policy="insufficient_evidence",
            )
        if intent == "procedure":
            return RagStrategyConfig(
                strategy_name="dense_bm25_rrf",
                retrieval_mode="hybrid",
                top_k_dense=60,
                top_k_bm25=60,
                fusion_top_k=60,
                final_top_k=8,
                use_reranker=False,
                use_query_rewrite=False,
                use_decompose=False,
                fallback_policy="insufficient_evidence",
            )
        if intent == "constrained":
            return RagStrategyConfig(
                strategy_name="dense_bm25_rrf_reranker",
                retrieval_mode="hybrid",
                top_k_dense=60,
                top_k_bm25=60,
                fusion_top_k=60,
                final_top_k=8,
                use_reranker=True,
                use_query_rewrite=False,
                use_decompose=False,
                fallback_policy="insufficient_evidence",
            )
        if intent == "follow_up":
            return RagStrategyConfig(
                strategy_name="history_rewrite_dense_bm25_rrf",
                retrieval_mode="hybrid",
                top_k_dense=40,
                top_k_bm25=40,
                fusion_top_k=40,
                final_top_k=5,
                use_reranker=False,
                use_query_rewrite=True,
                use_decompose=False,
                fallback_policy="insufficient_evidence",
            )
        return RagStrategyConfig(
            strategy_name="conservative_hybrid",
            retrieval_mode="hybrid",
            top_k_dense=40,
            top_k_bm25=40,
            fusion_top_k=40,
            final_top_k=5,
            use_reranker=False,
            use_query_rewrite=False,
            use_decompose=False,
            fallback_policy="insufficient_evidence",
        )


strategy_router = StrategyRouter()
