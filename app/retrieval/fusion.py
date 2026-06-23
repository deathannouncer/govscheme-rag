def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """
    Standard RRF: score(doc) = sum over lists of 1 / (k + rank_in_list).
    `ranked_lists` is a list of (doc_id, score) lists, each already sorted
    best-first.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
