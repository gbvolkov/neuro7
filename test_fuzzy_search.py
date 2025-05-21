# pip install sentence-transformers scikit-learn   (≈ 1-2 min)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# 🔌 Load once, reuse everywhere (CPU-friendly; GPU if available)
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_search(
    query: str,
    text: str,
    *,
    splitter=str.split,
    span: int = 20,
    top_n: int = 5
) -> List[Tuple[str, float, int]]:
    """
    Semantic search for *query* inside *text* using embeddings.

    Parameters
    ----------
    query   : str   – What you’re looking for.
    text    : str   – Corpus (a few KB to dozens of KB is fine).
    splitter: call  – Function that turns *text* into tokens. Default: words.
                       Swap in a sentence splitter for longer docs.
    span    : int   – How many tokens to group per window. Tune per use-case.
    top_n   : int   – How many hits to return.

    Returns
    -------
    matches : list[(snippet, score, char_index)]
              score ∈ [0,1] (higher = more similar)
    """
    # 1️⃣ Chunk the corpus
    tokens = splitter(text)
    windows, char_offsets = [], []
    for i in range(0, len(tokens), span):
        snippet = " ".join(tokens[i : i + span])
        windows.append(snippet)
        # record first character position of this snippet for provenance
        char_offsets.append(text.find(snippet))

    # 2️⃣ Build embeddings in batch (fast)
    corpus_embs = _model.encode(windows, batch_size=64, normalize_embeddings=True)
    query_emb  = _model.encode(query, normalize_embeddings=True)

    # 3️⃣ Rank by cosine similarity
    scores = cosine_similarity([query_emb], corpus_embs).flatten()
    best_idx = scores.argsort()[::-1][:top_n]

    return [(windows[i], float(scores[i]), char_offsets[i]) for i in best_idx]

# --- Example usage ----------------------------------------------------------
if __name__ == "__main__":
    #corpus = (
    #    "Fuzzy searching allows finding a piece of text that is similar to the search "
    #    "query, even when the exact words, order, or spelling differ slightly."
    #)
    with open("data/amdersen.md", "r", encoding="utf-8") as f:
        corpus = f.read()

    hits = embed_search("Какая инфраструктура представлена в комплексе?", corpus, span=150, top_n=3)
    for snippet, score, pos in hits:
        print(f"{score:0.3f} @char {pos}: {snippet}")