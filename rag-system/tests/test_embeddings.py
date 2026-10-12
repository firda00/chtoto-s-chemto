import numpy as np

from rag_system.embeddings import HashingEmbedder, get_embedder


def test_hashing_is_deterministic_and_normalized(embedder: HashingEmbedder) -> None:
    texts = ["Сколько стоит гостевой день?", "Wi-Fi speed"]
    a = embedder.embed(texts)
    b = HashingEmbedder().embed(texts)
    assert a.shape == (2, embedder.dim)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)


def test_hashing_similarity_is_meaningful(embedder: HashingEmbedder) -> None:
    q = embedder.embed_query("цена гостевого дня")
    docs = embedder.embed(["Гостевой день стоит 120 000 сум", "Dogs are welcome on Fridays"])
    sims = docs @ q
    assert sims[0] > sims[1]


def test_empty_input(embedder: HashingEmbedder) -> None:
    assert embedder.embed([]).shape == (0, embedder.dim)
    assert np.isfinite(embedder.embed_query("")).all()


def test_get_embedder_hashing() -> None:
    assert get_embedder("hashing").name == "hashing"
