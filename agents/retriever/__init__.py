"""Block 8: Hybrid Retriever package.

The Retriever runs two parallel channels (vector cosine, BM25 lexical)
over the Chunk Store and Fact Store, applies period filtering after
retrieval, and fuses the results via Reciprocal Rank Fusion. The public
entry point is ``retrieve()`` in ``retriever.py``; the four channel
modules can also be invoked directly for inspection and testing.
"""
