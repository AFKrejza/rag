"""Top-k semantic search over the indexed corpus."""

from dataclasses import dataclass

from . import config


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    distance: float

    @property
    def citation(self) -> str:
        return f"[{self.source}, p. {self.page}]"


def _neighbour_ids(chunk: RetrievedChunk) -> list[str]:
    i = chunk.chunk_index
    return [f"{chunk.source}:{chunk.page}:{j}" for j in (i - 1, i + 1) if j >= 0]


def retrieve(collection, question: str, k: int = config.TOP_K,
             max_distance: float = config.MAX_DISTANCE,
             neighbours: bool = config.INCLUDE_NEIGHBOURS) -> list[RetrievedChunk]:
    """Return up to k chunks closer than max_distance, nearest first.

    With neighbours=True, the chunks just before and after each hit on the same
    page are added too (inheriting the hit's distance), so lists or definitions
    that straddle a chunk boundary reach the LLM in one piece.
    """
    if collection.count() == 0:
        return []
    res = collection.query(query_texts=[question], n_results=min(k, collection.count()))
    hits = [
        RetrievedChunk(id=cid, text=doc, source=meta["source"], page=int(meta["page"]),
                       chunk_index=int(meta["chunk_index"]), distance=dist)
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0])
        if dist <= max_distance
    ]
    if not neighbours or not hits:
        return hits

    seen = {h.id for h in hits}
    # dict.fromkeys de-duplicates (two hits can share a neighbour) while keeping order.
    wanted = list(dict.fromkeys(nid for h in hits for nid in _neighbour_ids(h) if nid not in seen))
    found = collection.get(ids=wanted) if wanted else {"ids": [], "documents": [], "metadatas": []}
    by_id = dict(zip(found["ids"], zip(found["documents"], found["metadatas"])))

    # Keep each neighbour right next to its hit, in reading order.
    result: list[RetrievedChunk] = []
    for h in hits:
        for cid in (f"{h.source}:{h.page}:{h.chunk_index - 1}", h.id, f"{h.source}:{h.page}:{h.chunk_index + 1}"):
            if cid in seen and cid != h.id:
                continue
            if cid == h.id:
                result.append(h)
            elif cid in by_id:
                doc, meta = by_id[cid]
                result.append(RetrievedChunk(id=cid, text=doc, source=meta["source"], page=int(meta["page"]),
                                             chunk_index=int(meta["chunk_index"]), distance=h.distance))
            seen.add(cid)
    return result
