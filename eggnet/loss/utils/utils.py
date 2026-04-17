import torch

from eggnet.utils.mapping import get_target, get_weight


def hinge_loss(
    batch,
    edges,
    margin,
    y=None,
    w=None,
    f=None,
    node_filter=False,
    weighting_config=None,
    sum=False,
    node_score=False,
    use_double_metric_learning:bool = False,
):
    if y is None:
        y = get_target(edges, batch.hit_particle_id)

    if w is None:
        w = get_weight(batch, edges, y, weighting_config=weighting_config)
    elif weighting_config is not None:
        w *= get_weight(batch, edges, y, weighting_config=weighting_config)
    if node_score:
        beta = torch.sigmoid(batch.hit_score).flatten()
        w *= beta[edges[0]] * beta[edges[1]]

    if f is None:
        f = torch.ones(edges.shape[1], device=edges.device)
    if use_double_metric_learning:
        d = get_distances(
            (batch.tgt_embedding, batch.src_embedding), edges, batch.filter_node_list if node_filter else None,
            use_double_metric_learning=True
        )
    else:
        d = get_distances(
            batch.hit_embedding, edges, batch.filter_node_list if node_filter else None
        )

    loss = torch.nn.functional.hinge_embedding_loss(
        d,
        y,
        margin=margin,
        reduction="none",
    ).pow(2)
    if sum:
        return (loss * w * f).sum()
    else:
        return (loss * w * f).sum() / w.sum()

def get_distances(node_embedding, edges, filter_node_list=None, use_double_metric_learning=False):
    # Note: node_embedding is expected to be a tuple (tgt_emb, src_emb) if `use_double_metric_learning` is True
    if filter_node_list is not None:
        res = torch.full((edges.shape[1],), 2.0, device=node_embedding.device)
        node_map = torch.full(
            (filter_node_list.max() + 1,), -1, device=node_embedding.device
        )
        node_map[filter_node_list] = torch.arange(
            len(filter_node_list), device=node_embedding.device
        )
        edge_mask = torch.isin(edges, filter_node_list).all(dim=0)
        edges = node_map[edges.T[edge_mask].T]
    if use_double_metric_learning:
        if not isinstance(node_embedding, tuple) or len(node_embedding) != 2:
            raise ValueError("node_embedding must be a tuple of (tgt_emb, src_emb) when use_double_metric_learning is True.")
        reference = node_embedding[0][edges[1]] # src embedding
        neighbors = node_embedding[1][edges[0]] # tgt embedding
    else:
        if isinstance(node_embedding, tuple):
            raise ValueError("node_embedding must be a tuple of (tgt_emb, src_emb) when use_double_metric_learning is True.")
        reference = node_embedding[edges[1]]
        neighbors = node_embedding[edges[0]]

    try:  # This can be resource intensive, so we chunk it if it fails
        d = torch.sum((reference - neighbors) ** 2, dim=-1)
    except RuntimeError:
        d = [
            torch.sum((ref - nei) ** 2, dim=-1)
            for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))
        ]
        d = torch.cat(d)

    d = torch.sqrt(d + 1e-12)

    if filter_node_list is not None:
        res[edge_mask] = d
    else:
        res = d
    return res
