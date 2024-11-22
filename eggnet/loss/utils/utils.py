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


def get_distances(node_embedding, edges, filter_node_list=None):
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
