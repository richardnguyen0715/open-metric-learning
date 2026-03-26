import pytest
import torch
from torch import BoolTensor, FloatTensor, LongTensor
from math import log2

from oml.functional.metrics import calc_ndcg, calc_retrieval_metrics

def test_ndcg_edge_cases():
    # 1. Perfect ranking (NDCG = 1)
    gt_tops = [BoolTensor([1, 1, 0, 0])]
    n_gts = [2]
    res = calc_ndcg(gt_tops, n_gts, top_k=(2, 4))
    assert torch.allclose(res[0], FloatTensor([1.0]))
    assert torch.allclose(res[1], FloatTensor([1.0]))

    # 2. Worst-case ranking (NDCG = 0 if k <= bad)
    gt_tops = [BoolTensor([0, 0, 1, 1])]
    n_gts = [2]
    res = calc_ndcg(gt_tops, n_gts, top_k=(2,))
    assert torch.allclose(res[0], FloatTensor([0.0]))
    
    # 3. Multiple cutoffs
    gt_tops = [BoolTensor([1, 0])]
    n_gts = [2]
    res = calc_ndcg(gt_tops, n_gts, top_k=(1, 2))
    assert torch.allclose(res[0], FloatTensor([1.0]))
    expected_ndcg_at_2 = 1.0 / (1.0/log2(2) + 1.0/log2(3))
    assert torch.allclose(res[1], FloatTensor([expected_ndcg_at_2]), atol=1e-4)
    
    # 4. Empty targets (n_gts=0, len=0) -> 1.0
    gt_tops = [BoolTensor([])]
    n_gts = [0]
    res = calc_ndcg(gt_tops, n_gts, top_k=(5,))
    assert torch.allclose(res[0], FloatTensor([1.0]))

    # 5. Empty predictions, non-empty targets -> 0.0
    gt_tops = [BoolTensor([])]
    n_gts = [5]
    res = calc_ndcg(gt_tops, n_gts, top_k=(5,))
    assert torch.allclose(res[0], FloatTensor([0.0]))

    # 6. Non-empty predictions, empty targets -> 0.0
    gt_tops = [BoolTensor([0, 0])]
    n_gts = [0]
    res = calc_ndcg(gt_tops, n_gts, top_k=(5,))
    assert torch.allclose(res[0], FloatTensor([0.0]))

def test_ndcg_via_calc_retrieval_metrics():
    retrieved_ids = [LongTensor([0, 5, 4]), LongTensor([2, 1, 5]), LongTensor([2, 1, 5])]
    gt_ids = [LongTensor([0, 10, 4]), LongTensor([2, 3]), LongTensor([2, 3])]

    metrics = calc_retrieval_metrics(
        retrieved_ids=retrieved_ids, 
        gt_ids=gt_ids, 
        cmc_top_k=tuple(), 
        precision_top_k=tuple(), 
        map_top_k=tuple(), 
        ndcg_top_k=(1, 3),
        reduce=False
    )
    
    assert "ndcg" in metrics
    assert 1 in metrics["ndcg"]
    assert 3 in metrics["ndcg"]
    
    expected_ndcg_1 = [1.0, 1.0, 1.0]
    expected_ndcg_3 = [
        1.5 / (1.0 + 1.0/log2(3) + 0.5),
        1.0 / (1.0 + 1.0/log2(3)),
        1.0 / (1.0 + 1.0/log2(3))
    ]
    
    assert torch.allclose(metrics["ndcg"][1], FloatTensor(expected_ndcg_1), atol=1e-4)
    assert torch.allclose(metrics["ndcg"][3], FloatTensor(expected_ndcg_3), atol=1e-4)
