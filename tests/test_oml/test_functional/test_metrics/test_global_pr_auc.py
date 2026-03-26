import pytest
import torch
import numpy as np
from sklearn.metrics import average_precision_score

from oml.functional.metrics import calc_global_pr_auc, calc_retrieval_metrics

def test_global_pr_auc_edge_cases():
    # 1. No positive samples
    gt_tops = [torch.BoolTensor([0, 0]), torch.BoolTensor([0])]
    distances = [torch.FloatTensor([0.1, 0.2]), torch.FloatTensor([0.1])]
    res = calc_global_pr_auc(gt_tops, distances)
    assert res == 0.0
    
    # 2. Empty inputs
    res = calc_global_pr_auc([], [])
    assert res == 0.0

def test_global_pr_auc_values():
    # Similar to macro/micro evaluation
    gt_tops = [
        torch.BoolTensor([1, 0, 1]),
        torch.BoolTensor([0, 1]),
        torch.BoolTensor([0, 0])
    ]
    distances = [
        torch.FloatTensor([0.1, 0.5, 0.3]),
        torch.FloatTensor([0.4, 0.2]),
        torch.FloatTensor([0.6, 0.7])
    ]
    
    y_true = np.array([1, 0, 1, 0, 1, 0, 0])
    y_scores = -np.array([0.1, 0.5, 0.3, 0.4, 0.2, 0.6, 0.7])
    
    expected_auc = calc_global_pr_auc(gt_tops, distances)
    
    # using average_precision_score to compare conceptually (though APS uses a slightly different integral step, they should be close)
    aps = average_precision_score(y_true, y_scores)
    assert expected_auc > 0.0
    assert abs(expected_auc - aps) < 0.1  # PR AUC can differ slightly from AP due to interpolation

def test_calc_retrieval_metrics_integration():
    retrieved_ids = [torch.LongTensor([4, 2, 8]), torch.LongTensor([5, 1, 3])]
    distances = [torch.FloatTensor([0.1, 0.5, 0.9]), torch.FloatTensor([0.2, 0.4, 0.8])]
    gt_ids = [torch.LongTensor([2, 4]), torch.LongTensor([1])]

    metrics = calc_retrieval_metrics(
        retrieved_ids=retrieved_ids,
        gt_ids=gt_ids,
        distances=distances,
        calc_global_pr_auc_metric=True,
        reduce=False
    )
    
    assert "global_pr_auc" in metrics
    assert isinstance(metrics["global_pr_auc"], float)

