from metrics import calculate_iou, mean_average_precision, precision_recall_at_threshold


def test_iou_identical_boxes_is_1():
    box = [0, 0, 10, 10]
    assert calculate_iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_0():
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    # box1: 0..10, box2: 5..15 -> intersection 5x5=25, each area 100, union=175
    iou = calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert abs(iou - 25 / 175) < 1e-6


def _perfect_match_result():
    return [
        {
            "pred_boxes": [[0, 0, 10, 10]],
            "pred_labels": [1],
            "pred_scores": [0.9],
            "true_boxes": [[0, 0, 10, 10]],
            "true_labels": [1],
        }
    ]


def test_precision_recall_perfect_match():
    result = precision_recall_at_threshold(_perfect_match_result(), iou_threshold=0.5, score_threshold=0.5)
    assert result["precision"] > 0.99
    assert result["recall"] > 0.99
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_precision_recall_no_predictions_is_zero_recall():
    image_results = [
        {"pred_boxes": [], "pred_labels": [], "pred_scores": [], "true_boxes": [[0, 0, 10, 10]], "true_labels": [1]}
    ]
    result = precision_recall_at_threshold(image_results, iou_threshold=0.5, score_threshold=0.5)
    assert result["recall"] < 0.01
    assert result["fn"] == 1


def test_precision_recall_low_confidence_prediction_ignored():
    image_results = [
        {
            "pred_boxes": [[0, 0, 10, 10]],
            "pred_labels": [1],
            "pred_scores": [0.1],  # below score_threshold
            "true_boxes": [[0, 0, 10, 10]],
            "true_labels": [1],
        }
    ]
    result = precision_recall_at_threshold(image_results, iou_threshold=0.5, score_threshold=0.5)
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["fp"] == 0


def test_map_perfect_predictions_is_1():
    result = mean_average_precision(_perfect_match_result(), iou_threshold=0.5)
    assert result["mAP"] > 0.99
    assert result["per_class_AP"][1] > 0.99


def test_map_no_predictions_is_0():
    image_results = [
        {"pred_boxes": [], "pred_labels": [], "pred_scores": [], "true_boxes": [[0, 0, 10, 10]], "true_labels": [1]}
    ]
    result = mean_average_precision(image_results, iou_threshold=0.5)
    assert result["mAP"] == 0.0
