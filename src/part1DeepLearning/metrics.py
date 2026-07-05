# src/part1DeepLearning/metrics.py
# --------------------------------------------------
# Detection metrics: IoU, single-threshold precision/recall (what the original
# evaluation.py computed), and per-class Average Precision + mAP@IoU=0.5
# (standard object-detection benchmark metric, e.g. what "Results" sections in
# published detection papers report) using all-points PR-curve interpolation.
# --------------------------------------------------

from collections import defaultdict

import torch


def calculate_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def precision_recall_at_threshold(image_results: list[dict], iou_threshold: float, score_threshold: float) -> dict:
    """image_results: list of {"pred_boxes", "pred_labels", "pred_scores", "true_boxes", "true_labels"} per image."""
    total_tp = total_fp = total_fn = 0

    for res in image_results:
        pred_boxes, pred_labels, pred_scores = res["pred_boxes"], res["pred_labels"], res["pred_scores"]
        true_boxes, true_labels = res["true_boxes"], res["true_labels"]

        matched = []
        for i, true_box in enumerate(true_boxes):
            found_match = False
            for j, pred_box in enumerate(pred_boxes):
                if pred_scores[j] < score_threshold or j in matched:
                    continue
                iou = calculate_iou(true_box, pred_box)
                if iou >= iou_threshold and true_labels[i] == pred_labels[j]:
                    total_tp += 1
                    matched.append(j)
                    found_match = True
                    break
            if not found_match:
                total_fn += 1

        num_confident_preds = sum(1 for s in pred_scores if s >= score_threshold)
        total_fp += num_confident_preds - len(matched)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    return {"precision": precision, "recall": recall, "tp": total_tp, "fp": total_fp, "fn": total_fn}


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    """All-points interpolated AP (COCO/VOC-2010+ style)."""
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def mean_average_precision(image_results: list[dict], iou_threshold: float = 0.5) -> dict:
    """Returns {"mAP": float, "per_class_AP": {label: AP}}."""
    detections_by_class = defaultdict(list)  # label -> list of (score, image_idx, box)
    gt_by_class_image = defaultdict(lambda: defaultdict(list))  # label -> image_idx -> [boxes]
    gt_count_by_class = defaultdict(int)

    for img_idx, res in enumerate(image_results):
        for label, box in zip(res["true_labels"], res["true_boxes"]):
            label = int(label)
            gt_by_class_image[label][img_idx].append(list(box))
            gt_count_by_class[label] += 1

        for label, box, score in zip(res["pred_labels"], res["pred_boxes"], res["pred_scores"]):
            detections_by_class[int(label)].append((float(score), img_idx, list(box)))

    per_class_ap = {}
    for label, gt_total in gt_count_by_class.items():
        detections = sorted(detections_by_class.get(label, []), key=lambda d: d[0], reverse=True)
        matched_gt = defaultdict(set)  # image_idx -> set of matched gt box indices

        tps = []
        fps = []
        for _score, img_idx, box in detections:
            gts = gt_by_class_image[label].get(img_idx, [])
            best_iou, best_gt_idx = 0.0, -1
            for gt_idx, gt_box in enumerate(gts):
                if gt_idx in matched_gt[img_idx]:
                    continue
                iou = calculate_iou(box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx

            if best_iou >= iou_threshold:
                matched_gt[img_idx].add(best_gt_idx)
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)

        cum_tp = torch.cumsum(torch.tensor(tps, dtype=torch.float32), dim=0) if tps else torch.tensor([])
        cum_fp = torch.cumsum(torch.tensor(fps, dtype=torch.float32), dim=0) if fps else torch.tensor([])

        if len(cum_tp) == 0:
            per_class_ap[label] = 0.0
            continue

        recalls = (cum_tp / (gt_total + 1e-6)).tolist()
        precisions = (cum_tp / (cum_tp + cum_fp + 1e-6)).tolist()
        per_class_ap[label] = _average_precision(recalls, precisions)

    mAP = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0
    return {"mAP": mAP, "per_class_AP": per_class_ap}
