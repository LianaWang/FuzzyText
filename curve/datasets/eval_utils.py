def polygon_iou(poly1, poly2):
    """
    Intersection over union between two shapely polygons.
    """
    iou = 0.0
    if poly1.intersects(poly2):
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        iou = float(inter_area) / max(union_area, 1e-5)
    return iou

def eval_polygons(pred_list, gt_list):
    assert len(pred_list) == len(gt_list), "sample numbers should be same"
    num_gt = 0
    num_pred = 0
    num_gt_correct = 0
    num_pred_correct = 0
    measure_list = []
    for sample_idx in range(len(pred_list)):
        gt_polygons = gt_list[sample_idx]
        pred_polygons = pred_list[sample_idx]
        num_gt += len(gt_polygons)
        num_pred += len(pred_polygons)
        cur_correct_num = 0
        for gt_polygon, hard in gt_polygons:
            num_gt -= 1 if hard else 0
            gt_correct = False
            for pred_polygon, score in pred_polygons:
                iou = polygon_iou(gt_polygon, pred_polygon)
                if iou >= 0.5:
                    num_pred_correct += 1 if not hard else 0
                    num_pred -= 1 if hard else 0
                    gt_correct = True
            num_gt_correct += 1 if (gt_correct and not hard) else 0
            cur_correct_num += 1 if gt_correct else 0
        p = cur_correct_num / max(len(pred_polygons), 1e-10)
        r = cur_correct_num / max(len(gt_polygons), 1e-10)
        f = 2.0 * p * r / max(p + r, 1e-10)
        measure_list.append(f)
    p = 1.0 * num_pred_correct / max(num_pred, 1e-10)
    r = 1.0 * num_gt_correct / max(num_gt, 1e-10)
    f = 2.0 * p * r / max(p + r, 1e-10)
#    output_string = "correct_pred/all_pred (%d/%d); correct_gt/all_gt (%d/%d)\n" % (
#        num_pred_correct, num_pred, num_gt_correct, num_gt)
    output_string = ""
    for name, val in [('precision', p), ('recall', r), ('f_measure', f)]:
        output_string += "%s : %.3f%%\t" % (name, val * 100.0)
    output_string += "\n"
    return dict(precision=p, recall=r, f_measure=f, output_string=output_string)
