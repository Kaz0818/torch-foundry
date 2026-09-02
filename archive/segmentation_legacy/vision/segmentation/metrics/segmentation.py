from torch import Tensor


def segmentation_counts(
    pred: Tensor,
    target: Tensor,
    num_classes: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    各classについて、IoU / Diceに必要なcountを計算する。

    pred:   [B, H, W]
    target: [B, H, W]
    """
    intersections = [0] * num_classes
    unions = [0] * num_classes
    pred_counts = [0] * num_classes
    target_counts = [0] * num_classes

    for class_id in range(num_classes):
        pred_class = pred == class_id
        target_class = target == class_id

        intersections[class_id] = int((pred_class & target_class).sum().item())

        unions[class_id] = int((pred_class | target_class).sum().item())

        pred_counts[class_id] = int(pred_class.sum().item())

        target_counts[class_id] = int(target_class.sum().item())

    return (
        intersections,
        unions,
        pred_counts,
        target_counts,
    )


def segmentation_metrics(
    intersections: list[int],
    unions: list[int],
    pred_counts: list[int],
    target_counts: list[int],
) -> tuple[list[float], float, list[float], float]:
    class_ious: list[float] = []
    class_dices: list[float] = []

    for intersection, union, pred_count, target_count in zip(
        intersections,
        unions,
        pred_counts,
        target_counts,
    ):
        # IoU
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        # Dice
        denominator = pred_count + target_count

        if denominator == 0:
            dice = 1.0
        else:
            dice = (2 * intersection) / denominator

        class_ious.append(iou)
        class_dices.append(dice)

    mean_iou = sum(class_ious) / len(class_ious)
    mean_dice = sum(class_dices) / len(class_dices)

    return class_ious, mean_iou, class_dices, mean_dice
