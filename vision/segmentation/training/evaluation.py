import torch
from tqdm import tqdm

from vision.segmentation.metrics.segmentation import (
    segmentation_counts,
    segmentation_metrics,
)


@torch.inference_mode()
def test_evaluation(
    data_loader,
    model,
    criterion,
    device,
    num_classes=3,
):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    # test dataset全体でclassごとの値を蓄積
    test_intersections = [0] * num_classes
    test_unions = [0] * num_classes
    test_pred_counts = [0] * num_classes
    test_target_counts = [0] * num_classes

    for images, masks in tqdm(
        data_loader,
        desc="test",
        leave=False,
    ):
        images = images.to(device)
        masks = masks.to(device)

        batch_size = images.size(0)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        # [B, C, H, W] -> [B, H, W]
        pred_masks = logits.argmax(dim=1)

        # このbatchのclassごとのcount
        (
            batch_intersections,
            batch_unions,
            batch_pred_counts,
            batch_target_counts,
        ) = segmentation_counts(
            pred_masks,
            masks,
            num_classes,
        )

        # test dataset全体に加算
        for class_id in range(num_classes):
            test_intersections[class_id] += batch_intersections[class_id]
            test_unions[class_id] += batch_unions[class_id]
            test_pred_counts[class_id] += batch_pred_counts[class_id]
            test_target_counts[class_id] += batch_target_counts[class_id]

    # =================================================
    # Test Loss
    # =================================================
    avg_test_loss = running_loss / total_samples

    # =================================================
    # Metrics
    # =================================================
    class_ious, test_miou, class_dices, test_mdice = segmentation_metrics(
        test_intersections,
        test_unions,
        test_pred_counts,
        test_target_counts,
    )

    print(
        f"Test Loss: {avg_test_loss:.4f} | "
        f"mIoU: {test_miou:.4f} | "
        f"mDice: {test_mdice:.4f} | "
        f"class IoU: {[round(x, 4) for x in class_ious]} | "
        f"class Dice: {[round(x, 4) for x in class_dices]}"
    )

    return (
        avg_test_loss,
        class_ious,
        test_miou,
        class_dices,
        test_mdice,
    )
