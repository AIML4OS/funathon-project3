import torch


def IOU(output: torch.Tensor, labels: torch.Tensor, logits: bool) -> torch.Tensor:
    """
    Calculate Mean Intersection Over Union for segmentation tasks.
    Handles both binary and multiclass segmentation.

    Args:
        output: Output of the segmentation model (B, C, H, W) for multiclass or (B, H, W) for binary
        labels: True segmentation mask (B, H, W) with class indices
        logits: Boolean True if output contains logits instead of probabilities

    Returns:
        torch.Tensor: Mean IOU across all classes and batch
    """
    if output.dim() == 3:  # Binary case
        if logits:
            output = torch.sigmoid(output)
        preds = (output > 0.5).float()
        num_classes = 2
    else:  # Multiclass case
        if logits:
            output = torch.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)
        num_classes = output.shape[1]

    # Convert to one-hot for multiclass
    if num_classes > 2:
        labels_one_hot = torch.zeros_like(output)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        preds_one_hot = torch.zeros_like(output)
        preds_one_hot.scatter_(1, preds.unsqueeze(1), 1)
    else:  # Binary case
        labels_one_hot = labels
        preds_one_hot = preds

    ious = []
    # For binary, only compute IOU for positive class
    # For multiclass, compute IOU for all classes (TODO: To be discussed, "Autres" class may not be relevant)
    class_range = range(1, num_classes) if num_classes == 2 else range(num_classes)

    for cls in class_range:
        if num_classes > 2:
            pred_cls = preds_one_hot[:, cls]
            label_cls = labels_one_hot[:, cls]
        else:
            pred_cls = preds_one_hot
            label_cls = labels_one_hot

        intersection = torch.sum(pred_cls * label_cls, dim=[1, 2])
        union = torch.sum(torch.clamp(pred_cls + label_cls, max=1), dim=[1, 2])

        iou_cls = intersection / union

        if num_classes == 2:
            # Handle cases where class is not present in both prediction and ground truth for binary case
            # TODO: ici si on a des images sans aucun pixel bâtiment
            # on "biaise" potentiellement l'IOU vers le haut, parce que si
            # on prédit rien on a une IOU de 1. N'arrive pas si on
            # n'a pas d'images sans pixel bâtiment
            iou_cls = torch.tensor(
                [1 if torch.isnan(x) else x for x in iou_cls],
                dtype=torch.float,
                device=output.device,
            )
        else:
            # TODO: For now we do the same but to adjust soon
            iou_cls = torch.tensor(
                [1 if torch.isnan(x) else x for x in iou_cls],
                dtype=torch.float,
                device=output.device,
            )

        ious.append(torch.mean(iou_cls))

    # Nan could only appear in the multiclass case and we do not want to upper biased the IOU
    # TODO : depending on id2label return iou for each class
    return torch.mean(torch.stack(ious)), ious[1]


def positive_rate(output: torch.Tensor, logits: bool) -> torch.Tensor:
    """
    Compute percentage of pixels predicted as building.
    Handles both binary and multiclass segmentation.

    Args:
        output (torch.Tensor): Batch prediction (B, C, H, W) for multiclass or (B, H, W) for binary
        logits (bool): Boolean True if output contains logits instead of probabilities

    Returns:
        torch.Tensor: Average percentage of pixels predicted as non-background for each class
    """
    if output.dim() == 3:  # Binary case
        if logits:
            output = torch.sigmoid(output)
        preds = (output > 0.5).float()
    else:  # Multiclass case
        if logits:
            output = torch.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)

    # We assume the building class is the value 1. TODO : make it consistent with the labeler label2id
    return (preds == 1).float().mean()
