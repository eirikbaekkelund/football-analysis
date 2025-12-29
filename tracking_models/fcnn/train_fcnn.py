import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from soccernet.data import PlayerTrackingDataset, tracking_collate_fn


def get_player_detector_model(num_classes=1):
    """
    Returns a Pre-trained Faster R-CNN model.
    NOTE: more accurate for small objects (players) vs. YOLO with latency trade-off.

    Args:
        num_classes (int): Number of output classes (including background) (i.e., 1 for player + 1 for background)

    Returns:
        model (torchvision.models.detection.FasterRCNN): The modified Faster R-CNN model
    """
    # NOTE: default is pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # new head detection layer for our specific number of classes (2 = background + player)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def adapt_batch_for_torchvision(batch, device):
    """
    Torchvision models expect:
    - images: List[Tensor[C, H, W]]
    - targets: List[Dict{'boxes': Tensor, 'labels': Tensor}]

    Your custom Dataset returns stacked tensors, so we must unstack them.
    """
    images = list(image.to(device).float() / 255.0 for image in batch['images'])

    targets = []
    for i in range(len(images)):
        boxes = batch['boxes'][i].to(device)

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        num_objs = boxes.shape[0]
        labels = torch.ones((num_objs,), dtype=torch.int64, device=device)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([i], device=device)

        targets.append(target)

    return images, targets


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, print_freq=10):
    model.train()

    running_loss = 0.0

    for batch_idx, batch in enumerate(data_loader):
        images, targets = adapt_batch_for_torchvision(batch, device)

        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()

        if batch_idx % print_freq == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f} "
                f"(Cls: {loss_dict['loss_classifier'].item():.3f} | "
                f"Box: {loss_dict['loss_box_reg'].item():.3f})"
            )

    return running_loss / len(data_loader)


def run_training_pipeline(path_to_data_zip: str = "soccernet/tracking/tracking/train.zip"):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.005 * (BATCH_SIZE // 32)
    SAVE_PATH = "player_tracker_v1.pth"

    print(f"Starting Training on {DEVICE} with Batch Size {BATCH_SIZE}...")

    train_dataset = PlayerTrackingDataset(zip_path=path_to_data_zip, bbox_format="xyxy", extract_colors=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=tracking_collate_fn,
    )

    model = get_player_detector_model(num_classes=2)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler('cuda')

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, scaler)
        lr_scheduler.step()
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            },
            f"checkpoint_epoch_{epoch}.pth",
        )

    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    run_training_pipeline()
