import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from soccernet.data import SoccerNetTrackingDataset, tracking_collate_fn

def get_player_detector_model(num_classes=1):
    """
    Returns a Pre-trained Faster R-CNN model.
    We use Faster R-CNN because it is more accurate for small objects (players) 
    than standard YOLO, though slightly slower. Perfect for the 'Accuracy' phase.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier with a new one for our classes (Background + Player)
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
        num_objs = boxes.shape[0]
        labels = torch.ones((num_objs,), dtype=torch.int64, device=device)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([i], device=device)
        
        targets.append(target)
        
    return images, targets

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    
    running_loss = 0.0
    
    for batch_idx, batch in enumerate(data_loader):
        images, targets = adapt_batch_for_torchvision(batch, device)

        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f} "
                f"(Cls: {loss_dict['loss_classifier'].item():.3f} | "
                f"Box: {loss_dict['loss_box_reg'].item():.3f})")
            
    return running_loss / len(data_loader)

def run_training_pipeline(path_to_data_zip: str = "soccernet/tracking/tracking/train.zip"):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.005
    SAVE_PATH = "player_tracker_v1.pth"
    
    print(f"Starting Training on {DEVICE}...")

    train_dataset = SoccerNetTrackingDataset(
        zip_path=path_to_data_zip,
        bbox_format="xyxy" 
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        collate_fn=tracking_collate_fn
    )

    model = get_player_detector_model(num_classes=2)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        lr_scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"checkpoint_epoch_{epoch}.pth")

    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    run_training_pipeline()