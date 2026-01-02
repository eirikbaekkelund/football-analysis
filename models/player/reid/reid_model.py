import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm


@dataclass
class ReIDConfig:
    """Configuration for ReID model."""

    embedding_dim: int = 512
    num_classes: int = 1000  # Number of unique identities for training
    backbone: str = "osnet"  # osnet, resnet50, efficientnet
    input_size: Tuple[int, int] = (128, 256)  # width, height
    use_team_head: bool = True
    use_role_head: bool = True  # player, referee, goalkeeper
    dropout: float = 0.3
    pretrained: bool = True


class ConvBlock(nn.Module):
    """Basic conv-bn-relu block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution with depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """Channel attention gate (squeeze-excitation style)."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = in_channels // reduction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(mid_channels, in_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.global_pool(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return input * x


class OSBlock(nn.Module):
    """Omni-Scale Block from OSNet."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = out_channels // reduction

        self.conv1 = ConvBlock(in_channels, mid_channels, 1, 1, 0)

        # Multi-scale branches
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )

        self.gate = ChannelGate(mid_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)

        # Multi-scale aggregation
        x2a = self.conv2a(x)
        x2b = self.conv2b(x)
        x2c = self.conv2c(x)
        x2d = self.conv2d(x)

        x = x2a + x2b + x2c + x2d
        x = self.gate(x)
        x = self.conv3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return F.relu(x + residual)


class OSNet(nn.Module):
    """
    Omni-Scale Network (OSNet) for person re-identification.

    A lightweight and efficient CNN architecture designed for ReID tasks.
    Uses omni-scale feature learning with multi-scale convolutions.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        pretrained: bool = False,
    ):
        super().__init__()

        # Stem
        self.conv1 = ConvBlock(3, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Stages
        self.stage2 = nn.Sequential(
            OSBlock(64, 256),
            OSBlock(256, 256),
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(2, 2),
        )

        self.stage3 = nn.Sequential(
            OSBlock(256, 384),
            OSBlock(384, 384),
        )
        self.pool3 = nn.Sequential(
            nn.Conv2d(384, 384, 1, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.AvgPool2d(2, 2),
        )

        self.stage4 = nn.Sequential(
            OSBlock(384, 512),
            OSBlock(512, 512),
        )

        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Classifier for training
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]
            return_embedding: If True, return (logits, embedding)

        Returns:
            Logits or (logits, embedding) tuple
        """
        # Backbone
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.stage4(x)

        # Global pooling and embedding
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        embedding = self.bn(embedding)

        # Dropout and classification
        x = self.dropout(embedding)
        logits = self.classifier(x)

        if return_embedding:
            return logits, embedding
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding features only (for inference)."""
        _, embedding = self.forward(x, return_embedding=True)
        return F.normalize(embedding, p=2, dim=1)


class FootballReIDModel(nn.Module):
    """
    Football-specific ReID model with multi-task heads.

    Adds auxiliary heads for:
    - Team classification (team A, team B, referee)
    - Role classification (player, goalkeeper, referee, ball)
    - Jersey number recognition (optional)
    """

    def __init__(self, config: ReIDConfig):
        super().__init__()
        self.config = config

        # Backbone
        self.backbone = OSNet(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            pretrained=config.pretrained,
        )

        # Auxiliary heads
        if config.use_team_head:
            self.team_head = nn.Sequential(
                nn.Linear(config.embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3),  # team_a, team_b, neutral
            )

        if config.use_role_head:
            self.role_head = nn.Sequential(
                nn.Linear(config.embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 4),  # player, goalkeeper, referee, ball
            )

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.

        Args:
            x: Input tensor [B, 3, H, W]
            return_all: Return all outputs including auxiliary heads

        Returns:
            Dictionary with 'embedding', 'identity_logits', 'team_logits', 'role_logits'
        """
        identity_logits, embedding = self.backbone(x, return_embedding=True)

        outputs = {
            "embedding": F.normalize(embedding, p=2, dim=1),
            "identity_logits": identity_logits,
        }

        if return_all:
            if self.config.use_team_head:
                outputs["team_logits"] = self.team_head(embedding)
            if self.config.use_role_head:
                outputs["role_logits"] = self.role_head(embedding)

        return outputs

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized embedding features."""
        return self.backbone.extract_features(x)


class ReIDFeatureExtractor:
    """
    Feature extractor for inference.

    Extracts ReID embeddings from person crops for matching and tracking.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        input_size: Tuple[int, int] = (128, 256),
    ):
        """
        Initialize feature extractor.

        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
            input_size: Expected input size (width, height)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size[::-1]),  # (H, W)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, model_path: str) -> FootballReIDModel:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu")

        # Get config from checkpoint or use default
        config = checkpoint.get("config", ReIDConfig())
        if isinstance(config, dict):
            config = ReIDConfig(**config)

        model = FootballReIDModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    @torch.no_grad()
    def extract(
        self,
        images: Union[np.ndarray, List[np.ndarray], torch.Tensor],
    ) -> np.ndarray:
        """
        Extract features from images.

        Args:
            images: Single image, list of images, or batch tensor
                   Images should be RGB, shape (H, W, 3)

        Returns:
            Feature array of shape (N, embedding_dim)
        """
        # Handle single image
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        # Convert to PIL and apply transforms
        if isinstance(images, list):
            batch = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                batch.append(self.transform(img))
            batch = torch.stack(batch)
        else:
            batch = images

        batch = batch.to(self.device)

        # Extract features
        features = self.model.extract_features(batch)

        return features.cpu().numpy()

    def compute_distance(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute distance matrix between query and gallery features.

        Args:
            query_features: Query features (N, D)
            gallery_features: Gallery features (M, D)
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            Distance matrix (N, M)
        """
        if metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(query_features, gallery_features.T)
            return 1 - similarity
        elif metric == "euclidean":
            # Euclidean distance
            q2 = np.sum(query_features**2, axis=1, keepdims=True)
            g2 = np.sum(gallery_features**2, axis=1, keepdims=True)
            dist = q2 + g2.T - 2 * np.dot(query_features, gallery_features.T)
            return np.sqrt(np.maximum(dist, 0))
        else:
            raise ValueError(f"Unknown metric: {metric}")


class ReIDDataset(Dataset):
    """
    Dataset for ReID training.

    Expects directory structure:
        root/
            person_id_1/
                img1.jpg
                img2.jpg
                ...
            person_id_2/
                ...
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        min_samples: int = 2,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing person folders
            transform: Optional transforms
            min_samples: Minimum samples per identity
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.samples = []  # List of (image_path, person_id)
        self.id_to_label = {}  # Map person_id to integer label

        self._build_dataset(min_samples)

    def _build_dataset(self, min_samples: int) -> None:
        """Build dataset index."""
        person_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]

        label = 0
        for person_dir in person_dirs:
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

            if len(images) < min_samples:
                continue

            person_id = person_dir.name
            self.id_to_label[person_id] = label

            for img_path in images:
                self.samples.append((str(img_path), label))

            label += 1

        print(f"ReIDDataset: {len(self.samples)} samples, {label} identities")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.id_to_label)


class TripletLoss(nn.Module):
    """
    Triplet loss with hard mining.

    Selects hardest positive and negative samples within a batch.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with hard mining.

        Args:
            embeddings: Feature embeddings (B, D)
            labels: Identity labels (B,)

        Returns:
            Triplet loss value
        """
        # Compute pairwise distances
        dist = self._pairwise_distance(embeddings)

        # For each anchor, find hardest positive and negative
        loss = 0.0
        n_triplets = 0

        for i in range(len(labels)):
            anchor_label = labels[i]

            # Positive: same identity
            pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=labels.device) != i)
            # Negative: different identity
            neg_mask = labels != anchor_label

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # Hardest positive: furthest same-identity sample
            hardest_pos = dist[i][pos_mask].max()
            # Hardest negative: closest different-identity sample
            hardest_neg = dist[i][neg_mask].min()

            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss += triplet_loss
            n_triplets += 1

        return loss / max(n_triplets, 1)

    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Numerical stability
        return torch.sqrt(distances + 1e-8)


class ReIDTrainer:
    """
    Trainer for ReID models.

    Supports:
    - Cross-entropy loss for identity classification
    - Triplet loss with hard mining
    - Multi-task learning (team, role)
    """

    def __init__(
        self,
        model: FootballReIDModel,
        train_dataset: ReIDDataset,
        val_dataset: Optional[ReIDDataset] = None,
        device: str = "cuda",
        learning_rate: float = 3.5e-4,
        weight_decay: float = 5e-4,
        triplet_margin: float = 0.3,
        triplet_weight: float = 1.0,
        ce_weight: float = 1.0,
    ):
        """
        Initialize trainer.

        Args:
            model: ReID model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay
            triplet_margin: Triplet loss margin
            triplet_weight: Weight for triplet loss
            ce_weight: Weight for cross-entropy loss
        """
        self.device = torch.device(device if torch.cuda.is_available() else device)
        self.model = model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1,
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=triplet_margin)

        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_triplet_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images, return_all=False)

            # Cross-entropy loss
            ce_loss = self.ce_loss(outputs["identity_logits"], labels)

            # Triplet loss
            triplet_loss = self.triplet_loss(outputs["embedding"], labels)

            # Combined loss
            loss = self.ce_weight * ce_loss + self.triplet_weight * triplet_loss

            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item()

            _, predicted = outputs["identity_logits"].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": 100.0 * correct / total,
                }
            )

        return {
            "loss": total_loss / len(dataloader),
            "ce_loss": total_ce_loss / len(dataloader),
            "triplet_loss": total_triplet_loss / len(dataloader),
            "accuracy": 100.0 * correct / total,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        all_embeddings = []
        all_labels = []
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images, return_all=False)

            all_embeddings.append(outputs["embedding"].cpu())
            all_labels.append(labels.cpu())

            _, predicted = outputs["identity_logits"].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Compute CMC and mAP (simplified)
        accuracy = 100.0 * correct / total

        return {
            "accuracy": accuracy,
        }

    def train(
        self,
        num_epochs: int,
        batch_size: int = 32,
        save_dir: str = "checkpoints",
        save_every: int = 10,
    ) -> None:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs
            batch_size: Batch size
            save_dir: Directory for checkpoints
            save_every: Save checkpoint every N epochs
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

        best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}: {train_metrics}")

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                print(f"Validation: {val_metrics}")

                if val_metrics["accuracy"] > best_acc:
                    best_acc = val_metrics["accuracy"]
                    self.save_checkpoint(save_dir / "best_model.pth", epoch)

            self.scheduler.step()

            if epoch % save_every == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch}.pth", epoch)

        # Save final model
        self.save_checkpoint(save_dir / "final_model.pth", num_epochs)

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.config.__dict__,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")
