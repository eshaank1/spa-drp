"""
Phase 1: Behavioral Cloning Pre-training on Heuristic Bot Demonstrations.

This module provides a dataset and training loop to pre-train an Actor network
on demonstrations from the BaselineBot. The goal is to learn a reasonable policy
that the PPO phase can then refine.

Data Format:
  Each sample is a tuple: (state: np.ndarray, action_mask: np.ndarray, action: int)
  - state: 50-dimensional observation
  - action_mask: 14-dimensional binary mask (1=valid, 0=invalid)
  - action: integer action taken (0-13)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BCDataset(Dataset):
    """PyTorch Dataset for behavioral cloning demonstrations."""
    
    def __init__(
        self,
        demonstrations: List[Tuple[np.ndarray, np.ndarray, int]],
        device: torch.device = None
    ):
        """
        Args:
            demonstrations: List of (state, action_mask, action) tuples
                - state: np.ndarray of shape (50,)
                - action_mask: np.ndarray of shape (14,), binary
                - action: int in [0, 13]
            device: torch device (cpu or cuda)
        """
        self.demonstrations = demonstrations
        self.device = device or torch.device("cpu")
        
        self.states = torch.tensor(
            np.array([d[0] for d in demonstrations]),
            dtype=torch.float32,
            device=self.device
        )
        self.action_masks = torch.tensor(
            np.array([d[1] for d in demonstrations]),
            dtype=torch.float32,
            device=self.device
        )
        self.actions = torch.tensor(
            np.array([d[2] for d in demonstrations]),
            dtype=torch.long,
            device=self.device
        )
    
    def __len__(self) -> int:
        return len(self.demonstrations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[idx],
            self.action_masks[idx],
            self.actions[idx]
        )


class ActorNetwork(nn.Module):
    """
    Actor network for policy learning.
    
    MLP architecture with action masking applied before softmax.
    """
    
    def __init__(
        self,
        obs_dim: int = 50,
        action_dim: int = 14,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        activation: str = "relu"
    ):
        """
        Args:
            obs_dim: Observation space dimension (50)
            action_dim: Action space dimension (14)
            hidden_dim: Number of units in hidden layers
            num_hidden_layers: Number of hidden layers (2-3)
            activation: "relu" or "tanh"
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        prev_dim = obs_dim
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional action masking.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim) or (obs_dim,)
            action_mask: Binary mask of shape (batch_size, action_dim) or (action_dim,)
                        1 = valid action, 0 = invalid action
        
        Returns:
            logits: Raw logits of shape (batch_size, action_dim) or (action_dim,)
        """
        logits = self.net(obs)
        
        # Apply action masking
        if action_mask is not None:
            # Replace invalid logits with -1e9 (near -inf for softmax)
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        return logits


class BehavioralCloningTrainer:
    """Trainer for behavioral cloning with early stopping."""
    
    def __init__(
        self,
        actor_network: ActorNetwork,
        device: torch.device = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5
    ):
        """
        Args:
            actor_network: ActorNetwork instance
            device: torch device
            learning_rate: Optimizer learning rate
            weight_decay: L2 regularization
        """
        self.device = device or torch.device("cpu")
        self.actor = actor_network.to(self.device)
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: PyTorch DataLoader
        
        Returns:
            Average loss over the epoch
        """
        self.actor.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, action_masks, actions in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass with action masking
            logits = self.actor(states, action_mask=action_masks)
            
            # Cross-entropy loss (softmax is applied internally)
            loss = self.criterion(logits, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on a dataset.
        
        Args:
            dataloader: PyTorch DataLoader
        
        Returns:
            Dictionary with metrics: {"loss": float, "accuracy": float}
        """
        self.actor.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for states, action_masks, actions in dataloader:
                logits = self.actor(states, action_mask=action_masks)
                loss = self.criterion(logits, actions)
                
                total_loss += loss.item()
                
                # Compute accuracy
                pred_actions = torch.argmax(logits, dim=1)
                correct += (pred_actions == actions).sum().item()
                total += actions.shape[0]
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 100,
        target_accuracy: float = 0.75,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            num_epochs: Maximum number of epochs
            target_accuracy: Stop training once accuracy reaches this threshold
            patience: Early stopping patience (epochs without improvement)
            verbose: Print progress
        
        Returns:
            Dictionary with training history and final metrics
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Evaluate
            val_metrics = self.evaluate(val_dataloader)
            val_loss = val_metrics["loss"]
            val_accuracy = val_metrics["accuracy"]
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
            
            # Early stopping criteria
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Stop if target accuracy reached
            if val_accuracy >= target_accuracy:
                if verbose:
                    logger.info(
                        f"Target accuracy {target_accuracy:.2%} reached at epoch {epoch+1}"
                    )
                break
            
            # Stop if patience exceeded
            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break
        
        history["best_val_accuracy"] = best_val_accuracy
        history["final_epoch"] = epoch + 1
        
        return history


def create_behavioral_cloning_pipeline(
    demonstrations: List[Tuple[np.ndarray, np.ndarray, int]],
    val_split: float = 0.2,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    **actor_kwargs
) -> Tuple[ActorNetwork, BehavioralCloningTrainer, Dict]:
    """
    Convenience function to set up BC pre-training from scratch.
    
    Args:
        demonstrations: List of (state, action_mask, action) tuples
        val_split: Fraction of data to use for validation
        batch_size: Batch size for training
        device: torch device
        **actor_kwargs: Additional kwargs for ActorNetwork constructor
    
    Returns:
        Tuple of (actor_network, trainer, metrics_dict)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split data
    num_samples = len(demonstrations)
    num_val = int(num_samples * val_split)
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_demos = [demonstrations[i] for i in train_indices]
    val_demos = [demonstrations[i] for i in val_indices]
    
    # Create datasets
    train_dataset = BCDataset(train_demos, device=device)
    val_dataset = BCDataset(val_demos, device=device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create actor and trainer
    actor = ActorNetwork(**actor_kwargs)
    trainer = BehavioralCloningTrainer(actor, device=device)
    
    return actor, trainer, {"train_loader": train_loader, "val_loader": val_loader}


if __name__ == "__main__":
    # Quick example
    print("Testing BC pipeline...")
    
    # Create dummy demonstrations
    dummy_demos = []
    for _ in range(100):
        state = np.random.rand(50).astype(np.float32)
        action_mask = np.ones(14, dtype=np.float32)
        action_mask[np.random.choice(14, size=3, replace=False)] = 0
        action = np.random.choice(14)
        dummy_demos.append((state, action_mask, action))
    
    # Create pipeline
    actor, trainer, loaders = create_behavioral_cloning_pipeline(
        dummy_demos,
        batch_size=32,
        hidden_dim=128,
        num_hidden_layers=2
    )
    
    print(f"Actor network: {actor}")
    print(f"Training set size: {len(trainer.actor.parameters())}")
    
    # Quick train
    history = trainer.train(
        loaders["train_loader"],
        loaders["val_loader"],
        num_epochs=5,
        target_accuracy=0.5,
        verbose=True
    )
    
    print(f"Final accuracy: {history['best_val_accuracy']:.2%}")
