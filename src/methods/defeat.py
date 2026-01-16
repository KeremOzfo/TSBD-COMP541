"""
DEFEAT Backdoor Attack Implementation for Time Series

Based on: "DEFEAT: Deep Hidden Feature Backdoor Attacks by Imperceptible Perturbation and Latent Representation Constraints"
by Zhao et al.

Key Innovations:
1. Bilevel optimization: Alternating between trigger optimization and model poisoning
2. Feature hiding: Latent feature constraints force poisoned samples to have similar 
   intermediate representations as clean samples
3. Stealthiness budget: Constrains trigger magnitude using epsilon budget
4. Auxiliary classifiers: Train small classifiers on intermediate layers to measure
   feature similarity

Training Procedure (Algorithm 1):
1. Pre-train clean model F_θ on clean data
2. Train auxiliary logits h^(l) for selected layers
3. Initialize trigger function T_φ
4. Alternating optimization for R iterations:
   - Optimize φ: Generate stealthy triggers within budget ε
   - Optimize θ: Train model on clean + poisoned data with feature constraints
"""

import torch
import numpy as np
from src.methods.utils import cal_accuracy


class AuxiliaryLogit(torch.nn.Module):
    """
    Auxiliary classifier for intermediate layer features.
    
    h^(l)(z^(l)) = pool(z^(l)) @ ψ^(l) + ξ^(l)
    
    Args:
        in_features: Number of input features after pooling
        num_classes: Number of output classes
    """
    def __init__(self, in_features, num_classes):
        super(AuxiliaryLogit, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(in_features, num_classes)
    
    def forward(self, z):
        """
        Args:
            z: Layer features [B, T, C] or [B, C, T] for time series
        Returns:
            Logits [B, num_classes]
        """
        # Handle both [B, T, C] and [B, C, T] formats
        if z.dim() == 3:
            # If last dim is larger, likely [B, C, T], transpose to [B, T, C]
            if z.size(1) < z.size(2):
                z = z.transpose(1, 2)  # [B, T, C]
        # Global average pooling over time dimension (dim=1)
        pooled = z.mean(dim=1)  # [B, C]
        return self.fc(pooled)


def train_auxiliary_logits(model, train_loader, args, selected_layers=None):
    """
    Train auxiliary classifiers on intermediate layer features.
    
    Args:
        model: Pre-trained clean classifier F_θ (frozen)
        train_loader: Clean training data
        args: Configuration
        selected_layers: List of layer names to attach auxiliary classifiers
                        (e.g., last N layers for feature hiding)
    
    Returns:
        aux_logits: Dict mapping layer names to trained AuxiliaryLogit modules
    """
    model.eval()
    device = args.device
    num_classes = args.num_class
    
    # Auto-select last layers if not specified
    if selected_layers is None:
        # Get all intermediate layers from model
        all_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear, torch.nn.TransformerEncoderLayer)):
                all_layers.append(name)
        # Select last 3 layers by default
        selected_layers = all_layers[-3:] if len(all_layers) >= 3 else all_layers
    
    print(f"Training auxiliary logits for layers: {selected_layers}")
    
    # Register hooks to extract features
    features = {}
    hooks = []
    
    def get_hook(layer_name):
        def hook(module, input, output):
            features[layer_name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        if name in selected_layers:
            hooks.append(module.register_forward_hook(get_hook(name)))
    
    # Determine feature dimensions by doing a forward pass
    # Apply the same pooling logic to get actual feature dimension
    with torch.no_grad():
        for batch_x, label, padding_mask in train_loader:
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)
            _ = model(batch_x, padding_mask, None, None)
            break
    
    # Create auxiliary classifiers based on actual pooled feature dimensions
    aux_logits = {}
    for layer_name in selected_layers:
        if layer_name in features:
            feat = features[layer_name]
            print(f"  Layer {layer_name}: original feature shape={feat.shape}")
            
            # Create a temporary auxiliary classifier to test dimensions
            # Try different possible dimensions
            feat_copy = feat.clone()
            if feat_copy.dim() == 3:
                # If last dim is larger, likely [B, C, T], transpose to [B, T, C]
                if feat_copy.size(1) < feat_copy.size(2):
                    feat_copy = feat_copy.transpose(1, 2)  # [B, T, C]
                # Pool over time dimension to get [B, C]
                pooled = feat_copy.mean(dim=1)
                in_features = pooled.size(1)  # C dimension
            elif feat_copy.dim() == 2:
                # Already [B, C] format
                pooled = feat_copy
                in_features = feat_copy.size(1)
            else:
                # Fallback: use last dimension
                pooled = feat_copy
                in_features = feat_copy.shape[-1]
            
            print(f"  Layer {layer_name}: pooled shape={pooled.shape}, in_features={in_features}, num_classes={num_classes}")
            
            # Create auxiliary classifier
            aux_classifier = AuxiliaryLogit(in_features, num_classes).to(device)
            
            # Validate it works with the actual feature
            with torch.no_grad():
                try:
                    test_output = aux_classifier(feat)
                    print(f"  Layer {layer_name}: validation successful, output shape={test_output.shape}")
                except RuntimeError as e:
                    print(f"  Layer {layer_name}: validation failed - {e}")
                    raise
            
            aux_logits[layer_name] = aux_classifier
    
    # Train each auxiliary classifier
    epochs = getattr(args, 'aux_logit_epochs', 10)
    lr = getattr(args, 'aux_logit_lr', 0.001)
    
    for layer_name, aux_classifier in aux_logits.items():
        optimizer = torch.optim.Adam(aux_classifier.parameters(), lr=lr)
        criterion = args.criterion
        
        print(f"Training auxiliary logit for layer: {layer_name}")
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, label, padding_mask in train_loader:
                batch_x = batch_x.float().to(device)
                padding_mask = padding_mask.float().to(device)
                label = label.to(device)
                
                # Extract features
                with torch.no_grad():
                    _ = model(batch_x, padding_mask, None, None)
                
                feat = features[layer_name]
                
                # Train auxiliary classifier
                optimizer.zero_grad()
                logits = aux_classifier(feat)
                loss = criterion(logits, label.long().squeeze(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == label.long().squeeze(-1)).sum().item()
                total += label.size(0)
            
            acc = correct / total
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return aux_logits


def latent_feature_constraint_loss(model, aux_logits, x_clean, x_poisoned, padding_mask, layer_weights=None):
    """
    Compute latent feature constraint loss L_{lf}.
    
    L_{lf} = mean(Σ λ(l) |h^(l)(z^(l)_clean) - h^(l)(z^(l)_poisoned)|)
    
    Measures difference between latent features of clean and poisoned samples
    to enforce feature hiding.
    
    Args:
        model: Classifier F_θ
        aux_logits: Dict of auxiliary classifiers
        x_clean: Clean inputs
        x_poisoned: Poisoned inputs
        padding_mask: Padding mask
        layer_weights: Dict mapping layer names to weights λ(l)
    
    Returns:
        L_{lf}: Scalar loss
    """
    if layer_weights is None:
        # Equal weights across all layers
        layer_weights = {name: 1.0 / len(aux_logits) for name in aux_logits.keys()}
    
    # Extract features for both clean and poisoned
    features_clean = {}
    features_poisoned = {}
    hooks = []
    
    def get_hook(layer_name, features_dict):
        def hook(module, input, output):
            features_dict[layer_name] = output
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if name in aux_logits:
            hooks.append(module.register_forward_hook(get_hook(name, features_clean)))
    
    # Forward pass clean
    _ = model(x_clean, padding_mask, None, None)
    
    # Remove and re-register hooks for poisoned
    for hook in hooks:
        hook.remove()
    hooks = []
    
    for name, module in model.named_modules():
        if name in aux_logits:
            hooks.append(module.register_forward_hook(get_hook(name, features_poisoned)))
    
    # Forward pass poisoned
    _ = model(x_poisoned, padding_mask, None, None)
    
    # Compute constraint loss
    total_loss = 0.0
    for layer_name in aux_logits:
        if layer_name in features_clean and layer_name in features_poisoned:
            z_clean = features_clean[layer_name]
            z_poisoned = features_poisoned[layer_name]
            
            # Get logits from auxiliary classifiers
            h_clean = aux_logits[layer_name](z_clean)
            h_poisoned = aux_logits[layer_name](z_poisoned)
            
            # L1 distance weighted by layer weight
            diff = torch.abs(h_clean - h_poisoned).mean()
            total_loss += layer_weights[layer_name] * diff
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_loss


def epoch_defeat(
    bd_model,           # Trigger generator T_φ
    surr_model,         # Classifier F_θ
    aux_logits,         # Auxiliary classifiers h^(l)
    loader,             # Data loader
    args,
    opt_trig=None,      # Optimizer for trigger φ
    opt_class=None,     # Optimizer for classifier θ
    train=True,
    iteration=0         # Current iteration in alternating optimization
):
    """
    Single epoch of DEFEAT training with bilevel optimization.
    
    Equation (3):
    min_θ β_1 L_{clean}(θ) + β_2 L_{adv}(θ)
    s.t. φ(θ) = argmin_φ Σ[L_{ce}(F_θ(T_φ(x)), y_t) + max(||T_φ(x) - x||_2 - ε, 0)]
    
    Where:
    L_{adv}(θ) = Σ[L_{ce}(F_θ(T_φ(x)), y_t) + L_{lf}(x, T_φ(x), F_θ)]
    
    Args:
        bd_model: Trigger generator
        surr_model: Classifier
        aux_logits: Dict of auxiliary classifiers for feature constraints
        loader: Data loader
        args: Config with:
            - beta_1: Weight for clean loss (default: 1.0)
            - beta_2: Weight for adversarial loss (default: 0.1)
            - epsilon_budget: Stealthiness budget (default: 0.1 for time series)
            - poisoning_ratio: Fraction of data to poison (default: 0.1)
        opt_trig: Trigger optimizer
        opt_class: Classifier optimizer
        train: Training mode
        iteration: Current alternating optimization iteration
    
    Returns:
        Metrics dict
    """
    if train:
        bd_model.train()
        surr_model.train()
    else:
        bd_model.eval()
        surr_model.eval()
    
    device = args.device
    target_label = args.target_label
    num_classes = args.num_class
    bd_type = getattr(args, 'bd_type', 'all2one')
    
    # DEFEAT hyperparameters
    beta_1 = getattr(args, 'beta_1', 1.0)
    beta_2 = getattr(args, 'beta_2', 0.1)
    epsilon_budget = getattr(args, 'epsilon_budget', 0.1)
    poisoning_ratio = getattr(args, 'poisoning_ratio', 0.1)
    
    # Loss tracking
    losses = {
        'total': [], 'clean': [], 'backdoor_ce': [], 
        'latent_feature': [], 'stealthiness': []
    }
    
    all_preds, bd_preds = [], []
    trues, bd_trues = [], []
    
    for batch_idx, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(device)
        padding_mask = padding_mask.float().to(device)
        label = label.to(device)
        batch_size = batch_x.size(0)
        
        # ============== PHASE 1: OPTIMIZE TRIGGER φ ==============
        if train and opt_trig is not None:
            # Freeze classifier
            for p in surr_model.parameters():
                p.requires_grad = False
            for aux in aux_logits.values():
                for p in aux.parameters():
                    p.requires_grad = False
            
            opt_trig.zero_grad()
            
            # Sample poisoning subset
            num_poison = max(1, int(batch_size * poisoning_ratio))
            poison_indices = torch.randperm(batch_size)[:num_poison]
            
            x_poison = batch_x[poison_indices]
            mask_poison = padding_mask[poison_indices]
            
            # Generate target labels
            if bd_type == 'all2all':
                y_target = torch.randint(0, num_classes, (num_poison,), device=device)
            else:
                y_target = torch.ones(num_poison, dtype=torch.long, device=device) * target_label
            
            # Generate trigger: T_φ(x) = x + φ (universal pattern)
            # Trigger model returns (unclipped, clipped)
            trigger_unclipped, trigger_clipped = bd_model(x_poison, mask_poison, None, None, y_target)
            x_triggered = x_poison + trigger_clipped
            
            # Backdoor CE loss
            pred_bd = surr_model(x_triggered, mask_poison, None, None)
            loss_bd_ce = args.criterion(pred_bd, y_target.long().squeeze(-1))
            
            # Stealthiness constraint: max(||T_φ(x) - x||_2 - ε, 0)
            trigger_magnitude = torch.norm(trigger_clipped.reshape(num_poison, -1), p=2, dim=1)
            loss_stealth = torch.clamp(trigger_magnitude - epsilon_budget, min=0).mean()
            
            # Trigger loss
            loss_trigger = loss_bd_ce + loss_stealth
            loss_trigger.backward()
            opt_trig.step()
            
            # Unfreeze classifier
            for p in surr_model.parameters():
                p.requires_grad = True
        
        # ============== PHASE 2: OPTIMIZE CLASSIFIER θ ==============
        if train and opt_class is not None:
            # Freeze trigger generator
            for p in bd_model.parameters():
                p.requires_grad = False
            
            opt_class.zero_grad()
            
            # Sample poisoning subset
            num_poison = max(1, int(batch_size * poisoning_ratio))
            poison_indices = torch.randperm(batch_size)[:num_poison]
            clean_indices = torch.tensor([i for i in range(batch_size) if i not in poison_indices])
            
            # Clean loss L_{clean}
            if len(clean_indices) > 0:
                x_clean = batch_x[clean_indices]
                y_clean = label[clean_indices]
                mask_clean = padding_mask[clean_indices]
                
                pred_clean = surr_model(x_clean, mask_clean, None, None)
                loss_clean = args.criterion(pred_clean, y_clean.long().squeeze(-1))
            else:
                loss_clean = torch.tensor(0.0, device=device)
            
            # Adversarial loss L_{adv}
            if len(poison_indices) > 0:
                x_poison = batch_x[poison_indices]
                mask_poison = padding_mask[poison_indices]
                
                # Generate target labels
                if bd_type == 'all2all':
                    y_target = torch.randint(0, num_classes, (num_poison,), device=device)
                else:
                    y_target = torch.ones(num_poison, dtype=torch.long, device=device) * target_label
                
                # Generate triggered samples
                with torch.no_grad():
                    _, trigger_clipped = bd_model(x_poison, mask_poison, None, None, y_target)
                x_triggered = x_poison + trigger_clipped
                
                # Backdoor CE loss
                pred_bd = surr_model(x_triggered, mask_poison, None, None)
                loss_bd_ce = args.criterion(pred_bd, y_target.long().squeeze(-1))
                
                # Latent feature constraint L_{lf}
                loss_lf = latent_feature_constraint_loss(
                    surr_model, aux_logits, x_poison, x_triggered, mask_poison
                )
                
                # Adversarial loss
                loss_adv = loss_bd_ce + loss_lf
            else:
                loss_adv = torch.tensor(0.0, device=device)
                loss_bd_ce = torch.tensor(0.0, device=device)
                loss_lf = torch.tensor(0.0, device=device)
            
            # Total classifier loss (Equation 3)
            loss_total = beta_1 * loss_clean + beta_2 * loss_adv
            loss_total.backward()
            opt_class.step()
            
            # Unfreeze trigger
            for p in bd_model.parameters():
                p.requires_grad = True
            
            # Track losses
            losses['total'].append(loss_total.item())
            losses['clean'].append(loss_clean.item())
            losses['backdoor_ce'].append(loss_bd_ce.item())
            losses['latent_feature'].append(loss_lf.item())
        
        # ============== EVALUATION ==============
        with torch.no_grad():
            # Clean accuracy
            pred_clean = surr_model(batch_x, padding_mask, None, None)
            all_preds.append(pred_clean.argmax(dim=1).cpu())
            trues.append(label.cpu())
            
            # Backdoor accuracy
            num_poison = max(1, int(batch_size * poisoning_ratio))
            poison_indices = torch.randperm(batch_size)[:num_poison]
            
            if len(poison_indices) > 0:
                x_bd = batch_x[poison_indices]
                mask_bd = padding_mask[poison_indices]
                
                if bd_type == 'all2all':
                    y_target = torch.randint(0, num_classes, (num_poison,), device=device)
                else:
                    y_target = torch.ones(num_poison, dtype=torch.long, device=device) * target_label
                
                _, trigger_clip = bd_model(x_bd, mask_bd, None, None, y_target)
                x_triggered = x_bd + trigger_clip
                
                pred_bd = surr_model(x_triggered, mask_bd, None, None)
                bd_preds.append(pred_bd.argmax(dim=1).cpu())
                bd_trues.append(y_target.cpu())
    
    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    clean_acc = cal_accuracy(all_preds, trues)
    
    if len(bd_preds) > 0:
        bd_preds = torch.cat(bd_preds, dim=0).numpy()
        bd_trues = torch.cat(bd_trues, dim=0).numpy()
        bd_acc = cal_accuracy(bd_preds, bd_trues)
    else:
        bd_acc = 0.0
    
    # Average losses
    avg_losses = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in losses.items()}
    
    return avg_losses['total'], avg_losses, clean_acc, bd_acc


def train_defeat(trigger_model, surrogate_model, train_loader, args, train_data=None):
    """
    Full DEFEAT training procedure (Algorithm 1).
    
    1. Pre-train clean model
    2. Train auxiliary logits
    3. Alternating optimization for R iterations
    
    Args:
        trigger_model: Trigger generator T_φ
        surrogate_model: Classifier F_θ
        train_loader: Training data
        args: Configuration
        train_data: Training dataset
    
    Returns:
        results: Dict with training history
    """
    device = args.device
    
    # Step 1: Pre-train clean model (if not already done)
    print("="*70)
    print("DEFEAT STEP 1: Pre-training clean model")
    print("="*70)
    
    clean_epochs = getattr(args, 'clean_pretrain_epochs', 50)
    clean_lr = getattr(args, 'clean_pretrain_lr', 0.01)
    optimizer_clean = torch.optim.SGD(surrogate_model.parameters(), lr=clean_lr, momentum=0.9)
    
    for epoch in range(clean_epochs):
        surrogate_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, label, padding_mask in train_loader:
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)
            label = label.to(device)
            
            optimizer_clean.zero_grad()
            pred = surrogate_model(batch_x, padding_mask, None, None)
            loss = args.criterion(pred, label.long().squeeze(-1))
            loss.backward()
            optimizer_clean.step()
            
            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == label).sum().item()
            total += label.size(0)
        
        acc = correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Clean Pretrain] Epoch {epoch+1}/{clean_epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}")
    
    # Step 2: Train auxiliary logits
    print("\n" + "="*70)
    print("DEFEAT STEP 2: Training auxiliary logits")
    print("="*70)
    
    aux_logits = train_auxiliary_logits(surrogate_model, train_loader, args)
    
    # Step 3: Alternating optimization
    print("\n" + "="*70)
    print("DEFEAT STEP 3: Alternating optimization")
    print("="*70)
    
    R = getattr(args, 'defeat_iterations', 20)
    finetune_lr = getattr(args, 'finetune_lr', 0.001)
    trigger_lr = getattr(args, 'trigger_lr', 0.001)
    
    optimizer_trigger = torch.optim.Adam(trigger_model.parameters(), lr=trigger_lr)
    optimizer_classifier = torch.optim.SGD(surrogate_model.parameters(), lr=finetune_lr, momentum=0.9)
    
    history = {'clean_acc': [], 'bd_acc': [], 'loss': []}
    
    for iteration in range(R):
        loss, loss_dict, clean_acc, bd_acc = epoch_defeat(
            trigger_model, surrogate_model, aux_logits, train_loader, args,
            opt_trig=optimizer_trigger,
            opt_class=optimizer_classifier,
            train=True,
            iteration=iteration
        )
        
        history['clean_acc'].append(clean_acc)
        history['bd_acc'].append(bd_acc)
        history['loss'].append(loss)
        
        print(f"[DEFEAT Iter {iteration+1}/{R}] "
              f"Loss={loss:.4f}, Clean Acc={clean_acc:.4f}, BD Acc (ASR)={bd_acc:.4f}")
        print(f"  └─ L_clean={loss_dict['clean']:.4f}, L_bd_ce={loss_dict['backdoor_ce']:.4f}, "
              f"L_lf={loss_dict['latent_feature']:.4f}")
    
    print("\n" + "="*70)
    print("DEFEAT Training Completed")
    print("="*70)
    
    return {
        'history': history,
        'aux_logits': aux_logits,
        'final_clean_acc': clean_acc,
        'final_bd_acc': bd_acc
    }
