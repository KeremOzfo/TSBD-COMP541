"""List of methods for trainning dynamic trigger model"""

import torch
import torch.nn as nn
import numpy as np
from utils.eval_utils import cal_accuracy
from utils.loss_utils import reg_loss


def _frequency_basis(seq_len: int, channels: int, device: torch.device, cache: dict, t: int) -> torch.Tensor:
    """Return a real-valued temporal basis whose spectrum has energy only at freq bin t."""
    if t in cache:
        return cache[t]

    freq_len = seq_len // 2 + 1
    freq_mask = torch.zeros(freq_len, dtype=torch.cfloat, device=device)
    # place symmetric energy at bin t (0-indexed); irfft handles the conjugate symmetry
    if t < freq_len:
        freq_mask[t] = torch.tensor(1.0, dtype=torch.cfloat, device=device)
    basis = torch.fft.irfft(freq_mask, n=seq_len).real  # (seq_len,)
    basis = basis.view(1, seq_len, 1).expand(1, seq_len, channels)  # broadcast across channels
    cache[t] = basis
    return basis


def estimate_frequency_heatmap(model, loader, args, max_batches: int | None = None) -> torch.Tensor:
    """Estimate model sensitivity heatmap S_{t,m} via DFT-based perturbations"""
    model.eval()
    device = args.device
    seq_len = args.seq_len
    perturb_scale = getattr(args, "freq_lambda", 0.05)
    max_bins = getattr(args, "freq_max_bins", seq_len)
    freq_len = seq_len // 2 + 1

    heatmap = None
    basis_cache: dict[int, torch.Tensor] = {}
    batches_used = 0

    with torch.no_grad():
        for b_idx, (batch_x, label, padding_mask) in enumerate(loader):
            if max_batches is not None and batches_used >= max_batches:
                break
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)
            label = label.to(device)

            if heatmap is None:
                channels = batch_x.shape[-1]
                heatmap = torch.zeros(freq_len, channels, device=device)

            # base loss
            base_pred = model(batch_x, padding_mask, None, None)
            base_loss = args.criterion(base_pred, label.long().squeeze(-1))

            for t in range(min(freq_len, max_bins)):
                basis = _frequency_basis(seq_len, batch_x.shape[-1], device, basis_cache, t)
                perturbed = batch_x + perturb_scale * basis
                pert_pred = model(perturbed, padding_mask, None, None)
                pert_loss = args.criterion(pert_pred, label.long().squeeze(-1))
                delta = (pert_loss - base_loss).detach()
                heatmap[t] += delta

            batches_used += 1

    if heatmap is None:
        # empty loader fallback
        heatmap = torch.zeros(seq_len // 2 + 1, getattr(args, "enc_in", 1), device=device)
    else:
        heatmap = heatmap / max(1, batches_used)

    return heatmap.detach()


def epoch_frequency_guided(
    bd_model,
    surr_model,
    loader,
    args,
    opt=None,
    train: bool = True,
):
    """Frequency-domain guided trigger training"""

    # Hyperparameters
    alpha = getattr(args, "freq_alpha", 1.0)  # weight for L_Freq
    beta = getattr(args, "freq_beta", 1e-3)   # weight for regularization
    heatmap_batches = getattr(args, "freq_heatmap_batches", 1)

    # Pre-compute frequency sensitivity heatmap once
    freq_target = estimate_frequency_heatmap(surr_model, loader, args, max_batches=heatmap_batches)
    freq_target = freq_target.detach()  # (freq_len, channels)

    total_loss = []
    loss_dict = {"CE_bd": [], "L_freq": [], "L_reg": []}
    bd_preds = []
    trues = []
    bd_labels_all = []

    if train:
        bd_model.train()
        surr_model.train()
    else:
        bd_model.eval()
        surr_model.eval()

    freq_len = freq_target.shape[0]

    for batch_x, label, padding_mask in loader:
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)

        # target labels (all-to-one default)
        bd_labels = torch.ones_like(label).to(args.device) * args.target_label

        # mask if attacking only non-target
        if getattr(args, "attack_only_nontarget", False):
            attack_mask = (label != args.target_label).float().to(args.device)
        else:
            attack_mask = torch.ones_like(label).float().to(args.device)
        attack_mask = attack_mask.unsqueeze(-1).expand_as(batch_x)

        # Generate trigger
        trigger, trigger_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
        triggered_inputs = batch_x + trigger_clip * attack_mask

        # Classification loss on triggered inputs
        pred_bd = surr_model(triggered_inputs, padding_mask, None, None)
        loss_ce = args.criterion(pred_bd, bd_labels.long().squeeze(-1))

        # Frequency alignment loss (Eq. 3): compare spectrum magnitude with sensitivity heatmap
        trigger_freq = torch.fft.rfft(trigger_clip, dim=1)
        trigger_mag = torch.abs(trigger_freq)
        target_mag = freq_target[: trigger_mag.shape[1]].to(args.device)
        loss_freq = torch.mean((trigger_mag - target_mag) ** 2)

        # Regularization in both frequency and temporal domains (Eq. 4)
        loss_reg = torch.mean(trigger_mag ** 2) + torch.mean(trigger_clip ** 2)

        loss = loss_ce + alpha * loss_freq + beta * loss_reg

        if opt is not None and train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss.append(loss.item())
        loss_dict["CE_bd"].append(loss_ce.item())
        loss_dict["L_freq"].append(loss_freq.item())
        loss_dict["L_reg"].append(loss_reg.item())

        bd_preds.append(pred_bd)
        trues.append(label)
        bd_labels_all.append(bd_labels)

    if not total_loss:
        return 0.0, loss_dict, 0.0, 0.0

    total_loss = float(np.average(total_loss))
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels_all = torch.cat(bd_labels_all, 0)

    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds, dim=1), dim=1).cpu().numpy()
    trues_np = trues.flatten().cpu().numpy()
    bd_labels_np = bd_labels_all.flatten().cpu().numpy()

    # Clean accuracy is not directly optimized here; report ASR as main metric
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels_np)
    clean_accuracy = cal_accuracy(bd_predictions, trues_np)

    return total_loss, loss_dict, clean_accuracy, bd_accuracy

def epoch_vanilla_training():
  """ Vanilla trigger training framework *** cite related work ***

  """
  return


def epoch_marksma(
    bd_model,
    surr_model,
    loader,
    args,
    opt_cls=None,
    opt_trig=None,
    train: bool = True,
    step_offset: int = 0,
):
    """
    This implements the constrained objective:
        - Update classifier (θ): CE clean + α * CE backdoor
        - Update trigger (ξ): CE backdoor - β * ||g||_2
        - Class-conditional targets sampled per-sample (c ≠ y) # dynamic case, for now fixed target only
        - Trigger updated less frequently (controlled by args.marksman_update_T / marksman_k)

    Args:
        bd_model: trigger generator g(c, x)
        surr_model: classifier f_θ
        loader: dataloader yielding (x, y, padding_mask)
        args: config with fields {criterion, device, num_class/numb_class, marksman_alpha, marksman_beta,
              marksman_update_T|marksman_k}
        opt_cls: optimizer for classifier
        opt_trig: optimizer for trigger generator
        train: toggle grad/step
        step_offset: global step offset to align multi-epoch scheduling
    Returns:
        total_loss (float), loss_dict, clean_accuracy, attack_success_rate
    """

    alpha = getattr(args, "marksman_alpha", 1.0)
    beta = getattr(args, "marksman_beta", 0.0)
    update_T = getattr(args, "marksman_update_T", getattr(args, "marksman_k", 1))
    num_classes = getattr(args, "numb_class", getattr(args, "num_class", None))
    if num_classes is None:
        raise ValueError("args.num_class (or args.numb_class) is required for Marksman training")

    if train:
        bd_model.train()
        surr_model.train()
    else:
        bd_model.eval()
        surr_model.eval()

    total_loss = []
    loss_dict = {"CE_clean": [], "CE_bd": [], "CE_trig": []}
    clean_preds, bd_preds = [], []
    clean_labels, bd_labels_all = [], []

    for step, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)

        # Sample class-conditional targets different from true label. All-to-all case by default
        #rand = torch.randint(0, num_classes - 1, label.shape, device=args.device)
        #bd_labels = (rand + label) % num_classes
        bd_labels = torch.ones_like(label).to(args.device) * args.target_label

        # -------- Classifier update (clean + backdoor) --------
        if train and opt_cls is not None:
            opt_cls.zero_grad()
        if train and opt_trig is not None:
            opt_trig.zero_grad()

        with torch.no_grad():  # keep generator fixed for classifier step
            _, trigger_clip_det = bd_model(batch_x, padding_mask, None, None, bd_labels)
        triggered_inputs = batch_x + trigger_clip_det

        pred_clean = surr_model(batch_x, padding_mask, None, None)
        pred_bd = surr_model(triggered_inputs, padding_mask, None, None)

        loss_clean = args.criterion(pred_clean, label.long().squeeze(-1))
        loss_bd = args.criterion(pred_bd, bd_labels.long().squeeze(-1))
        loss_cls = loss_clean + alpha * loss_bd

        if train and opt_cls is not None:
            loss_cls.backward()
            opt_cls.step()

        # -------- Trigger update (backdoor CE - beta * ||g||_2) --------
        trig_loss_val = None
        if train and opt_trig is not None and ((step_offset + step) % update_T == 0):
            # Freeze classifier parameters during trigger update
            requires_backup = [p.requires_grad for p in surr_model.parameters()]
            for p in surr_model.parameters():
                p.requires_grad = False

            opt_trig.zero_grad()
            trigger, trigger_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
            attacked = batch_x + trigger_clip
            pred_trig = surr_model(attacked, padding_mask, None, None)
            loss_trig = args.criterion(pred_trig, bd_labels.long().squeeze(-1))
            loss_trig = loss_trig - beta * torch.mean(trigger ** 2)
            loss_trig.backward()
            opt_trig.step()
            trig_loss_val = loss_trig.detach().item()

            # Restore classifier requires_grad
            for p, req in zip(surr_model.parameters(), requires_backup):
                p.requires_grad = req

        total_loss.append(loss_cls.item())
        loss_dict["CE_clean"].append(loss_clean.item())
        loss_dict["CE_bd"].append(loss_bd.item())
        if trig_loss_val is not None:
            loss_dict["CE_trig"].append(trig_loss_val)

        clean_preds.append(pred_clean.detach())
        bd_preds.append(pred_bd.detach())
        clean_labels.append(label.detach())
        bd_labels_all.append(bd_labels.detach())

    if not total_loss:
        return 0.0, loss_dict, 0.0, 0.0

    total_loss = float(np.average(total_loss))
    clean_preds = torch.cat(clean_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    clean_labels = torch.cat(clean_labels, 0)
    bd_labels_all = torch.cat(bd_labels_all, 0)

    clean_predictions = torch.argmax(torch.nn.functional.softmax(clean_preds, dim=1), dim=1).cpu().numpy()
    clean_accuracy = cal_accuracy(clean_predictions, clean_labels.flatten().cpu().numpy())

    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds, dim=1), dim=1).cpu().numpy()
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels_all.flatten().cpu().numpy())

    return total_loss, loss_dict, clean_accuracy, bd_accuracy

def epoch_diversity(bd_model,surr_model, loader1, args, loader2=None, opt=None,opt2=None,train=True): 
    """


    Parameters:


    Returns:

    """
  
    total_loss = []
    all_preds = []
    bd_preds = []
    trues = []
    bds = []
    bd_label = args.target_label
    loss_dict = {'CE_c':[],'CE_bd':[],'reg':[]}
    ratio = args.poisoning_ratio_train
    criterion_div = nn.MSELoss(reduction="none")

    # to make the zipped loop consistent with or without diversity loss
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)

    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    for i, (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(range(len(loader1)), loader1, loader2):
            loss_div = 0.0 # for consistency with optional diversity loss
            loss_clean = torch.tensor([0.0])
            loss_bd = torch.tensor([0.0]) # for consistency with optional cross losses and logging.
            bd_model.zero_grad()
            surr_model.zero_grad()
            #### Fetch clean data
            batch_x = batch_x.float().to(args.device)
            #### Fetch mask (for forecast task)
            padding_mask = padding_mask.float().to(args.device)
            #### Fetch labels
            label = label.to(args.device)
            #### Generate backdoor labels ####### so far we focus on fixed target scenario
            bd_labels = torch.ones_like(label).to(args.device) * bd_label ## comes from argument
            if batch_x2 is not None:
                batch_x2 = batch_x2.float().to(args.device)
                padding_mask2 = padding_mask2.float().to(args.device)
                label2 = label2.to(args.device)
                bd_labels2 = torch.ones_like(label2).to(args.device) * bd_label ## comes from argument
        
            trigger, trigger_clip = bd_model(batch_x, padding_mask,None,None)
            if batch_x2 is not None and args.div_reg:
                
                trigger2, trigger_clip2 = bd_model(batch_x2, padding_mask2, None, None)

                ### DIVERGENCE LOSS CALCULATION
                input_distances = criterion_div(batch_x, batch_x2)
                input_distances = torch.mean(input_distances, dim=(1, 2))
                input_distances = torch.sqrt(input_distances)

                ### TODO: do we use trigger or trigger_clip here?
                trigger_distances = criterion_div(trigger, trigger2)
                trigger_distances = torch.mean(trigger_distances, dim=(1, 2))
                trigger_distances = torch.sqrt(trigger_distances)

                loss_div = input_distances / (trigger_distances + 1e-6) # second value is the epsilon, arbitrary for now
                loss_div = torch.mean(loss_div) * args.div_reg

            
            mask = (label != bd_label).float().to(args.device) if args.attack_only_nontarget else torch.ones_like(label).float().to(args.device)
            mask = mask.unsqueeze(-1).expand(-1,trigger_clip.shape[-2],trigger_clip.shape[-1])
            clean_pred = surr_model(batch_x, padding_mask,None,None)
            bd_pred = surr_model(batch_x + trigger_clip * mask, padding_mask,None,None)
            if batch_x2 is not None and args.div_reg:
                # cross loss from input aware paper (coupled with diversity loss from the same work)
                bs = batch_x.shape[0]
                num_bd = int(0.5 * bs) # args.p_attack, values taken from the best result from input-aware paper
                num_cross = int(0.1 * bs) # args.p_cross
                bd_inputs = (batch_x + trigger_clip * mask)[:num_bd]
                cross_inputs = (batch_x + trigger_clip2 * mask)[num_bd : num_bd + num_cross]
                total_inputs = torch.cat((bd_inputs, cross_inputs, batch_x[num_bd + num_cross:])).to(args.device)
                total_targets = torch.cat((bd_labels[:num_bd], label[num_bd:])).to(args.device)
                total_pred = surr_model(total_inputs, padding_mask, None, None)
                total_cross_loss = args.criterion(total_pred, total_targets.long().squeeze(-1))       
            else:
                loss_clean = args.criterion(clean_pred, label.long().squeeze(-1))
                loss_bd = args.criterion(bd_pred, bd_labels.long().squeeze(-1))
            loss_reg = reg_loss(batch_x,trigger,trigger_clip,args) ### We can use regularizer loss as well
            if loss_reg is None:
                loss_reg = torch.zeros_like(total_cross_loss) if batch_x2 is not None and args.div_reg else torch.zeros_like(loss_bd) 
            if batch_x2 is not None and args.div_reg:
                loss = total_cross_loss + loss_reg + loss_div
            else:
                loss = loss_clean + loss_bd + loss_reg + loss_div
            loss_dict['CE_c'].append(loss_clean.item())
            loss_dict['CE_bd'].append(loss_bd.item())
            loss_dict['reg'].append(loss_reg.item())
            total_loss.append(loss.item())
            all_preds.append(clean_pred)
            bd_preds.append(bd_pred)
            trues.append(label)
            bds.append(bd_labels)

            if opt is not None:
                loss.backward()
                opt.step()
            if opt2 is not None:
                opt2.step()
    total_loss = np.average(total_loss)
    all_preds = torch.cat(all_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels = torch.cat(bds, 0)
    probs = torch.nn.functional.softmax(
        all_preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds), dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels.flatten().cpu().numpy())
    return total_loss,loss_dict, accuracy,bd_accuracy


"""
Input-Aware Backdoor Attack Training Module

This module implements the strictly input-aware backdoor attack as described in the diversity.tex paper.
The attack satisfies:
1. Diversity: triggers vary significantly across different inputs
2. Nonreusability: a trigger generated for one input cannot activate the backdoor on another input

The training uses three modes:
- Clean mode: model learns to classify clean inputs correctly
- Attack mode: model learns to classify triggered inputs as target class
- Cross-trigger mode: model learns to classify inputs with "wrong" triggers (from other samples) correctly

"""

import math
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static checking only
    from utils.model_ops import pull_model as pull_model  # type: ignore
else:
    # pull_model utility may not exist in this repo; provide a fallback copier
    try:  # pragma: no cover - optional dependency
        from utils.model_ops import pull_model  # type: ignore
    except Exception:  # pragma: no cover - fallback copy
        def pull_model(dst, src):
            dst.load_state_dict(src.state_dict())


def diversity_loss(x1, x2, g_x1, g_x2, epsilon=1e-6):
    """
    Compute the diversity enforcement regularization loss.
    
    L_div = ||x - x'|| / ||g(x) - g(x')||
    
    This loss encourages the trigger generator to produce diverse triggers
    for different inputs, preventing saturation to a uniform trigger.
    
    Args:
        x1: First batch of clean inputs (B x T x C or B x C x T)
        x2: Second batch of clean inputs (different from x1)
        g_x1: Triggers generated for x1
        g_x2: Triggers generated for x2
        epsilon: Small constant for numerical stability
    
    Returns:
        Scalar diversity loss value
    """
    # Compute input distances: ||x - x'||
    input_diff = x1 - x2
    input_distances = torch.sqrt(torch.mean(input_diff ** 2, dim=(1, 2)) + epsilon)
    
    # Compute trigger distances: ||g(x) - g(x')||
    trigger_diff = g_x1 - g_x2
    trigger_distances = torch.sqrt(torch.mean(trigger_diff ** 2, dim=(1, 2)) + epsilon)
    
    # L_div = ||x - x'|| / ||g(x) - g(x')||
    # We want to minimize this, which means maximizing trigger diversity
    loss_div = input_distances / (trigger_distances + epsilon)
    
    return torch.mean(loss_div)


def apply_trigger(x, trigger, mask=None):
    """
    Apply trigger to input: B(x, t) = x + trigger (additive model)
    For the paper's formulation: B(x,t) = x ⊙ (1-m) + p ⊙ m
    Here we use additive triggers where the trigger already incorporates the mask.
    
    Args:
        x: Clean input
        trigger: Generated trigger (already clipped)
        mask: Optional mask for selective triggering
    
    Returns:
        Triggered input
    """
    if mask is not None:
        return x + trigger * mask
    return x + trigger


def epoch_marksman_input_aware(
    bd_model,           # Current trigger generator g
    bd_model_prev,      # Previous trigger generator (for stable classifier training)
    surr_model,         # Surrogate classifier f
    loader1,            # Primary data loader
    args,
    loader2=None,       # Secondary data loader for diversity/cross-trigger
    opt_trig=None,      # Optimizer for trigger generator
    opt_class=None,     # Optimizer for classifier
    train=True
):
    """
    Implements the strictly input-aware backdoor attack training (Algorithm 1 from diversity.tex).
    
    This function trains both the classifier f and trigger generator g with:
    1. Classification loss from three modes (clean, attack, cross-trigger)
    2. Diversity loss to ensure trigger variety
    
    The key insight for NONREUSABILITY:
    - In cross-trigger mode, we apply trigger g(x') to input x
    - The model should predict the ORIGINAL label y (not target c)
    - This enforces that triggers are input-specific and cannot be reused
    
    Args:
        bd_model: Trigger generator network g (being trained)
        bd_model_prev: Frozen copy of trigger generator (for stable training)
        surr_model: Classifier network f
        loader1: Primary data loader
        args: Arguments containing:
            - target_label: Target class c for backdoor
            - p_attack (rho_b): Probability of attack mode
            - p_cross (rho_c): Probability of cross-trigger mode
            - div_reg (lambda_div): Weight for diversity loss
            - attack_only_nontarget: Whether to only attack non-target samples
        loader2: Secondary data loader for cross-trigger samples
        opt_trig: Optimizer for trigger generator
        opt_class: Optimizer for classifier
        train: Whether in training mode
    
    Returns:
        total_loss: Average total loss
        loss_dict: Dictionary with breakdown of losses
        accuracy: Clean accuracy
        bd_accuracy: Backdoor attack success rate
    """
    total_loss = []
    all_preds = []
    bd_preds = []
    cross_preds = []
    trues = []
    bds = []
    cross_trues = []
    
    bd_label = args.target_label
    loss_dict = {'CE_clean': [], 'CE_bd': [], 'CE_cross': [], 'div': [], 'reg': []}
    
    # Probabilities for each mode (from paper)
    rho_b = getattr(args, 'p_attack', 0.5)      # backdoor probability ρ_b
    rho_c = getattr(args, 'p_cross', 0.1)       # cross-trigger probability ρ_c
    lambda_div = getattr(args, 'div_reg', 1.0)  # diversity loss weight λ_div
    
    assert rho_b + rho_c < 1.0, "ρ_b + ρ_c must be less than 1"
    
    # Ensure loader2 is available for cross-trigger mode
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)
    
    bd_model_prev.eval()  # Frozen trigger generator for classifier training
    
    if train:
        surr_model.train()
        bd_model.train()
    else:
        surr_model.eval()
        bd_model.eval()
    
    for i, (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(
        range(len(loader1)), loader1, loader2
    ):
        batch_size = batch_x.shape[0]
        
        # Move data to device
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        
        if batch_x2 is not None:
            batch_x2 = batch_x2.float().to(args.device)
            padding_mask2 = padding_mask2.float().to(args.device)
            label2 = label2.to(args.device)
        else:
            # If no second loader, use shuffled version of first batch
            indices = torch.randperm(batch_size)
            batch_x2 = batch_x[indices]
            padding_mask2 = padding_mask[indices]
            label2 = label[indices]
        
        # Generate backdoor labels (target class c)
        if args.bd_type == 'all2all':
            bd_labels = torch.randint(0, args.numb_class, (batch_size,)).to(args.device)
        elif args.bd_type == 'all2one':
            bd_labels = torch.ones_like(label).to(args.device) * bd_label
        else:
            raise ValueError('bd_type should be all2all or all2one')
        
        # Compute number of samples for each mode
        num_bd = int(rho_b * batch_size)
        num_cross = int(rho_c * batch_size)
        num_clean = batch_size - num_bd - num_cross
        
        # Ensure we have at least one sample in each mode
        num_bd = max(1, num_bd)
        num_cross = max(1, min(num_cross, batch_size - num_bd - 1))
        num_clean = batch_size - num_bd - num_cross
        
        # Create mask for attacking only non-target samples if specified
        if args.attack_only_nontarget:
            attack_mask = (label != bd_label).float().to(args.device)
        else:
            attack_mask = torch.ones_like(label).float().to(args.device)
        attack_mask = attack_mask.unsqueeze(-1).unsqueeze(-1)
        if len(batch_x.shape) == 3:
            attack_mask = attack_mask.expand(-1, batch_x.shape[1], batch_x.shape[2])
        
        # ============== PHASE 1: Train Classifier with Frozen Trigger Generator ==============
        bd_model.zero_grad()
        surr_model.zero_grad()
        
        with torch.no_grad():
            # Generate triggers with frozen model
            trigger_t, trigger_t_clip = bd_model_prev(batch_x, padding_mask, None, None, bd_labels)
            trigger_t_prime, trigger_t_prime_clip = bd_model_prev(batch_x2, padding_mask2, None, None, bd_labels)
        
        # Split batch into three modes
        # Attack mode samples (indices 0 to num_bd)
        x_attack = batch_x[:num_bd]
        y_attack = bd_labels[:num_bd]  # Target label c
        trigger_attack = trigger_t_clip[:num_bd]
        mask_attack = attack_mask[:num_bd]
        
        # Cross-trigger mode samples (indices num_bd to num_bd+num_cross)
        x_cross = batch_x[num_bd:num_bd + num_cross]
        y_cross = label[num_bd:num_bd + num_cross]  # Original label y (NOT target!)
        trigger_cross = trigger_t_prime_clip[num_bd:num_bd + num_cross]  # Trigger from x'
        mask_cross = attack_mask[num_bd:num_bd + num_cross]
        
        # Clean mode samples (remaining)
        x_clean = batch_x[num_bd + num_cross:]
        y_clean = label[num_bd + num_cross:]
        
        # Compute predictions for each mode
        # Attack mode: B(x, g(x)) -> should predict c
        bd_inputs = apply_trigger(x_attack, trigger_attack, mask_attack)
        pred_attack = surr_model(bd_inputs, padding_mask[:num_bd], None, None)
        
        # Cross-trigger mode: B(x, g(x')) -> should predict y (NONREUSABILITY!)
        cross_inputs = apply_trigger(x_cross, trigger_cross, mask_cross)
        pred_cross = surr_model(cross_inputs, padding_mask[num_bd:num_bd + num_cross], None, None)
        
        # Clean mode: x -> should predict y
        pred_clean = surr_model(x_clean, padding_mask[num_bd + num_cross:], None, None)
        
        # Compute classification losses
        loss_ce_attack = args.criterion(pred_attack, y_attack.long().squeeze(-1))
        loss_ce_cross = args.criterion(pred_cross, y_cross.long().squeeze(-1))  # Original label!
        loss_ce_clean = args.criterion(pred_clean, y_clean.long().squeeze(-1))
        
        # Total classification loss
        loss_class = loss_ce_attack + loss_ce_cross + loss_ce_clean
        
        if opt_class is not None and train:
            loss_class.backward()
            opt_class.step()
        
        # ============== PHASE 2: Train Trigger Generator ==============
        surr_model.eval()  # Freeze classifier for trigger training
        bd_model.zero_grad()
        
        # Generate triggers with active model
        trigger_t, trigger_t_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
        trigger_t_prime, trigger_t_prime_clip = bd_model(batch_x2, padding_mask2, None, None, bd_labels)
        
        # ============== Diversity Loss (Equation from paper) ==============
        # L_div = ||x - x'|| / ||g(x) - g(x')||
        loss_div = diversity_loss(batch_x, batch_x2, trigger_t, trigger_t_prime)
        loss_div = loss_div * lambda_div
        
        # ============== Backdoor Loss for Trigger Generator ==============
        # The trigger should make classifier predict target class
        bd_inputs_trig = apply_trigger(batch_x[:num_bd], trigger_t_clip[:num_bd], mask_attack)
        pred_bd_trig = surr_model(bd_inputs_trig, padding_mask[:num_bd], None, None)
        loss_bd_trig = args.criterion(pred_bd_trig, bd_labels[:num_bd].long().squeeze(-1))
        
        # ============== Cross-Trigger Loss for Nonreusability ==============
        # When using trigger from x' on x, model should predict original label y
        # This enforces that triggers cannot be reused across inputs
        cross_inputs_trig = apply_trigger(
            batch_x[num_bd:num_bd + num_cross],
            trigger_t_prime_clip[num_bd:num_bd + num_cross],
            mask_cross
        )
        pred_cross_trig = surr_model(cross_inputs_trig, padding_mask[num_bd:num_bd + num_cross], None, None)
        loss_cross_trig = args.criterion(pred_cross_trig, label[num_bd:num_bd + num_cross].long().squeeze(-1))
        
        # Optional regularization loss
        loss_reg = reg_loss(batch_x, trigger_t, trigger_t_clip, args)
        if loss_reg is None:
            loss_reg = torch.tensor(0.0).to(args.device)
        
        # Total trigger loss: L_total = L_cla + λ_div * L_div
        loss_trig = loss_bd_trig + loss_cross_trig + loss_div + loss_reg
        
        if opt_trig is not None and train:
            loss_trig.backward()
            opt_trig.step()
        
        # ============== Logging ==============
        total_loss.append((loss_class.item() + loss_trig.item()))
        loss_dict['CE_clean'].append(loss_ce_clean.item())
        loss_dict['CE_bd'].append(loss_ce_attack.item())
        loss_dict['CE_cross'].append(loss_ce_cross.item())
        loss_dict['div'].append(loss_div.item())
        loss_dict['reg'].append(loss_reg.item())
        
        # Store predictions for accuracy calculation
        all_preds.append(pred_clean.detach())
        bd_preds.append(pred_attack.detach())
        cross_preds.append(pred_cross.detach())
        trues.append(y_clean)
        bds.append(y_attack)
        cross_trues.append(y_cross)
    
    # Synchronize trigger generators
    pull_model(bd_model_prev, bd_model)
    
    # Compute metrics
    total_loss = np.average(total_loss)
    
    all_preds = torch.cat(all_preds, 0)
    bd_preds = torch.cat(bd_preds, 0)
    cross_preds = torch.cat(cross_preds, 0)
    trues = torch.cat(trues, 0)
    bd_labels_all = torch.cat(bds, 0)
    cross_trues = torch.cat(cross_trues, 0)
    
    # Clean accuracy
    predictions = torch.argmax(torch.nn.functional.softmax(all_preds, dim=-1), dim=1).cpu().numpy()
    trues_np = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues_np)
    
    # Backdoor attack success rate
    bd_predictions = torch.argmax(torch.nn.functional.softmax(bd_preds, dim=-1), dim=1).cpu().numpy()
    bd_accuracy = cal_accuracy(bd_predictions, bd_labels_all.flatten().cpu().numpy())
    
    # Cross-trigger accuracy (should be high = nonreusability)
    cross_predictions = torch.argmax(torch.nn.functional.softmax(cross_preds, dim=-1), dim=1).cpu().numpy()
    cross_accuracy = cal_accuracy(cross_predictions, cross_trues.flatten().cpu().numpy())
    
    # Add cross accuracy to loss dict for monitoring
    loss_dict['cross_acc'] = [cross_accuracy]
    
    return total_loss, loss_dict, accuracy, bd_accuracy


def epoch_marksman_input_aware_eval(
    bd_model,
    surr_model,
    loader1,
    args,
    loader2=None,
    mode='all'  # 'clean', 'attack', 'cross', or 'all'
):
    """
    Evaluation function for the input-aware backdoor attack.
    
    This function evaluates:
    1. Clean accuracy (CA): accuracy on clean inputs
    2. Attack success rate (ASR): accuracy of backdoor attack
    3. Cross-trigger accuracy (CTA): accuracy when using wrong triggers (nonreusability)
    
    High CTA means the attack satisfies the nonreusability criterion.
    
    Args:
        bd_model: Trigger generator
        surr_model: Classifier
        loader1: Primary data loader
        args: Arguments
        loader2: Secondary data loader for cross-trigger evaluation
        mode: Which metrics to compute
    
    Returns:
        Dictionary with evaluation metrics
    """
    bd_model.eval()
    surr_model.eval()
    
    clean_preds = []
    attack_preds = []
    cross_preds = []
    clean_labels = []
    attack_labels = []
    cross_labels = []
    
    bd_label = args.target_label
    
    if loader2 is None:
        loader2 = [[None, None, None]] * len(loader1)
    
    with torch.no_grad():
        for (batch_x, label, padding_mask), (batch_x2, label2, padding_mask2) in zip(loader1, loader2):
            batch_x = batch_x.float().to(args.device)
            padding_mask = padding_mask.float().to(args.device)
            label = label.to(args.device)
            
            if batch_x2 is not None:
                batch_x2 = batch_x2.float().to(args.device)
                padding_mask2 = padding_mask2.float().to(args.device)
            else:
                indices = torch.randperm(batch_x.shape[0])
                batch_x2 = batch_x[indices]
                padding_mask2 = padding_mask[indices]
            
            bd_labels = torch.ones_like(label).to(args.device) * bd_label
            
            # Generate triggers
            _, trigger_clip = bd_model(batch_x, padding_mask, None, None, bd_labels)
            _, trigger_clip2 = bd_model(batch_x2, padding_mask2, None, None, bd_labels)
            
            if mode in ['clean', 'all']:
                pred_clean = surr_model(batch_x, padding_mask, None, None)
                clean_preds.append(pred_clean)
                clean_labels.append(label)
            
            if mode in ['attack', 'all']:
                bd_input = batch_x + trigger_clip
                pred_attack = surr_model(bd_input, padding_mask, None, None)
                attack_preds.append(pred_attack)
                attack_labels.append(bd_labels)
            
            if mode in ['cross', 'all']:
                # Cross-trigger: apply trigger from x2 to x1
                cross_input = batch_x + trigger_clip2
                pred_cross = surr_model(cross_input, padding_mask, None, None)
                cross_preds.append(pred_cross)
                cross_labels.append(label)  # Should predict original label!
    
    results = {}
    
    if clean_preds:
        clean_preds = torch.cat(clean_preds, 0)
        clean_labels = torch.cat(clean_labels, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(clean_preds, dim=-1), dim=1).cpu().numpy()
        results['clean_accuracy'] = cal_accuracy(predictions, clean_labels.flatten().cpu().numpy())
    
    if attack_preds:
        attack_preds = torch.cat(attack_preds, 0)
        attack_labels = torch.cat(attack_labels, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(attack_preds, dim=-1), dim=1).cpu().numpy()
        results['attack_success_rate'] = cal_accuracy(predictions, attack_labels.flatten().cpu().numpy())
    
    if cross_preds:
        cross_preds = torch.cat(cross_preds, 0)
        cross_labels = torch.cat(cross_labels, 0)
        predictions = torch.argmax(torch.nn.functional.softmax(cross_preds, dim=-1), dim=1).cpu().numpy()
        # High cross_accuracy = good nonreusability (triggers don't transfer)
        results['cross_trigger_accuracy'] = cal_accuracy(predictions, cross_labels.flatten().cpu().numpy())
        # Low attack rate with wrong trigger = good nonreusability
        bd_labels_cross = torch.ones_like(cross_labels).to(args.device) * bd_label
        results['cross_trigger_attack_rate'] = cal_accuracy(
            predictions, bd_labels_cross.flatten().cpu().numpy()
        )
    
    return results


