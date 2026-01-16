import torch
import numpy as np

from .utils import cal_accuracy

def epoch_marksman(
    trigger_model,
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
    update_T = getattr(args, "marksman_update_T", 1)
    num_classes = getattr(args, "numb_class", getattr(args, "num_class", None))
    if num_classes is None:
        raise ValueError("args.num_class (or args.numb_class) is required for Marksman training")

    if train:
        trigger_model.train()
        surr_model.train()
    else:
        trigger_model.eval()
        surr_model.eval()

    total_loss = []
    loss_dict = {"CE_clean": [], "CE_bd": [], "CE_trig": []}
    clean_preds, bd_preds = [], []
    clean_labels, bd_labels_all = [], []

    for step, (batch_x, label, padding_mask) in enumerate(loader):
        batch_x = batch_x.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device).long()

        # Sample class-conditional targets based on bd_type
        bd_type = getattr(args, 'bd_type', 'all2one')
        if bd_type == 'all2all':
            # All-to-all: each sample gets a random target different from its true label
            rand = torch.randint(0, num_classes - 1, label.shape, device=args.device)
            bd_labels = (rand + label + 1) % num_classes
        else:  # all2one
            bd_labels = (torch.ones_like(label) * args.target_label).long().to(args.device)

        # -------- Classifier update (clean + backdoor) --------
        if train and opt_cls is not None:
            opt_cls.zero_grad()
        if train and opt_trig is not None:
            opt_trig.zero_grad()

        with torch.no_grad():  # keep generator fixed for classifier step
            _, trigger_clip_det = trigger_model(batch_x, padding_mask, None, None, bd_labels)
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
            trigger, trigger_clip = trigger_model(batch_x, padding_mask, None, None, bd_labels)
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