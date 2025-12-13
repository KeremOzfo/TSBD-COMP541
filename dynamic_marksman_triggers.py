def epoch_marksman_input_aware(
    trigger_model,          
    trigger_model_prev,      
    surrogate_model,        
    loader1,            
    args,
    loader2=None,    
    opt_trig=None,      
    opt_class=None,    
    train=True
):
    """ Implementation of Marksman framework introduced in [1].
    
    [1]Doan, Khoa D., Yingjie Lao, and Ping Li. "Marksman backdoor: Backdoor attacks with arbitrary target class." Advances in Neural Information Processing Systems 35 (2022): 38260-38273.
    
    Parameters:
        trigger_model( torch.nn.Module): Neural network architecture for generating backdoor trigger
        trigger_model_prev (torch.nn.Module):
        surrogate_model( torch.nn.Module):
        loader1: (torch.utils.data.DataLoader): data loader
        args: Arguments containing:
            - target_label: Target class c for backdoor
            - p_attack (rho_b): Probability of attack mode
            - p_cross (rho_c): Probability of cross-trigger mode
            - div_reg (lambda_div): Weight for diversity loss
            - attack_only_nontarget: Whether to only attack non-target samples
        loader2:  (torch.utils.data.DataLoader): data loader
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
    
    for i, (batch_1, label, padding_mask), (batch_2, label2, padding_mask2) in zip(
        range(len(loader1)), loader1, loader2
    ):
        batch_size = batch_1.shape[0]
        batch_1 = batch_1.float().to(args.device)
        padding_mask = padding_mask.float().to(args.device)
        label = label.to(args.device)
        
        if batch_2 is not None:
            batch_2 = batch_2.float().to(args.device)
            padding_mask2 = padding_mask2.float().to(args.device)
            label2 = label2.to(args.device)
        else: # If there is only one dataloader use the suffled version of the first batch
            indices = torch.randperm(batch_size)
            batch_2 = batch_1[indices]
            padding_mask2 = padding_mask[indices]
            label2 = label[indices]
        if args.bd_type == 'all2all': 
            bd_labels = torch.randint(0, args.numb_class, (batch_size,)).to(args.device)
        elif args.bd_type == 'all2one':
            bd_labels = torch.ones_like(label).to(args.device) * bd_label
        else:
            raise ValueError('bd_type should be all2all or all2one')
        
        num_bd = int(rho_b * batch_size) # number of packdoor samples (trigger added data and false label)
        num_cross = int(rho_c * batch_size) # number of trigger only samples (trigger added data and clean label)
        num_clean = batch_size - num_bd - num_cross # number of clean samples (clean data and clean label)
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
        bd_model.zero_grad()
        surr_model.zero_grad()
        
        with torch.no_grad():
            trigger_t, trigger_t_clip = trigger_model_prev(batch_1, padding_mask, None, None, bd_labels) # Generate triggers using frozen trigger model
            trigger_t_prime, trigger_t_prime_clip = trigger_model_prev(batch_2, padding_mask2, None, None, bd_labels)
        
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

