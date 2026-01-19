import torch
import torch.nn as nn
import numpy as np

from .utils import cal_accuracy   # reuse shared accuracy util


def reg_loss(batch_x, trigger, trigger_clip, args):
    """Regularization loss placeholder - can be extended with magnitude penalty"""
    return None




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
            label = label.to(args.device).long()
            #### Generate backdoor labels based on bd_type
            bd_type = getattr(args, 'bd_type', 'all2one')
            num_classes = getattr(args, 'num_class', getattr(args, 'numb_class', None))
            if bd_type == 'all2all' and num_classes is not None:
                bd_labels = torch.randint(0, num_classes, label.shape, device=args.device)
            else:  # all2one
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
                
                # Gradient clipping
                if hasattr(args, 'trigger_grad_clip') and args.trigger_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(bd_model.parameters(), args.trigger_grad_clip)
                if hasattr(args, 'surrogate_grad_clip') and args.surrogate_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(surr_model.parameters(), args.surrogate_grad_clip)
                
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
