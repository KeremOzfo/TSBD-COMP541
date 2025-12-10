from data_provider.data_loader import UEAloader, UEAloader_bd, UEAloader_bd2
from data_provider.uea import collate_fn, collate_fn_bd
from torch.utils.data import DataLoader
import functools


data_dict = {
    'UEA': UEAloader
}


def data_provider(args, flag):
    """Create dataset and dataloader for UEA classification."""

    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    # ======== UEA classification only ==========
    data_set = Data(
        root_path=args.root_path,
        flag=flag,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=functools.partial(collate_fn, max_len=args.seq_len)
    )

    return data_set, data_loader



def custom_data_loader(dataset, args, flag, force_bs=None):

    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        batch_size = args.batch_size

    if force_bs is not None:
        batch_size = force_bs

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=functools.partial(collate_fn, max_len=args.seq_len)
    )

    return data_loader



def bd_data_provider(args, flag, bd_model):
    """Backdoored on-the-fly UEA loader"""

    Data = UEAloader_bd
    bd_model.to("cpu")

    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        batch_size = args.batch_size

    data_set = Data(
        bd_model=bd_model,
        poision_rate=args.poisoning_ratio,
        silent_poision=args.silent_poisoning,
        target_label=args.target_label,
        root_path=args.root_path,
        flag=flag,
        max_len=args.seq_len,
        enc_in=args.enc_in
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=functools.partial(collate_fn_bd,
            G=bd_model, max_len=args.seq_len, target_label=args.target_label
        )
    )

    return data_set, data_loader



def bd_data_provider2(args, flag, bd_model):
    """Backdoor labeling (no injection inside dataset)."""

    Data = UEAloader_bd2
    bd_model.to("cpu")

    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        batch_size = args.batch_size

    data_set = Data(
        bd_model=bd_model,
        poision_rate=args.poisoning_ratio,
        silent_poision=args.silent_poisoning,
        target_label=args.target_label,
        root_path=args.root_path,
        flag=flag
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=functools.partial(collate_fn_bd,
            G=bd_model, max_len=args.seq_len, target_label=args.target_label
        )
    )

    if flag == 'test':
        return (
            data_set,
            data_loader,
            data_set.bd_inds,
            data_set.silent_bd_set
        )

    return data_set, data_loader
