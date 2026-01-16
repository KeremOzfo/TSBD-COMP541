from dataclasses import dataclass


@dataclass
class AttackFormation:
    target_label: int = 0
    input_target_mapping: str = "all2one"


@dataclass
class VanillaTriggerConfigs:
    attack_formation: AttackFormation



@dataclass
class MarksmanTriggerConfigs:
    alpha: float = 1.0
    beta: float = 0.0
    sync_period: int = 5
    attack_formation: AttackFormation


TriggerConfig = Union[MarksmanTriggerConfigs, VanillaTriggerConfigs]

@dataclass
class TriggerTrainConfig:
    trigger_lr: float =  1e-3
    surrogate_lr: float = 1e-3
    train_epochs: int = 10
    trigger_train: TriggerConfig

@dataclass
class TrainConfig: # can be a seperate class rtaher than nested
    num_workers: int = 10
    trigger_config : TriggerTrainConfig
    
