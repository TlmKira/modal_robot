"""GR00T N1.7 ALOHA: joints in radians and separate normalized grippers."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

keys = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["head", "left_wrist", "right_wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=keys),
    "action": ModalityConfig(delta_indices=list(range(40)), modality_keys=keys,
        action_configs=[ActionConfig(
            rep=ActionRepresentation.ABSOLUTE if "gripper" in key else ActionRepresentation.RELATIVE,
            type=ActionType.NON_EEF, format=ActionFormat.DEFAULT) for key in keys]),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
