import numpy as np
import torch
from typing import Any
from contextlib import nullcontext
from copy import copy
import logging
from dataclasses import dataclass, field
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


class JointAdapter:
    """Bridges a policy that controls only a subset of the robot's joints with the
    robot's full command space.

    The policy's action/observation space is whatever the training dataset recorded
    (e.g. a single arm + hand). The robot, however, must always be commanded over its
    full joint set. This adapter maps between the two using the joint *names* as the
    single source of truth, so it generalizes to any subset without index arithmetic:

      * ``to_policy`` gathers the controlled joints out of the full robot state, in the
        exact order the policy expects.
      * ``to_robot`` scatters a policy action back into a full command vector, leaving
        every uncontrolled joint at its default (held) value.

    The full command vector is assembled, in order, as
    ``[dual_arm_q (arm_dof), left_ee (ee_dof), right_ee (ee_dof)]`` which matches the
    canonical motor ordering in ``constants.ROBOT_CONFIGS``.

    Defaults for the uncontrolled joints are not part of the dataset — they are seeded
    from the live robot state via ``set_defaults`` (i.e. wherever the operator placed
    the unused limb), then frozen there.
    """

    def __init__(self, policy_names: list[str], canonical_names: list[str], full_dim: int):
        missing = [n for n in policy_names if n not in canonical_names]
        if missing:
            raise ValueError(
                f"Policy joints {missing} are not in the robot's canonical joint layout. "
                f"Available joints: {canonical_names}"
            )
        if len(canonical_names) != full_dim:
            raise ValueError(
                f"Canonical layout has {len(canonical_names)} joints but the robot command "
                f"vector has {full_dim} dims; check the arm/end-effector configuration."
            )
        self.full_dim = full_dim
        self.robot_idx = np.array([canonical_names.index(n) for n in policy_names], dtype=int)
        self.default_full = np.zeros(full_dim, dtype=np.float64)

    def set_defaults(self, full_state: np.ndarray) -> None:
        """Seed the held pose for uncontrolled joints from the live robot state."""
        full_state = np.asarray(full_state, dtype=np.float64)
        if full_state.shape[0] != self.full_dim:
            raise ValueError(
                f"Expected full state of dim {self.full_dim}, got {full_state.shape[0]}."
            )
        self.default_full = full_state.copy()

    def to_policy(self, full_state: np.ndarray) -> np.ndarray:
        """Gather the controlled joints out of the full robot state."""
        return np.asarray(full_state, dtype=np.float64)[self.robot_idx]

    def to_robot(self, policy_action: np.ndarray) -> np.ndarray:
        """Scatter a policy action into a full robot command, holding the rest."""
        full = self.default_full.copy()
        full[self.robot_idx] = np.asarray(policy_action, dtype=np.float64)
        return full


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if not use_dataset:
                # Skip non-tensor observations (like task strings)
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        action = postprocessor(action)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def reset_policy(policy: PreTrainedPolicy):
    policy.reset()


def cleanup_resources(image_info: dict[str, Any]):
    """Safely close and unlink shared memory resources."""
    logger_mp.info("Cleaning up shared memory resources.")
    for shm in image_info["shm_resources"]:
        if shm:
            shm.close()
            shm.unlink()


def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Basic control parameters
    arm: str = "G1_29"  # G1_29, G1_23
    ee: str = "dex3"  # dex3, dex1, inspire1, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    visualization: bool = False
    send_real_robot: bool = False
    use_dataset: bool = False

    rename_map: dict[str, str] = field(default_factory=dict)

    image_host: str = "192.168.123.164"

    # Network interface (NIC) connected to the robot, e.g. "enx00e04c685878".
    # Passed to ChannelFactoryInitialize(0, net_interface): the Unitree SDK builds
    # its own inline CycloneDDS config and IGNORES CYCLONEDDS_URI, so the NIC must
    # be pinned here or DDS auto-selects the wrong interface and never sees rt/lowstate.
    net_interface: str = "enx00e04c685878"

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
