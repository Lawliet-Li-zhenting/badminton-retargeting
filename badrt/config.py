"""Tunable parameters of the body retargeter and the source (vision) pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RetargetConfig:
    # SQP
    sqp_iters_first: int = 25
    sqp_iters: int = 3
    step_size: float = 0.8                 # SOC trust region ||dq||_2 per SQP step
    topology_reuse_frames: int = 10
    # interaction mesh
    use_laplacian: bool = True
    laplacian_weight: float = 10.0
    laplacian_kappa: float = 8.0           # body scale is ~10x a hand, so kappa is ~1/4 of the hand value
    floor_points: bool = True              # static floor samples in the interaction mesh (foot-ground relation)
    # other costs
    bone_direction_weight: float = 2.0
    keypoint_position_weight: float = 0.5
    root_position_weight: float = 20.0
    smooth_weight: float = 0.05
    rest_weight: float = 0.01
    hand_weight: float = 1.0               # scale of the index/pinky terms (hand orientation -> racket)
    accel_weight: float = 0.0              # ||(q - q_prev) - (q_prev - q_prev2)||^2
    reproj_weight: float = 0.0             # per px^2, image reprojection of the keypoints (calibrated camera)
    racket_front_weight: float = 0.0       # racket head in front of the torso when the hand is below the shoulder
    racket_front_margin: float = 0.1
    # racket tip as an extra joint (detected in the image): mesh vertex / wrist->tip direction scale,
    # and a robust (Geman-McClure) image term per px^2
    racket_weight: float = 0.0
    racket_reproj_weight: float = 0.0
    racket_reproj_sigma_px: float = 10.0
    racket_in_mesh: bool = True            # tip as an interaction-mesh vertex (couples racket and body)
    racket_arm_only: bool = True           # direction / image terms act only through the racket arm
    racket_prior_dofs: str = "arm"         # prior-only racket targets (no detection) steer "arm" or "wrist" DOFs
    racket_prior_in_mesh: bool = True      # prior-only tip as a mesh vertex
    racket_prior_keep_front: bool = False  # keep the hand-made racket-front term at full gate on prior-only frames
    # foot contact lock: a source sole point low and slow is held at its previous character position
    contact_lock_weight: float = 30.0
    contact_height_m: float = 0.04
    contact_speed_mps: float = 0.6
    # footplant anchors (Kovar et al. 2002): planted sole points pulled to fixed world targets
    anchor_weight: float = 300.0
    # self-collision spheres (soft rows, slack cost) and sphere-floor rows
    self_collision: bool = True
    collision_margin: float = 0.005
    collision_activation: float = 0.04     # pairs closer than this (gap, m) enter the QP
    collision_soft_weight: float = 800.0
    max_collision_rows: int = 32
    # constraints
    floor_constraint: bool = True
    prewarm: bool = True

    @classmethod
    def preset(cls, name: str = "fast", **kw) -> "RetargetConfig":
        cfg = cls()
        if name == "accurate":
            cfg.sqp_iters_first, cfg.sqp_iters, cfg.topology_reuse_frames = 50, 10, 1
        elif name == "bones_only":
            cfg.use_laplacian = False
        elif name == "no_contact":
            cfg.contact_lock_weight = 0.0
        elif name == "no_floor":
            cfg.floor_constraint, cfg.contact_lock_weight = False, 0.0
        elif name == "no_collision":
            cfg.self_collision = False
        elif name == "broadcast":
            cfg.sqp_iters_first, cfg.sqp_iters, cfg.topology_reuse_frames = 40, 5, 5
            cfg.contact_lock_weight = 0.0           # replaced by explicit footplant anchors
            cfg.keypoint_position_weight = 1.0
            cfg.reproj_weight = 0.004               # 10 px ~ 0.4: image evidence beats MediaPipe depth
            cfg.accel_weight = 0.3
            cfg.racket_front_weight = 30.0
            cfg.collision_soft_weight = 4000.0
            cfg.racket_weight = 1.5
            cfg.racket_reproj_weight = 0.015
            cfg.racket_reproj_sigma_px = 30.0
        elif name != "fast":
            raise ValueError(f"unknown preset {name}")
        for k, v in kw.items():
            if not hasattr(cfg, k):
                raise AttributeError(k)
            setattr(cfg, k, v)
        return cfg

    def to_dict(self):
        return asdict(self)


@dataclass
class SourceConfig:
    hfov_deg: float = 60.0                 # assumed horizontal FOV of the camera for root translation
    filter_min_cutoff: float = 1.2         # One-Euro filter on world landmarks (Hz)
    filter_beta: float = 0.08
    root_min_cutoff: float = 0.6
    root_beta: float = 0.02
    court_gain: tuple = (1.0, 1.0)         # camera-space metres -> court metres (x lateral, y depth)
    track_translation: bool = True
    translation_mode: str = "scale"        # scale (perspective fit) | ground_ray (experimental)
    max_root_speed_mps: float = 7.0        # elite badminton footwork peaks around 5-6 m/s
    contact_height_m: float = 0.04         # sole keypoint within this of the lowest one
    contact_image_speed: float = 0.35      # body heights per second in the image
    jump_deadband_m: float = 0.03
    handedness: str = "right"              # right | left (left: mirror source to the right-handed character)
    min_torso_up: float = -0.1             # reject frames whose hip->shoulder direction points below this

    def to_dict(self):
        return asdict(self)
