#!/usr/bin/env python3
"""Fail-closed high-level action adapter for the packaged A03 Type-2 API."""
from dataclasses import dataclass
import numpy as np

class A03ContractError(ValueError): pass

@dataclass(frozen=True)
class A03IkCommand:
    target_position_m: np.ndarray
    target_tip_direction: np.ndarray
    gripper_command: float
    translation_clipped: bool

def make_local_ik_command(action7,current_position_m,current_tip_direction,*,max_translation_step_m=0.01,rotation_threshold_rad=0.001):
    action=np.asarray(action7,dtype=np.float64); position=np.asarray(current_position_m,dtype=np.float64); direction=np.asarray(current_tip_direction,dtype=np.float64)
    if action.shape!=(7,) or position.shape!=(3,) or direction.shape!=(3,): raise A03ContractError("schema mismatch")
    if not np.isfinite(action).all() or not np.isfinite(position).all() or not np.isfinite(direction).all(): raise A03ContractError("non-finite input")
    if float(np.linalg.norm(action[3:6]))>rotation_threshold_rad: raise A03ContractError("unmapped rotation semantics")
    norm=float(np.linalg.norm(direction))
    if norm<1e-9: raise A03ContractError("zero tip direction")
    direction=direction/norm; delta=action[:3]; delta_norm=float(np.linalg.norm(delta)); clipped=delta_norm>max_translation_step_m
    if clipped: delta=delta*(max_translation_step_m/delta_norm)
    # Frozen OpenVLA LIBERO execution convention: -1=open, +1=close.
    grip=float(np.clip((action[6]+1.0)/2.0,0.0,1.0))
    return A03IkCommand(position+delta,direction,grip,clipped)
