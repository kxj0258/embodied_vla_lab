#!/usr/bin/env python3
"""Small dependency-light VLA-to-continuum bridge reference implementation."""
from dataclasses import dataclass
import math
import numpy as np

class ContractError(ValueError): pass

@dataclass(frozen=True)
class PCCState:
    bend_angle_rad: float
    bending_plane_rad: float
    length_m: float

def _finite(name, value):
    a=np.asarray(value,dtype=np.float64)
    if not np.isfinite(a).all(): raise ContractError(f"{name} must be finite")
    return a

def pcc_forward(state: PCCState) -> np.ndarray:
    theta=float(state.bend_angle_rad); phi=float(state.bending_plane_rad); length=float(state.length_m)
    _finite("PCC state",[theta,phi,length])
    if length<=0: raise ContractError("length must be positive")
    if abs(theta)<1e-10: return np.array([0.0,0.0,length],dtype=np.float64)
    radius=length/theta; rho=radius*(1.0-math.cos(theta))
    return np.array([rho*math.cos(phi),rho*math.sin(phi),radius*math.sin(theta)],dtype=np.float64)

def pcc_inverse_position(target_xyz_m, limits) -> PCCState:
    x,y,z=_finite("target_xyz_m",target_xyz_m).reshape(3); rho=math.hypot(x,y)
    if rho<1e-12:
        if z<=0: raise ContractError("straight target must have positive z")
        state=PCCState(0.0,0.0,float(z))
    else:
        theta=2.0*math.atan2(rho,z)
        chord=math.hypot(rho,z); length=theta*chord/(2.0*math.sin(theta/2.0)); phi=math.atan2(y,x)
        state=PCCState(theta,phi,length)
    curvature=0.0 if abs(state.bend_angle_rad)<1e-12 else state.bend_angle_rad/state.length_m
    if not limits["min_length_m"]<=state.length_m<=limits["max_length_m"]: raise ContractError("length limit")
    if abs(curvature)>limits["max_abs_curvature_per_m"]: raise ContractError("curvature limit")
    if abs(state.bend_angle_rad)>limits["max_bend_angle_rad"]: raise ContractError("bend-angle limit")
    return state

def validate_workspace(target, workspace):
    x,y,z=_finite("target",target).reshape(3); radial=math.hypot(x,y)
    if radial>workspace["radial_max_m"] or not workspace["z_min_m"]<=z<=workspace["z_max_m"]: raise ContractError("workspace limit")

def adapt_libero_delta(action7, current_tip_policy_m, config):
    action=_finite("action7",action7).reshape(7); current=_finite("current_tip_policy_m",current_tip_policy_m).reshape(3)
    rotation_norm=float(np.linalg.norm(action[3:6]))
    if rotation_norm>config["adapter"]["reject_rotation_norm_above_rad"]: raise ContractError("unsupported rotation request")
    delta=action[:3]; norm=float(np.linalg.norm(delta)); clipped=False
    if norm>config["adapter"]["max_translation_step_m"]:
        delta=delta*(config["adapter"]["max_translation_step_m"]/norm); clipped=True
    frame=config["frames"]["policy_to_continuum"]; rot=_finite("frame rotation",frame["rotation"]).reshape(3,3); trans=_finite("frame translation",frame["translation_m"]).reshape(3)
    if not np.allclose(rot.T@rot,np.eye(3),atol=1e-9) or np.linalg.det(rot)<0.999999: raise ContractError("frame rotation must be proper orthonormal")
    target=rot@(current+delta)+trans; validate_workspace(target,config["workspace"]); state=pcc_inverse_position(target,config["pcc_limits"])
    lo,hi=config["adapter"]["gripper_range"]; gripper=float(np.clip(action[6],lo,hi))
    return {"target_tip_continuum_m":target,"pcc_state":state,"gripper":gripper,"translation_clipped":clipped,"rotation_ignored":False}
