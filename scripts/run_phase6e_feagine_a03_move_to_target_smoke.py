#!/usr/bin/env python3
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/task/phase6e_feagine_a03_move_to_target_smoke_v1.yaml"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def tip_direction(pose):
    w, x, y, z = np.asarray(pose.quaternion, dtype=np.float64)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return rotation @ np.array([0.0, 0.0, 1.0])


def main():
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / config["output_dir"]
    result_path = ROOT / config["structured_result"]
    for path in (output, result_path):
        if path.exists(): raise FileExistsError(f"refusing overwrite: {path}")

    import mujoco
    import pyfeagine_sim_core as core
    from feagine_mujoco import FeagineMjcfRobot, load_mujoco_model

    start = time.perf_counter()
    model = load_mujoco_model(preset_id=config["preset_id"], model_type="mjcf")
    data = mujoco.MjData(model)
    robot = FeagineMjcfRobot(model, data, preset_id=config["preset_id"])
    mujoco.mj_forward(model, data)
    initial_tip = np.asarray(robot.tip_pose().position, dtype=np.float64)

    angles = config["target_generator"]["section_angles_rad"]
    truth = [core.PccSectionConfig(0.16, angles[i], angles[i+1]) for i in range(0, 6, 2)]
    truth_fk = core.compute_forward_kinematics(robot.base_pose, truth, 5)
    pcc_target = truth_fk.section_end_poses[-1]
    direction = tip_direction(pcc_target)
    direction /= np.linalg.norm(direction)
    tool_length = float(np.linalg.norm(np.asarray(robot.preset.grasper_tool_tip_position, dtype=np.float64)))
    target = np.asarray(pcc_target.position, dtype=np.float64) + tool_length * direction

    options = core.InverseKinematicsOptions()
    options.max_iterations = config["ik"]["max_iterations"]
    options.position_tolerance = config["ik"]["position_tolerance_m"]
    options.direction_tolerance = config["ik"]["direction_tolerance"]
    options.clamp_theta = True
    options.theta_limit = config["ik"]["theta_limit_rad"]
    solved = robot.solve_inverse_kinematics(target, direction, options=options, apply_solution=True)
    robot.set_grip_command(config["protocol"]["gripper_command"])
    robot.drive_grasper_rotation(config["protocol"]["grasper_rotation_rad"])

    limited = []
    for joint in range(model.njnt):
        if model.jnt_limited[joint]:
            limited.append((int(model.jnt_qposadr[joint]), int(model.jnt_type[joint]), *map(float, model.jnt_range[joint])))
    max_hinge = max_slide = 0.0
    finite = True
    max_contacts = 0
    for _ in range(config["protocol"]["steps"]):
        mujoco.mj_step(model, data)
        finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all())
        max_contacts = max(max_contacts, int(data.ncon))
        for address, joint_type, lower, upper in limited:
            q = float(data.qpos[address]); penetration = max(lower-q, q-upper, 0.0)
            if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE): max_hinge = max(max_hinge, penetration)
            if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE): max_slide = max(max_slide, penetration)

    final_tip = np.asarray(robot.tip_pose().position, dtype=np.float64)
    initial_error = float(np.linalg.norm(initial_tip-target))
    final_error = float(np.linalg.norm(final_tip-target))
    progress = float((initial_error-final_error)/initial_error) if initial_error > 0 else 0.0
    gate = config["success_gate"]
    material = max_hinge > config["diagnostic_thresholds"]["hinge_material_violation_rad"] or max_slide > config["diagnostic_thresholds"]["slide_material_violation_m"]
    checks = {
        "ik_success": bool(solved.success), "finite_state": finite,
        "exact_time": abs(float(data.time)-config["protocol"]["steps"]*config["protocol"]["timestep_s"]) < 1e-12,
        "target_displacement": gate["minimum_target_displacement_m"] <= initial_error <= gate["maximum_target_displacement_m"],
        "terminal_error": final_error <= gate["terminal_position_error_m"], "progress": progress >= gate["minimum_progress_fraction"],
        "no_material_limit_violation": not material, "step_budget": config["protocol"]["steps"] == 400,
        "no_policy_or_hardware": config["authorization"]["policy_inference"] is False and config["authorization"]["hardware"] is False,
    }
    success = all(checks.values())
    result = {"classification":"PASS" if success else "FAIL", "experiment":config["experiment_id"], "decision":"BOUNDED_CORE_TARGET_REACHING_PASS" if success else "BOUNDED_CORE_TARGET_REACHING_FAIL", "checks":checks, "checks_passed":sum(checks.values()), "checks_total":len(checks), "steps":config["protocol"]["steps"], "simulated_time_s":float(data.time), "target_position_m":target.tolist(), "initial_tip_position_m":initial_tip.tolist(), "final_tip_position_m":final_tip.tolist(), "initial_error_m":initial_error, "final_error_m":final_error, "progress_fraction":progress, "ik":{"success":bool(solved.success), "iterations":int(solved.iterations), "position_error_m":float(solved.final_position_error), "direction_error":float(solved.final_direction_error), "solved_section_angles_rad":[v for section in solved.solved_sections for v in (float(section.theta),float(section.phi))]}, "max_hinge_penetration_rad":max_hinge, "max_slide_penetration_m":max_slide, "max_contacts":max_contacts, "elapsed_s":time.perf_counter()-start, "cuda_initialized":False, "model_predictions":0, "reward":None, "success":success, "hardware_commands":0, "source_hashes":{"config":sha(CONFIG)}, "boundary":config["boundary"]}
    output.mkdir(parents=True)
    (output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    result_path.write_text(yaml.safe_dump(result, sort_keys=False))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__": main()
