# Phase 6B — Feagine A03 Type-2 Asset Contract

The supplied v1.0.1 distribution contains core 0.1.0 (`2c59c71`) and MuJoCo wheel 0.1.2 with the
`a03_type_2` preset. The archive is audited in place without installing or extracting it.

The actual robot contract is three fixed 0.16 m PCC sections with five samples per section. Each section exposes
`theta` and `phi`; core IK optimizes those six values against target position and unit tip direction. The MJCF uses
a 0.005 s timestep and a 16-frame articulated proxy (48 capsule joints/actuators, including its terminal orientation
frame), plus grasper rotation and two finger joints. The wrapper converts PCC section angles to proxy joint targets.

The high-level adapter now outputs the actual core-compatible IK target rather than Phase 6A's provisional
single-section extensible state. Translation is limited to 0.01 m per command. LIBERO/OpenVLA gripper convention
`-1=open,+1=close` maps to wrapper `0=open,1=close`. Rotation requests remain rejected until frame/orientation
semantics are calibrated.

One packaging issue is explicit: package code defaults to preset `a03`, while this wheel contains only
`a03_type_2`; every project call must pass `preset_id="a03_type_2"`. Static PASS does not mean the wheel/core is
installed or that MJCF loads. Those require a separate host-authorized environment change and runtime gate.
