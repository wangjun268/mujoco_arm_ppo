# LinkerHand L20 (灵心巧手) — vendored model

Source: the vendor's official URDF repository, **https://github.com/linker-bot/linkerhand-urdf**
(`L20/right/`), licence **Apache-2.0**.  The website (https://www.linkerhand.com) links to the
same repository.

```
right/linkerhand_l20_right.urdf   # vendor file, unchanged
right/meshes/*.STL                # 22 STL parts, unchanged
```

## Generated files — do not edit by hand

MuJoCo merges a URDF's root link into the *world* body, so the vendor URDF cannot be bolted onto
the Pro7 wrist directly.  `tools/convert_hand_urdf.py` (in the checkout's `tools/`) re-emits the same kinematics as
MJCF fragments the arm scenes `<include>`, and calibrates the grasp by simulating it:

```
linkerhand_l20_right_assets.xml     # <mesh> declarations
linkerhand_l20_right_body.xml       # the 22 bodies, rooted at hand_base, + grasp_center site
linkerhand_l20_right_actuators.xml  # one position servo per joint
linkerhand_l20_poses.json           # open / power-grasp presets + finger -> geom map
linkerhand_l20_right.xml            # standalone model, for eyeballing the hand on its own
```

Regenerate everything with:

```bash
python3 tools/convert_hand_urdf.py
```

## Kinematics

21 hinge joints.  Index/middle/ring/pinky each have `mcp_roll`, `mcp_pitch`, `pip`, `dip`;
the thumb has `cmc_yaw`, `cmc_roll`, `cmc_pitch`, `mcp`, `dip`.  In the hand frame the fingers
run along `+z` and the palm faces `+x`; the arm scenes mount it so the fingers follow the tool's
approach axis.
