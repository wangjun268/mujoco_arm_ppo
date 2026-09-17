# One snapshot of the cell: the plant, the vision estimate and the running job.
# Published at ``status_hz`` (see the node parameters).

std_msgs/Header header

# "idle" between jobs, otherwise the expert phase:
# approach -> settle -> lift -> transport -> lower -> release -> done
string phase
bool running                  # a pick & place job is being executed
bool holding                  # the jaws are pinching the cube right now
bool picked                   # the cube was pinched at some point
bool placed                   # the cube ended up on the pad, jaws open
int32 steps                   # control steps of the current/last job

# Vision: what the eye-in-hand camera sees (``detected`` is false when the red
# cube is not visible, in which case ``detected_cube`` is stale).
bool detected
geometry_msgs/Point detected_cube

# Plant state.
geometry_msgs/Point cube_position
geometry_msgs/Point place_target
float32 place_error           # |cube - place_target| (m)
float32 gripper_closure       # 0 = open, 1 = closed (hand synergy)
