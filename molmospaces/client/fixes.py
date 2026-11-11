"""Episode-setup fix for MolmoSpaces v0.2.9 (never touches scoring).

At v0.2.9 the eval samplers refuse an articulated target unless get_joint_grasp_path finds a grasp file, and that
lookup returns nothing for objects built into the house (allenai/molmospaces#218), so about half the Open/Close
episodes fail to load. Upstream removed the check (allenai/molmospaces#225): at eval time the target joint and its
start position come from the episode spec and grasps are never used. Here we make the same check pass by retrying
the lookup with the builtin `droid` library, as that function's own comment prescribes.
"""
import molmo_spaces.utils.grasps as grasps

_lookup = grasps.get_joint_grasp_path


def get_joint_grasp_path(uid, joint_name, grasp_libraries=None):
    found = _lookup(uid, joint_name, grasp_libraries)
    if found is None and grasp_libraries is None:
        found = _lookup(uid, joint_name, ["droid"])
    return found


grasps.get_joint_grasp_path = get_joint_grasp_path
