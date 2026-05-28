"""Move editor viewport camera, take screenshot. Bypass SceneCaptureComponent2D."""

import unreal

# Set editor viewport camera to a defined position
loc = unreal.Vector(74769.0, 85346.0, 44700.0)  # 1m above floor
rot = unreal.Rotator(roll=0.0, pitch=-10.0, yaw=45.0)

unreal.EditorLevelLibrary.set_level_viewport_camera_info(loc, rot)
unreal.log(
    f"[viewport] camera set to ({loc.x:.0f},{loc.y:.0f},{loc.z:.0f}) yaw={rot.yaw} pitch={rot.pitch}"
)
