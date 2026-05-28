"""Clean up every test actor I spawned during this session."""

import unreal

sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = sub.get_editor_world()

LABEL_PREFIXES = (
    "BP_SUV_test",
    "WheelTest_",
    "Car_",
    "CarSpawn_",
    "OneWheel_",
    "WheelsBP_test",
)

removed = 0
for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
    lbl = a.get_actor_label() or ""
    if lbl.startswith(LABEL_PREFIXES):
        unreal.EditorLevelLibrary.destroy_actor(a)
        removed += 1

unreal.log(f"[cleanup] removed {removed} test actors")
