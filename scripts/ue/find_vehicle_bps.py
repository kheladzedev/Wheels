"""Clean my CarSpawn_* + enumerate Blueprint assets that look like vehicle BPs.

A "vehicle BP" is recognized by package path under known car packs OR by name
hints (BP_Veh, BP_Car, Vehicle, etc.).
"""

import unreal

sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = sub.get_editor_world()

# 1. Clean spawned junk
actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
removed = 0
for a in actors:
    lbl = a.get_actor_label() or ""
    if lbl.startswith("CarSpawn_"):
        unreal.EditorLevelLibrary.destroy_actor(a)
        removed += 1
unreal.log(f"[clean] destroyed {removed} CarSpawn_* actors")

# 2. Enumerate Blueprint assets across all known car-ish content roots
ar = unreal.AssetRegistryHelpers.get_asset_registry()
ROOTS = (
    "/Game/CitySampleVehicles",
    "/Game/VehicleVarietyPack",
    "/Game/SketchfabCars",
    "/Game/Wheels",
)
for root in ROOTS:
    assets = ar.get_assets_by_path(root, recursive=True)
    bps = []
    for a in assets:
        try:
            cls_name = str(a.asset_class_path.asset_name)
        except AttributeError:
            cls_name = ""
        if cls_name == "Blueprint":
            bps.append(str(a.package_name))
    unreal.log(f"=== {root} :: {len(bps)} Blueprint assets ===")
    for p in bps[:25]:
        unreal.log(f"  {p}")
