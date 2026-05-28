"""Enumerate actors with a static/skeletal mesh under the known car package paths."""

import unreal

CAR_PATHS = (
    "/Game/SketchfabCars/",
    "/Game/CitySampleVehicles/",
    "/Game/VehicleVarietyPack/",
    "/Game/Wheels/",  # potential wheel-only meshes
)

sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = sub.get_editor_world()
actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)

hits: list[tuple[str, str, str]] = []
for a in actors:
    if a is None:
        continue
    for comp in a.get_components_by_class(unreal.StaticMeshComponent):
        mesh = comp.get_editor_property("static_mesh")
        if mesh is None:
            continue
        p = mesh.get_path_name()
        if any(p.startswith(prefix) for prefix in CAR_PATHS):
            hits.append(
                (a.get_actor_label() or a.get_name(), a.get_class().get_name(), p)
            )
            break
    else:
        for comp in a.get_components_by_class(unreal.SkeletalMeshComponent):
            mesh = comp.get_editor_property(
                "skeletal_mesh"
            ) or comp.get_editor_property("skinned_asset")
            if mesh is None:
                continue
            p = mesh.get_path_name()
            if any(p.startswith(prefix) for prefix in CAR_PATHS):
                hits.append(
                    (a.get_actor_label() or a.get_name(), a.get_class().get_name(), p)
                )
                break

unreal.log(f"[cars] {len(hits)} actors with car-package mesh")
for label, cls, mesh in hits[:30]:
    unreal.log(f"  {label} <{cls}>  mesh={mesh}")

# Also: any actor with label starting with Actor2 / Actor3 / Actor4 (Outliner showed these)
for a in actors:
    lbl = a.get_actor_label() or ""
    if lbl.lower().startswith("actor") and lbl[5:].isdigit():
        loc = a.get_actor_location()
        unreal.log(
            f"[actorN] {lbl} <{a.get_class().get_name()}> loc=({loc.x:.0f},{loc.y:.0f},{loc.z:.0f})"
        )
