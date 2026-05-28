"""Inspect Actor2/3/4 — list every component + its mesh asset path."""

import unreal

sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = sub.get_editor_world()
actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)

for a in actors:
    lbl = a.get_actor_label() or ""
    if not (lbl.lower().startswith("actor") and lbl[5:].isdigit()):
        continue
    unreal.log(f"=== {lbl} <{a.get_class().get_name()}> ===")
    components = a.get_components_by_class(unreal.ActorComponent)
    unreal.log(f"  total components: {len(components)}")
    for c in components:
        cls = c.get_class().get_name()
        line = f"  - {c.get_name()} <{cls}>"
        if isinstance(c, unreal.StaticMeshComponent):
            mesh = c.get_editor_property("static_mesh")
            if mesh is not None:
                line += f"  mesh={mesh.get_path_name()}"
        elif isinstance(c, unreal.SkeletalMeshComponent):
            mesh = c.get_editor_property("skeletal_mesh") or c.get_editor_property(
                "skinned_asset"
            )
            if mesh is not None:
                line += f"  skel={mesh.get_path_name()}"
        unreal.log(line)
    origin, ext = a.get_actor_bounds(True)
    unreal.log(
        f"  bounds origin=({origin.x:.0f},{origin.y:.0f},{origin.z:.0f}) ext=({ext.x:.0f},{ext.y:.0f},{ext.z:.0f})"
    )
