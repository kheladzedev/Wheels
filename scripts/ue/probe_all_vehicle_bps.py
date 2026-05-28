"""Enumerate every Blueprint asset under /Game/VehicleVarietyPack/Blueprints,
inspect its parent class, and report which ones are spawnable in editor world
(i.e. not WheeledVehiclePawn / ChaosWheeledVehicle Pawns).
"""

import unreal

ar = unreal.AssetRegistryHelpers.get_asset_registry()
bps = []
for a in ar.get_assets_by_path("/Game/VehicleVarietyPack/Blueprints", recursive=True):
    try:
        cls = str(a.asset_class_path.asset_name)
    except AttributeError:
        cls = ""
    if cls == "Blueprint":
        bps.append(str(a.package_name))

unreal.log(f"[probe] {len(bps)} BP assets in VehicleVarietyPack/Blueprints")
for pkg in bps:
    bp = unreal.EditorAssetLibrary.load_asset(pkg)
    if bp is None:
        unreal.log(f"  {pkg}  <CANT LOAD>")
        continue
    parent = None
    try:
        parent = bp.get_editor_property("parent_class")
    except Exception:
        pass
    p_name = parent.get_name() if parent else "?"
    is_pawn = "Pawn" in p_name or "Vehicle" in p_name
    unreal.log(f"  {pkg}  parent={p_name}  is_pawn={is_pawn}")
