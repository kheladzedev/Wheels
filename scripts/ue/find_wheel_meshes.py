"""Search asset registry for StaticMesh assets whose name contains wheel/tire/rim."""

import unreal

ar = unreal.AssetRegistryHelpers.get_asset_registry()
all_assets = ar.get_assets_by_path("/Game", recursive=True)

hits = []
for a in all_assets:
    try:
        cls = str(a.asset_class_path.asset_name)
    except AttributeError:
        cls = ""
    if cls != "StaticMesh":
        continue
    name = str(a.asset_name).lower()
    if any(tag in name for tag in ("wheel", "tire", "tyre", "rim")):
        hits.append((str(a.asset_name), str(a.package_name)))

unreal.log(f"[meshes] {len(hits)} wheel/tire/rim StaticMesh assets")
for name, pkg in hits[:50]:
    unreal.log(f"  {name}  ({pkg})")
