#IF PHSYICS NEMO BREAK DO THIS IN TERMINAL


python3 - <<'EOF'
path = "/lustre09/project/6003252/htruchla/physicsnemo/physicsnemo/datapipes/cae/__init__.py"
with open(path) as f:
    content = f.read()
content = content.replace(
    "from .domino_datapipe import DoMINODataPipe",
    "try:\n    from .domino_datapipe import DoMINODataPipe\nexcept ImportError:\n    pass"
)
content = content.replace(
    "from .mesh_datapipe import MeshDatapipe",
    "try:\n    from .mesh_datapipe import MeshDatapipe\nexcept ImportError:\n    pass"
)
with open(path, "w") as f:
    f.write(content)
print("Done")
EOF