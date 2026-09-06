import pathlib, sys
p = pathlib.Path("lexicon.yaml"); src = p.read_text()
old = """  - name: immunosuppressed_state
    label: Transplant / immunosuppressed state
    entity_types: [DISEASE]"""
new = """  - name: immunosuppressed_state
    label: Transplant / immunosuppressed state
    entity_types: [DISEASE, PROCEDURE]"""
if src.count(old) != 1:
    sys.exit(f"anchor matched {src.count(old)} times")
p.write_text(src.replace(old, new))
print("ok  immunosuppressed_state: allow PROCEDURE spans")
