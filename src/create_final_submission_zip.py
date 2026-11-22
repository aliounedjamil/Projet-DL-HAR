import zipfile
import os

zip_paths = [
    "deliverables/M2_DL_2026_DIOP_FALL.zip",
    "M2_DL_2026_DIOP_FALL.zip"
]

include_dirs = ["src", "notebooks", "reports", ".vscode"]
include_files = ["README.md", "requirements.txt", ".gitignore"]

for zip_filename in zip_paths:
    os.makedirs(os.path.dirname(zip_filename) if os.path.dirname(zip_filename) else ".", exist_ok=True)
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Fichiers racine
        for f in include_files:
            if os.path.exists(f):
                zipf.write(f, f)
                
        # Dossiers
        for d in include_dirs:
            for root, _, files in os.walk(d):
                for file in files:
                    filepath = os.path.join(root, file)
                    # Exclure __pycache__ et fichiers temporaires
                    if "__pycache__" not in filepath and not filepath.endswith(".pyc"):
                        zipf.write(filepath, filepath)

        # Inclure .gitkeep de data/HAR
        if os.path.exists("data/HAR/.gitkeep"):
            zipf.write("data/HAR/.gitkeep", "data/HAR/.gitkeep")

    print(f"Archive créée : {zip_filename} ({os.path.getsize(zip_filename) / (1024*1024):.2f} Mo)")
