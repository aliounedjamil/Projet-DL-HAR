import zipfile
import os

zip_filename = "deliverables/Projet_DL_Alioune_DIOP.zip"
os.makedirs("deliverables", exist_ok=True)

# Liste des répertoires et fichiers à inclure
include_dirs = ["src", "notebooks", "reports", ".vscode"]
include_files = ["README.md", "requirements.txt", ".gitignore"]

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    # Fichiers racine
    for f in include_files:
        if os.path.exists(f):
            zipf.write(f, f)
            print(f"Ajouté : {f}")
            
    # Dossiers
    for d in include_dirs:
        for root, _, files in os.walk(d):
            for file in files:
                filepath = os.path.join(root, file)
                # On évite d'inclure les caches pyc
                if "__pycache__" not in filepath and not filepath.endswith(".pyc"):
                    zipf.write(filepath, filepath)
                    print(f"Ajouté : {filepath}")

    # On inclut le .gitkeep de data/HAR
    if os.path.exists("data/HAR/.gitkeep"):
        zipf.write("data/HAR/.gitkeep", "data/HAR/.gitkeep")

print(f"\nFichier ZIP créé avec succès : {zip_filename}")
print(f"Taille du ZIP : {os.path.getsize(zip_filename) / (1024*1024):.2f} Mo")
