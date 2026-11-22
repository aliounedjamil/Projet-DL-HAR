import sys
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebook(nb_path, working_dir):
    print(f"\n=======================================================")
    print(f"Lancement de l'exécution réelle de : {nb_path}")
    print(f"Répertoire de travail : {working_dir}")
    print(f"=======================================================")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Preprocessor avec timeout généreux (30 min max)
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": working_dir}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"Succès ! Le notebook {nb_path} a été exécuté et sauvegardé avec toutes ses sorties réelles !")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "v1"
    
    if target == "v1":
        execute_notebook(
            "notebooks/version1_alioune/v1_baseline_mlp_alioune.ipynb",
            "notebooks/version1_alioune"
        )
    elif target == "v2":
        execute_notebook(
            "notebooks/version2_madjiguene/v2_deep_mlp_madjiguene.ipynb",
            "notebooks/version2_madjiguene"
        )
    elif target == "all":
        execute_notebook(
            "notebooks/version1_alioune/v1_baseline_mlp_alioune.ipynb",
            "notebooks/version1_alioune"
        )
        execute_notebook(
            "notebooks/version2_madjiguene/v2_deep_mlp_madjiguene.ipynb",
            "notebooks/version2_madjiguene"
        )

