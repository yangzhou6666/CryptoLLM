import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ORIGINAL_REPO = Path("benchmarks/owasp_original")
OUTPUT_DIR = Path("benchmarks/owasp")


def edit_owasp_instance(path: Path, instance_id_str: str) -> str:
    file = path.read_text()
    file = file.replace(path.stem, "A")
    idx = file.find("import")
    assert idx != -1
    file = file[idx:]
    file = re.sub(r"@WebServlet\([^)]*\)", "", file)
    output_dir = OUTPUT_DIR / instance_id_str
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "A.java"
    output_file.write_text(file)
    return file


def git_init_and_commit(directory: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(directory), check=True)
    subprocess.run(["git", "add", "."], cwd=str(directory), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add files"], cwd=str(directory), check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotations_orig = pd.read_csv(ORIGINAL_REPO / "expectedresults-1.2.csv")
    annotations_orig = annotations_orig[
        annotations_orig[" category"].isin(["weakrand", "hash", "crypto"])
    ].reset_index(drop=True)

    annotations = dict()
    metadata = list()

    count = 0

    for _, row in annotations_orig.iterrows():
        test_name = row["# test name"]
        is_vul = row[" real vulnerability"]
        category = row[" category"]
        instance_id_str = f"i{test_name.replace('BenchmarkTest', '')}"

        file_path = ORIGINAL_REPO / f"src/main/java/org/owasp/benchmark/testcode/{test_name}.java"
        assert file_path.exists()
        if file_path.exists():
            annotations[instance_id_str] = dict()
            annotations[instance_id_str]["is_vulnerability"] = is_vul
            annotations[instance_id_str]["category"] = category

            metadata.append(
                dict(
                    id=instance_id_str,
                    repo_name=str(OUTPUT_DIR / instance_id_str),
                    problem_statement="",
                    image_name="sweagent-api-misuse",
                )
            )

            edit_owasp_instance(file_path, instance_id_str)

            git_init_and_commit(OUTPUT_DIR / instance_id_str)

            count += 1

    assert count == 975

    with open(OUTPUT_DIR / "annotations.yaml", "w") as f:
        yaml.dump(annotations, f)

    with open(OUTPUT_DIR / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


if __name__ == "__main__":
    main()
