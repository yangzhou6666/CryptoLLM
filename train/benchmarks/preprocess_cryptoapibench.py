import subprocess
from pathlib import Path

import pandas as pd
import yaml

ORIGINAL_REPO = Path("benchmarks/cryptoapibench_original")
OUTPUT_DIR = Path("benchmarks/cryptoapibench")


def get_max_prefix_length(s1: str, s2: str) -> int:
    min_length = min(len(s1), len(s2))
    prefix_length = 0
    for i in range(min_length):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break
    return prefix_length


def get_category(filename: str, categories: list[str]) -> str | None:
    filename = filename.lower()
    category, max_prefix_length = None, 0
    for c in categories:
        prefix_length = get_max_prefix_length(c, filename)
        if prefix_length > max_prefix_length:
            category = c
            max_prefix_length = prefix_length

    if category is None:
        assert filename.startswith("lessthan")
        return "pbeiteration"

    return category


def git_init_and_commit(directory: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(directory), check=True)
    subprocess.run(["git", "add", "."], cwd=str(directory), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add files"], cwd=str(directory), check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = []
    testcases_dir = ORIGINAL_REPO / "src/main/java/org/cryptoapi/bench"
    for e in testcases_dir.iterdir():
        if e.is_dir():
            categories.append(e.name)

    annotations_orig = pd.read_excel(ORIGINAL_REPO / "CryptoAPI-Bench_details.xlsx")
    instance_id = 1

    annotations = dict()
    metadata = list()

    for _, row in annotations_orig.iterrows():
        if not isinstance(row["Files"], str):
            continue

        file_names = row["Files"].split("\n")
        assert len(file_names) > 0

        category = get_category(file_names[0], categories)
        assert category is not None

        category_subtype = row["Type of Vulnerability"]
        is_vul = row["Vulnerability Exists?"]

        instance_id_str = f"i{instance_id:03}"
        cur_output_dir = OUTPUT_DIR / instance_id_str

        new_class_names = [f"A{i+1}" for i in range(len(file_names))]
        old_class_names = [f.replace(".java", "") for f in file_names]

        valid_instance = True
        for file_num, file_name in enumerate(file_names):
            if not file_name.endswith(".java"):
                file_name = file_name + ".java"
            path = testcases_dir / category / file_name
            if not path.is_file():
                valid_instance = False
                print(f"[WARNING] Failed to find file {category}/{file_name}")
                continue

            file = path.read_text()
            assert file.startswith("package") and file.find(";") != -1
            file = file[file.find(";") + 1 :]
            for ocn, ncn in zip(old_class_names, new_class_names):
                file = file.replace(ocn, ncn)

            new_path = cur_output_dir / f"{new_class_names[file_num]}.java"
            cur_output_dir.mkdir(parents=True, exist_ok=True)
            new_path.write_text(file.strip())

        if valid_instance:
            annotations[instance_id_str] = dict()
            annotations[instance_id_str]["category"] = category
            annotations[instance_id_str]["category_subtype"] = category_subtype
            annotations[instance_id_str]["is_vulnerability"] = is_vul

            metadata.append(
                dict(
                    id=instance_id_str,
                    repo_name=str(cur_output_dir),
                    problem_statement="",
                    image_name="sweagent-api-misuse",
                )
            )

            git_init_and_commit(cur_output_dir)

            instance_id += 1

    with open(OUTPUT_DIR / "annotations.yaml", "w") as f:
        yaml.dump(annotations, f)

    with open(OUTPUT_DIR / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


if __name__ == "__main__":
    main()
