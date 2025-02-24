import json
from pathlib import Path
import yaml

def process_owasp():
    owasp_dir = Path("benchmarks/owasp")
    annotations_path = owasp_dir / "annotations.yaml"
    
    with open(annotations_path, 'r') as f:
        annotations = yaml.safe_load(f)
    
    # Ensure the dataset directory exists
    dataset_dir = Path("./dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    output_path = dataset_dir / "owasp_benchmark.jsonl"
    with open(output_path, 'w') as f_out:
        for instance_id, data in annotations.items():
            instance_path = owasp_dir / instance_id
            java_file = instance_path / "A.java"
            if not java_file.exists():
                print(f"Warning: {java_file} not found. Skipping.")
                continue
            code = java_file.read_text()
            
            is_vul = data['is_vulnerability']
            if isinstance(is_vul, bool):
                label = 1 if is_vul else 0
            else:
                label = 1 if is_vul.lower() == 'true' else 0
            
            json_line = json.dumps({"code": code, "label": label})
            f_out.write(json_line + '\n')

def process_cryptoapibench():
    crypto_dir = Path("benchmarks/cryptoapibench")
    annotations_path = crypto_dir / "annotations.yaml"
    
    with open(annotations_path, 'r') as f:
        annotations = yaml.safe_load(f)
    
    # Ensure the dataset directory exists
    dataset_dir = Path("./dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    output_path = dataset_dir / "cryptoapibench_benchmark.jsonl"
    with open(output_path, 'w') as f_out:
        for instance_id, data in annotations.items():
            instance_path = crypto_dir / instance_id
            java_files = sorted(instance_path.glob("*.java"))
            if not java_files:
                print(f"Warning: No Java files found in {instance_path}. Skipping.")
                continue
            
            code = ""
            for java_file in java_files:
                code += java_file.read_text() + "\n"
            code = code.strip()
            
            is_vul = data['is_vulnerability']
            if isinstance(is_vul, bool):
                label = 1 if is_vul else 0
            else:
                label = 1 if is_vul.lower() == 'yes' else 0
            
            json_line = json.dumps({"code": code, "label": label})
            f_out.write(json_line + '\n')

if __name__ == "__main__":
    process_owasp()
    # process_cryptoapibench()
