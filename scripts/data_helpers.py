from datasets import load_dataset, Dataset
import os

class Data:
    @staticmethod
    def json_to_parquet(json_path: str, output_path: str = None):
        """Convert a JSONL file to Parquet format"""
        dataset = load_dataset("json", data_files=json_path, split="train")
        output_path = output_path or json_path.replace(".json", ".parquet")
        dataset.to_parquet(output_path)
        print(f"✅ Saved Parquet to: {output_path}")

    @staticmethod
    def parquet_to_json(parquet_path: str, output_path: str = None, lines: bool = True):
        """Convert a Parquet file to JSONL (or JSON)"""
        dataset = load_dataset("parquet", data_files=parquet_path, split="train")
        ext = ".jsonl" if lines else ".json"
        output_path = output_path or parquet_path.replace(".parquet", ext)
        dataset.to_json(output_path, orient="records", lines=lines)
        print(f"✅ Saved JSON to: {output_path}")

    @staticmethod
    def load(path: str) -> Dataset:
        """Auto-load JSONL or Parquet"""
        ext = os.path.splitext(path)[1]
        if ext == ".json" or ext == ".jsonl":
            return load_dataset("json", data_files=path, split="train")
        elif ext == ".parquet":
            return load_dataset("parquet", data_files=path, split="train")
        else:
            raise ValueError("❌ Unsupported file extension: " + ext)
