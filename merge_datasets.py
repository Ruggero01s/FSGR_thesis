import shutil
from pathlib import Path

def merge_datasets(old_dir: Path, new_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists.")
    # 1) Copy entire old dataset
    shutil.copytree(old_dir, output_dir)

    # 2) Collect all problem‐folder names from the old dataset
    old_ids = {p.name for p in old_dir.rglob("p*") if p.is_dir()}
    old_count = len(old_ids)
    
    # print(old_ids)

    # 3) Walk new dataset and copy only unseen problems
    added_count = 0
    for problem_dir in new_dir.rglob("p*"):
        if not problem_dir.is_dir(): 
            continue
        pid = problem_dir.name
        if pid in old_ids:
            continue
        rel = problem_dir.relative_to(new_dir)
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(problem_dir, dest)
        added_count += 1

    print(f"Merged complete: {old_count} existing problems found, {added_count} new problems added.")
    return old_count, added_count

def main():
    old_dir = Path("datasets/gr_logistics/problems")
    new_dir = Path("code/generated_gr_dataset/logistics")
    output_dir = Path("merged_dataset")

    old_count, added_count = merge_datasets(old_dir, new_dir, output_dir)
    print(f"Summary:")
    print(f"  Old problems count : {old_count}")
    print(f"  New problems added  : {added_count}")

if __name__ == "__main__":
    main()