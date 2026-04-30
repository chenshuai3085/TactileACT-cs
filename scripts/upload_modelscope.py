"""Upload latest 2 checkpoints from each DP experiment to ModelScope (parallel)."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from modelscope.hub.api import HubApi

# === Config ===
TOKEN = "ms-a69fa6d7-5d49-4da8-84d9-2f2531a0c45f"
MODEL_ID = "chenshuai3085/dp-ckpt-02090210"
MAX_WORKERS = 6  # parallel upload threads

EXPERIMENTS = {
    "dp_dagger": "/sharedata/chenshuai/ckpt/dp_dagger_02090210",
    "dp_only_vision_onlysuccess": "/sharedata/chenshuai/ckpt/dp_only_vision_onlysuccess_02090210",
    "dp_only_vision_all": "/sharedata/chenshuai/ckpt/dp_only_vision_02090210",
}

TOP_N = 2  # upload latest N checkpoints per experiment


def get_latest_pth(dir_path, n=2):
    """Get the N most recently modified .pth files in a directory."""
    pth_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pth')]
    pth_files.sort(key=os.path.getmtime, reverse=True)
    return pth_files[:n]


def upload_one_file(fpath, repo_path):
    """Upload a single file to ModelScope. Each thread creates its own API instance."""
    api = HubApi()
    api.login(TOKEN)
    size_gb = os.path.getsize(fpath) / (1024**3)
    print(f"  [START] {repo_path} ({size_gb:.2f} GB)")
    try:
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=repo_path,
            repo_id=MODEL_ID,
            repo_type='model',
            commit_message=f'Upload {repo_path}',
        )
        print(f"  [DONE]  {repo_path}")
        return (repo_path, True, None)
    except Exception as e:
        print(f"  [ERROR] {repo_path}: {e}")
        return (repo_path, False, str(e))


def main():
    api = HubApi()
    api.login(TOKEN)
    print(f"Logged in. Uploading to: {MODEL_ID}")

    # Create repo if not exists
    try:
        api.create_model(model_id=MODEL_ID, visibility=1)  # 1=private
        print(f"Created model repo: {MODEL_ID}")
    except Exception as e:
        if "already exist" in str(e).lower() or "exists" in str(e).lower():
            print(f"Repo {MODEL_ID} already exists, skipping creation.")
        else:
            print(f"Warning creating repo: {e}")

    # Collect all upload tasks
    upload_tasks = []
    for exp_name, exp_dir in EXPERIMENTS.items():
        if not os.path.isdir(exp_dir):
            print(f"[SKIP] {exp_dir} not found")
            continue

        latest_files = get_latest_pth(exp_dir, TOP_N)

        # Also upload config.json if exists
        config_path = os.path.join(exp_dir, 'config.json')
        if os.path.exists(config_path):
            latest_files.append(config_path)

        for fpath in latest_files:
            fname = os.path.basename(fpath)
            repo_path = f"{exp_name}/{fname}"
            upload_tasks.append((fpath, repo_path))

    print(f"\nTotal files to upload: {len(upload_tasks)}")
    print(f"Using {MAX_WORKERS} parallel threads\n")

    # Parallel upload
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_one_file, fpath, rpath): rpath
                   for fpath, rpath in upload_tasks}
        for future in as_completed(futures):
            results.append(future.result())

    # Summary
    success = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print(f"\n{'='*50}")
    print(f"Upload complete: {len(success)} success, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for rpath, _, err in failed:
            print(f"  {rpath}: {err}")
    print(f"\nModel: https://modelscope.cn/models/{MODEL_ID}")


if __name__ == '__main__':
    main()
