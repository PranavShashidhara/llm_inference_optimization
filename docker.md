# Docker Setup — Jetson Orin

Replaces `setenv.sh`. Two commands cover all cases.

---

## Commands

### Build image + start container (first time, or after Dockerfile changes)
```bash
docker compose up --build -d
```

### Start existing container as-is (no rebuild)
```bash
docker compose up -d
```

To drop into a shell inside the running container:
```bash
docker exec -it llm-inference-orin bash
```

To stop:
```bash
docker compose down
```

---

## VS Code — Attach to running container

1. Start the container with either command above.
2. Open VS Code → **Remote Explorer** (sidebar) → **Dev Containers**.
3. Find `llm-inference-orin` → click **Attach**.
4. The `.devcontainer/devcontainer.json` is picked up automatically and installs extensions.

Alternatively: open the Command Palette → `Dev Containers: Attach to Running Container`.

---

## What's mounted

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `.` (entire repo) | `/workspace` | Live bind — edits on host are instant inside container |
| `huggingface-cache` volume | `/root/.cache/huggingface` | Model weights persist across container rebuilds |

---

## GPU verification

After attaching, the terminal will auto-run:
```
CUDA: True | Device: Orin
```

Or run manually:
```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Notes

- The base image (`nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3`) already includes PyTorch with CUDA — `requirements.txt` lines starting with `torch` are skipped during the build to avoid conflicts.
- `network_mode: host` is used so the container can reach the internet for HuggingFace downloads without extra port mapping.
- `shm_size: 2gb` avoids shared-memory errors if you later add DataLoader workers.
- If your JetPack version differs, swap the base image tag at the top of `Dockerfile`. Find available tags at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch