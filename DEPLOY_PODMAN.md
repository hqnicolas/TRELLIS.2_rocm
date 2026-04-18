# Podman Deployment

This repo is transport-ready for a Debian host. The transported bundle is this repo itself, including the root-level `Dependency/` directory.

## Recommended Workflow

1. Build and start the container:

   ```bash
   export TRELLIS_HOST_DIR="$(pwd)"
   export HF_TOKEN=hf_xxx
   podman-compose -f deploy/podman/podman-compose.yml up --build -d
   ```

2. Install the ROCm 7.2.2 Python stack and TRELLIS dependencies inside the container:

   ```bash
   podman exec -it trellis-rocm72 bash -lc 'cd /AI/TRELLIS.2_rocm && bash deploy/podman/install_trellis_rocm.sh'
   ```

3. Verify the environment:

   ```bash
   podman exec -it trellis-rocm72 bash -lc 'cd /AI/TRELLIS.2_rocm && bash deploy/podman/verify_rocm.sh'
   ```

4. Run the app:

   ```bash
   podman exec -it trellis-rocm72 bash -lc 'cd /AI/TRELLIS.2_rocm && bash deploy/podman/run_trellis.sh'
   ```

5. Optionally verify the Gradio endpoint after startup:

   ```bash
   podman exec -it trellis-rocm72 bash -lc 'cd /AI/TRELLIS.2_rocm && bash deploy/podman/verify_rocm.sh --require-info'
   ```

## Hugging Face Troubleshooting

If TRELLIS fails with an error like `401 Client Error`, `Unauthorized`, or `You are trying to access a gated repo`, the container reached Hugging Face successfully but does not have permission to read `facebook/dinov3-vitl16-pretrain-lvd1689m`.

Checklist:

- Confirm your Hugging Face account has access to the gated DINOv3 repo.
- Confirm `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is exported on the Debian host.
- If the container was started before you exported the token, either recreate it with `podman-compose up --build -d` or pass `-e HF_TOKEN="$HF_TOKEN"` to `podman exec`.
- If you want a fully offline startup path for the image encoder, place the model on disk and export `TRELLIS_IMAGE_COND_MODEL` to that local directory.
- Run `podman exec -it trellis-rocm72 bash -lc 'cd /AI/TRELLIS.2_rocm && bash deploy/podman/diagnose_gpu.sh'` to confirm whether the token and override env vars are visible inside the container.

## Rootless Podman Troubleshooting

If `run_trellis.sh` or `verify_rocm.sh` reports `No HIP GPUs are available`, the usual cause is that the container can see `/dev/kfd` and `/dev/dri` only partially or with the wrong user/group mapping.

Run this inside the container first:

```bash
podman exec -it trellis-rocm72 bash -lc 'cd /AI/TRELLIS.2_rocm && bash deploy/podman/diagnose_gpu.sh'
```

On the Debian host, also check:

```bash
id
podman info --format "{{.Host.Security.Rootless}} {{.Host.OCIRuntime.Name}}"
ls -l /dev/kfd /dev/dri/render*
```

What to look for:

- Your host user should normally be in the `render` and `video` groups.
- Rootless Podman GPU access depends on supplementary groups being preserved, and Podman documents `keep-groups` as only available with the `crun` OCI runtime.
- If the container shows `/dev/kfd` or `/dev/dri/render*` as inaccessible, or `rocminfo` inside the container shows no GPU agents, rebuild the container in rootful mode as the fastest fallback:

```bash
podman-compose -f deploy/podman/podman-compose.yml down
sudo env TRELLIS_HOST_DIR="$PWD" podman-compose -f deploy/podman/podman-compose.yml up --build -d
sudo podman exec -it trellis-rocm72 bash -lc 'cd /AI/TRELLIS.2_rocm && bash deploy/podman/diagnose_gpu.sh'
```

If you want to stay rootless, make sure:

- `podman info` reports `Rootless=true`
- `podman info` reports `crun` as the OCI runtime
- your login user belongs to `render` and `video`
- you recreate the container after any group membership change by logging out and back in, then running `podman-compose ... up --build -d` again

## Default ROCm Values

- `TARGET_GFX=gfx1100`
- `GPU_ARCHS=gfx1100`
- `PYTORCH_ROCM_ARCH=gfx1100`
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- `ROCM_PATH=/opt/rocm`

Override them in the shell or compose environment if your Debian target needs different ROCm values.
