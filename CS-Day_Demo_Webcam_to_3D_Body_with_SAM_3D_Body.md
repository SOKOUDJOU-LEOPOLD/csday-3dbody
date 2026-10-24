# CS Day Demo: Webcam to 3D Body with SAM 3D Body

Oct 23, 2026 · @SOKOUDJOU Leopold

## Goal

Build a one-page web app for CS Day where a student stands in front of the laptop camera, presses one button, and a few seconds later sees a 3D mesh of their own body that they can spin around with the mouse.

**Model choice, read this first.** SAM 3 on its own is a segmentation model: it returns masks, not 3D bodies. The model that turns one photo of a person into a full-body 3D mesh is **SAM 3D Body** (`facebookresearch/sam-3d-body`, checkpoint `facebook/sam-3d-body-dinov3`), released by Meta alongside SAM 3. It is a promptable single-image human mesh recovery model whose output is a Momentum Human Rig (MHR) mesh covering body, hands and feet ([model card](https://huggingface.co/facebook/sam-3d-body-dinov3)). SAM 3 can optionally serve as its person detector, which matches Meta's online playground ([repo](https://github.com/SimonR99/cambody)). So the pipeline is: webcam photo, person detection (default detector or SAM 3), SAM 3D Body, mesh shown in the browser.

The demo screen shows three things side by side: the captured photo, the mesh rendered over the photo, and an interactive 3D viewer of the mesh.

## Architecture

&#91;embedded content: demo architecture · laptop and GPU server\]

The browser sends one JPEG through the SSH tunnel; the server returns a GLB mesh and an overlay image, and the page shows both.

## Prerequisites

SAM 3D Body needs Linux and an NVIDIA GPU ([Rerun example](https://rerun.io/examples/video-image/sam3d_body)), so the model will almost certainly not run on the presenting laptop. The laptop only runs the browser and the camera; inference runs on a GPU machine (a Vision Lab workstation, or a TAMU HPRC node if the lab box is unavailable).

| Item | What is needed | Who does it |
| --- | --- | --- |
| Checkpoint access | Request access to `facebook/sam-3d-body-dinov3` on Hugging Face (gated, SAM License). If SAM 3 is used as the detector, also request `facebook/sam3` | You, before handing off |
| Hugging Face token | `hf auth login` or `HF_TOKEN` env var on the GPU machine | You |
| GPU machine | Linux, NVIDIA GPU, CUDA drivers, conda or uv | You provide; agent sets up |
| Python env | Exactly as in the repo's `INSTALL.md`; do not guess versions | Agent |
| Presenting laptop | Chrome, a working webcam, SSH access to the GPU machine | You |
| Network at the venue | Must reach the GPU machine; have a phone hotspot as backup | You test on the day before |

Why the SSH tunnel matters: browsers only allow camera access (`getUserMedia`) on HTTPS pages or on `localhost`. Forwarding the server port over SSH (`ssh -L 8000:localhost:8000 user@gpu-box`) lets the laptop open `http://localhost:8000`, so the camera works with no certificates to manage.

## Backend spec (FastAPI on the GPU machine)

One Python process loads the model once at startup and serves both the API and the static frontend. Use FastAPI with Uvicorn, `trimesh` for mesh export, OpenCV for image handling.

**Model loading.** Clone `facebookresearch/sam-3d-body`, follow `INSTALL.md`, and download weights with `hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3`. In the server, put the repo root on `sys.path` and load the estimator the way the model card shows ([model card](https://huggingface.co/facebook/sam-3d-body-dinov3)):

```python
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
# per request:
outputs = estimator.process_one_image(img_rgb)   # list, one dict per person
overlay = visualize_sample_together(img_bgr, outputs, estimator.faces)
```

Each person dict contains `pred_vertices` (camera coordinates), `pred_keypoints_3d`, `pred_keypoints_2d`, `pred_cam_t`, `focal_length`, and body, hand and shape parameters ([mirror card](https://huggingface.co/jetjodh/sam-3d-body-dinov3)). Run one dummy image through the model at startup so the first student does not wait for CUDA warmup. Make the detector configurable (default detector vs SAM 3) via an env var, and check `demo.py` in the repo for how each is wired.

**Endpoints.**

| Method and path | Input | Output |
| --- | --- | --- |
| `GET /` | none | `static/index.html` |
| `GET /api/health` | none | `{"ready": true, "gpu": "<name>"}` once warmup is done |
| `POST /api/reconstruct` | multipart field `image` (JPEG from the webcam) | JSON below |

`/api/reconstruct` returns `{"people": n, "overlay_png": "<base64>", "mesh_url": "/meshes/<id>.glb", "elapsed_ms": t}`, or HTTP 422 with `{"error": "no_person"}` when nothing is detected.

**Processing steps for one request.**

1. Decode the JPEG, convert to RGB, downscale so the long side is at most 1280 px.
2. Run `process_one_image`. If zero people, return the 422 above.
3. Keep the person with the largest 2D keypoint bounding box (the student in front), unless a `?all=1` flag is passed.
4. Build a `trimesh.Trimesh(vertices=pred_vertices, faces=estimator.faces)`. Rotate 180 degrees about the X axis because camera coordinates have Y pointing down, center it at the origin, and export to GLB in a temp folder. Give it one flat color so it looks good under default lighting.
5. Render the overlay with `visualize_sample_together`, encode as PNG, base64 it.
6. Return the JSON and log only timing, never the image.

**Concurrency and safety.** Guard inference with a single `asyncio.Lock` (one GPU, one request at a time) and run it in a thread pool so the server stays responsive. Reject uploads over 5 MB. Delete GLB files older than 10 minutes on a background timer. Do not write captured photos to disk.

## Frontend spec (one static page)

Plain HTML, CSS and vanilla JS in `static/`, no build step. For the 3D viewer use Google's `<model-viewer>` web component, which loads a GLB and gives orbit, zoom and auto-rotate for free. **Vendor the script into `static/vendor/`** instead of loading it from a CDN, so the demo still works if the venue Wi-Fi blocks outside traffic.

**Screen states.**

1. **Ready.** Large live mirror-flipped webcam preview with a dashed silhouette guide ("Stand so your whole body fits here"). One big button: **Take my picture**.
2. **Countdown.** 3, 2, 1 shown large over the video, then a white flash. Capture the *unflipped* frame from the `<video>` into a `<canvas>` and encode as JPEG quality 0.9.
3. **Processing.** Freeze the captured photo, show a spinner and a short line such as "Building your 3D body...". Disable the button.
4. **Result.** Three panels: the photo, the overlay PNG, and `<model-viewer src=mesh_url camera-controls auto-rotate>`. Buttons: **Try again** and **Spin faster** (optional fun). Auto-return to Ready after 60 seconds of no interaction.
5. **Error.** For `no_person`, show "I couldn't find a person. Step back so I can see more of you!" and a Try again button. For network or 500 errors, show a short message and keep the camera running.

**Camera details.** Request `getUserMedia({video: {width: {ideal: 1280}, height: {ideal: 720}, facingMode: "user"}})`. Add a camera picker dropdown (hidden behind a small gear icon) in case the laptop has an external webcam. Spacebar also triggers capture, which is handy for the presenter.

**Look and feel.** Full-screen, dark background, large type readable from two meters, TAMU maroon accent. A small footer: "Powered by Meta SAM 3D Body. Photos are not saved." Run the browser in kiosk or full-screen mode on the day.

## Repo structure and how to run

```
csday-3dbody/
  README.md            setup, run, and event-day steps
  environment.yml      or pyproject.toml, matching sam-3d-body INSTALL.md
  third_party/
    sam-3d-body/       git submodule, pinned to a known-good commit
  checkpoints/         gitignored, filled by hf download
  server/
    app.py             FastAPI app, endpoints, static mount
    model.py           estimator load, warmup, reconstruct(img) -> result
    mesh.py            vertices + faces -> centered, flipped GLB
    config.py          env vars: DETECTOR, PORT, MAX_UPLOAD_MB, MESH_TTL_S
  static/
    index.html
    app.js
    style.css
    vendor/model-viewer.min.js
  scripts/
    run_server.sh      activate env, start uvicorn on 0.0.0.0:8000
    tunnel.sh          ssh -L 8000:localhost:8000 user@gpu-box
    smoke_test.py      posts sample images, checks response shape and timing
  samples/             3 to 5 full-body test photos (team members, with consent)
```

**Run order on the day.** On the GPU machine: `scripts/run_server.sh`, wait for `/api/health` to report ready. On the laptop: `scripts/tunnel.sh`, then open `http://localhost:8000` in Chrome, allow the camera, press F11.

## Milestones for the agent

Work in this order and do not start a milestone until the previous one's check passes. Report back after each one.

| # | Milestone | Done when |
| --- | --- | --- |
| 1 | Environment and model | Repo's own `demo.py` runs on a sample photo on the GPU machine and writes a rendered output image |
| 2 | `model.py` + `mesh.py` | A script turns `samples/*.jpg` into GLB files that open correctly (upright, centered) in an online GLB viewer |
| 3 | API | `smoke_test.py` passes: valid JSON for every sample, 422 for an empty-room photo, server survives 10 back-to-back requests |
| 4 | Frontend | Full flow works at `http://localhost:8000` on the GPU machine itself: preview, countdown, result panels, error state, auto reset |
| 5 | Remote use | Same flow works on the presenting laptop through the SSH tunnel |
| 6 | Polish | Kiosk styling, vendored viewer, README with every command needed on the day |

**Rules for the agent.** Follow the upstream `INSTALL.md` instead of inventing package versions. Do not commit checkpoints or tokens. Do not save user photos anywhere. If something in the upstream API differs from this plan (function names, output keys), trust the repo and note the difference in the README. Measure and report real end-to-end latency rather than estimating it.

## Event day: privacy, fallbacks, checklist

**Privacy with minors.** Most visitors are high school students, so nothing is stored: photos live only in memory for the request, meshes are deleted after 10 minutes, and logs hold timing only. Put a small sign by the laptop saying so, and check whether the CS Day organizers require a photo notice. A "Download my 3D model" button is a nice extra, but only add it if organizers are fine with it.

**Fallbacks.** If the GPU machine or network fails, switch to a pre-recorded screen capture of the demo and the `samples/` results loaded from disk (add a `?offline=1` mode that shows them). Meta's own [web demo](https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d) is a last resort, though it needs internet and uploads the photo to Meta, so tell students before using it.

**Getting good results.** Full body in frame, about 2 to 3 meters from the camera, decent front lighting, plain background if possible. Tape a mark on the floor where students should stand.

- [ ] Checkpoint access approved on Hugging Face
- [ ] Full dry run on the venue network the day before
- [ ] Phone hotspot tested as backup network
- [ ] Laptop power, external webcam and tripod if available
- [ ] Floor mark and privacy sign
- [ ] Offline fallback loaded and tested
