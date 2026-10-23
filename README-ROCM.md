# ROCm / AMD Radeon changes to SceneWorks inference

A modified copy of [`SceneWorks/inference`](https://github.com/SceneWorks/inference)
(Apache-2.0) adding AMD Radeon support for SeedVR2 via ROCm, plus SeedVR2
performance work that is not AMD-specific.

Published at Michael Trefry's request so the changes can be pulled into
upstream. Companion to the SceneWorks-side plumbing at
[`Prog-Newb/SceneWorks`](https://github.com/Prog-Newb/SceneWorks), branch
`amd-rocm-0.8.7`.

Validated on an RX 7900 XTX (gfx1100), ROCm 7.2.1, Windows.

## Read this first: no shared history

**This repo was created with `git init` over a copied tree, not by forking.** It
has no commit in common with `SceneWorks/inference`, so `git log` comparisons
against upstream will not work and the root commit shows all 3,459 files as
additions.

Diffing still works, because tree-to-tree diffs do not need common ancestry, and
everything that matters lives in two directories. See "How to extract this"
below.

The root commit (`be669fc`, "Baseline: inference tree as received") is the
inference tree as it stood at the revision SceneWorks `release/0.8.7` pins,
`0e4adc6f05884e6891e29dfecd32ee695efe8b18`. **That correspondence is asserted,
not proven** -- without shared history nothing here verifies it. Confirm it by
diffing trees before trusting it.

Upstream `main` has since moved to `d47ce76f`, roughly 3,400 commits ahead, so
expect to port rather than merge.

## What changed, and where

Only two directories:

| Path | What |
|---|---|
| `crates/bundles/runtime-rocm/` | New bundle. A `runtime-cuda` sibling implementing the same `gen-core` contract, catalog restricted to SeedVR2. 3 files. |
| `crates/media/candle-gen/candle-gen-seedvr2/` | The engine work: `conv3d.rs`, `nn.rs`, `pipeline.rs`, `vae.rs`, `video.rs`, `lib.rs`, `debug.rs`, plus a `rocm_video_smoke.rs` test. |

Ignore `Cargo.lock` and `.gitignore` in these commits -- they are resolved
against the 0.8.7-era tree and will be wrong for you.

## The commits worth taking

Oldest first. The middle four are the performance work and are **not
AMD-specific** -- they should help CUDA too, with the caveat noted below.

| Commit | What it does |
|---|---|
| `be669fc` | Baseline. Also contains pre-existing ROCm `candle-gen-seedvr2` work and backend fixes folded in, so diff its two crate directories rather than reading it as pure upstream. |
| `aba2098` | Add the `runtime-rocm` bundle. |
| `a3e0b64` | Mirror `runtime-cuda`'s public surface for compile compatibility; keep the catalog SeedVR2-only. |
| `37b209f` | Pad output dims to a `VAE_SCALE` multiple before VAE encode, crop back via `decode_crop`. |
| `95f659b` | Add `rocm_video_smoke`, a functional smoke test for ROCm SeedVR2 video upscale. |
| `2959307` | **Load f16, not bf16.** bf16 was destroying the output on this path. Worth checking whether the same latent bug exists on CUDA. |
| `6211c6e` | **Rectangular spatial tiles**, plus per-tile stage timing. Portable. |
| `1e0bfbd` | Cleanup only -- drops a `_backup_before_cleanup_2026-10-09/` folder. Skip it. |
| `81a4aac` | **Collapse the conv3d tap loop when `T == 1`**, and stop copying weights to the CPU every forward. Portable, and the second half is a straight win. |
| `724387c` | Opt-in per-block VAE timing. Diagnostic only. |
| `65cbdb2` | **GroupNorm through the fused `layer_norm` kernel.** See the warning below. |

Together these took 1440p video from the first working build to 15.3 s/frame,
a 4.0x improvement, measured on gfx1100 with the ROCm candle fork in place.
That figure is for this repo plus that fork, not for either alone.

### Warning on `65cbdb2`

The GroupNorm fusion **changes numerics**. It was validated only by output
inspection on ROCm, never against a CUDA reference. Take it last, separately,
and check it against your own reference output before letting it near the CUDA
lane. Everything above it is safe to evaluate independently.

## How to extract this

```sh
git remote add rocm https://github.com/Prog-Newb/Inference-Rocm.git
git fetch rocm

# the whole change surface, scoped to the two directories that matter
git diff <your-base-rev>:crates/media/candle-gen/candle-gen-seedvr2 \
         rocm/amd-rocm:crates/media/candle-gen/candle-gen-seedvr2

git diff <your-base-rev>:crates/bundles \
         rocm/amd-rocm:crates/bundles

# or take commits one at a time -- cherry-pick does not need shared ancestry
git cherry-pick 81a4aac      # conv3d T==1 collapse
```

## What is NOT here

The ROCm backend itself. `runtime-rocm` builds against a ROCm-backend fork of
candle 0.10.2 (`candle-core/src/rocm_backend` plus `candle-rocm-kernels`, which
carries the `.hip` kernels). Upstream candle has no ROCm backend at all.

That fork is **not published**. 17 of its 25 commits are a third-party ROCm port
of candle by another author; 8 are mine (performance work and the dtype fix).
candle is MIT / Apache-2.0. Anyone wanting to publish it should preserve that
attribution in history rather than squashing it.

So nothing in this repo builds for ROCm without that fork. The SeedVR2
performance commits, however, are plain candle and should build and be
measurable on CUDA as they are.

## License

Apache-2.0, as upstream. `LICENSE` is unmodified.

Per Apache-2.0 section 4(b), this is the statement of changes: files under
`crates/media/candle-gen/candle-gen-seedvr2/` were modified and
`crates/bundles/runtime-rocm/` was added, as described above. All modifications
are by Chris (GitHub: Prog-Newb) except where noted, and the ROCm candle backend
this depends on is third-party work credited above.
