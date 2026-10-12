//! Real-checkpoint ROCm video smoke for the SeedVR2 video upscale path (epic 4811 / sc-5928, AMD).
//!
//! `#[ignore]` by default (needs the weights + a ROCm build). Run on the RX 7900 XTX with the
//! ROCm env recipe (see amd/HANDOFF.md) then:
//! ```text
//! $env:SEEDVR2_CKPT = "C:\SceneWorks-AMD\data\models"
//! cargo test -p candle-gen-seedvr2 --features rocm --release --test rocm_video_smoke -- --ignored --nocapture
//! ```
//! Optional: SEEDVR2_DTYPE=f16|bf16|f32 (default f16, matching the image smoke).
//!
//! A functional check that SeedVR2 video mode runs end to end on ROCm and produces a
//! frame-count-preserving, finite, structurally faithful upscale. Mirrors the CUDA video
//! chunk/overlap smoke but trimmed to one GPU run (the detailed seam/coherence asserts live in
//! cuda_video_smoke.rs; the model math is backend-neutral once generate_video runs).

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{imageops, Image};
use candle_gen_seedvr2::config::DitConfig;
use candle_gen_seedvr2::pipeline::Seedvr2Pipeline;
use candle_gen_seedvr2::video;

const DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";

/// Deterministic structured frame (gradient + checkerboard + rings) translated by (dx,dy) so the
/// clip has coherent motion for the temporal path to act on.
fn synth_frame(side: usize, dx: i32, dy: i32) -> Image {
    let mut pixels = vec![0u8; side * side * 3];
    let s = side as i32;
    for y in 0..side {
        for x in 0..side {
            let i = (y * side + x) * 3;
            let sx = ((x as i32 - dx).rem_euclid(s)) as usize;
            let sy = ((y as i32 - dy).rem_euclid(s)) as usize;
            let check = (((sx / 12) + (sy / 12)) % 2) as u8 * 90;
            let cx = side as f32 / 2.0;
            let dr = (((sx as f32 - cx).powi(2) + (sy as f32 - cx).powi(2)).sqrt() * 0.18).sin();
            pixels[i] = (sx * 255 / side) as u8;
            pixels[i + 1] = (40 + check as usize).min(255) as u8;
            pixels[i + 2] = (((dr + 1.0) * 0.5) * 255.0) as u8;
        }
    }
    Image { width: side as u32, height: side as u32, pixels }
}

fn synth_clip(side: usize, n: usize, motion: i32) -> Vec<Image> {
    (0..n)
        .map(|t| synth_frame(side, t as i32 * motion, (t as i32 * motion) / 2))
        .collect()
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt()).max(1e-12)
}

fn load_pipe() -> Option<Seedvr2Pipeline> {
    let ckpt = match std::env::var("SEEDVR2_CKPT") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: set SEEDVR2_CKPT to a numz/SeedVR2_comfyUI checkpoint dir");
            return None;
        }
    };
    let dtype = match std::env::var("SEEDVR2_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f32") => DType::F32,
        _ => DType::F16,
    };
    let device = candle_gen::default_device().expect("ROCm device");
    eprintln!(
        "[seedvr2-rocm-video] device={device:?} dtype={dtype:?} budget={:.1} GiB ckpt={ckpt}",
        video::safe_budget_gib()
    );
    let cfg = DitConfig::seedvr2_3b();
    Some(Seedvr2Pipeline::load(&ckpt, DIT_FILE, &cfg, dtype, &device).expect("load pipeline"))
}

#[test]
#[ignore = "real SeedVR2 ROCm video smoke: needs SEEDVR2_CKPT weights + a ROCm build"]
fn seedvr2_rocm_video_2x() {
    let Some(pipe) = load_pipe() else { return };

    // 128 -> 256 keeps the per-frame cost low while 256 stays a multiple of 16 (VAE_SCALE).
    let (src, tgt) = (128usize, 256usize);
    let n = 6usize;
    let clip = synth_clip(src, n, 2); // <=2 px/frame coherent motion

    let t0 = std::time::Instant::now();
    let mut steps: Vec<(usize, usize)> = Vec::new();
    let out = {
        let mut on_step = |done: usize, total: usize| steps.push((done, total));
        pipe.generate_video(&clip, tgt, tgt, 42, 0.0, None, None, Some(&mut on_step))
            .expect("generate_video on ROCm")
    };
    eprintln!(
        "[seedvr2-rocm-video] {n}x{src} -> {tgt} in {:?} -> {} frames, progress={steps:?}",
        t0.elapsed(),
        out.len()
    );

    // Frame count preserved, dims correct, no degenerate (constant) frame.
    assert_eq!(out.len(), n, "frame count must be preserved");
    for (i, f) in out.iter().enumerate() {
        assert_eq!((f.width, f.height), (tgt as u32, tgt as u32), "frame {i} dims");
        let mn = *f.pixels.iter().min().unwrap();
        let mx = *f.pixels.iter().max().unwrap();
        assert!(mx > mn, "frame {i} is constant (degenerate): min={mn} max={mx}");
    }

    // Structural faithfulness: each output frame correlates with the bicubic upscale of its LR frame.
    let mut min_corr = 1.0f64;
    for (i, f) in out.iter().enumerate() {
        let base = imageops::resize_bicubic_u8(&clip[i].pixels, src, src, tgt, tgt).unwrap();
        let of: Vec<f32> = f.pixels.iter().map(|&v| v as f32).collect();
        min_corr = min_corr.min(pearson(&of, &base));
    }
    eprintln!("[seedvr2-rocm-video] min per-frame corr_vs_bicubic = {min_corr:.4}");
    assert!(min_corr > 0.7, "a frame is not structurally faithful (corr={min_corr:.4})");
}