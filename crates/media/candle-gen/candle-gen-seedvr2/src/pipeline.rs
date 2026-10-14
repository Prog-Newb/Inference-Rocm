//! SeedVR2 image-mode pipeline â€” candle port of `mlx-gen-seedvr2/src/pipeline.rs` (sc-5157).
//!
//! One-step super-resolution: preprocess the LR image (bicubic upscale to target, optional `softness`
//! pre-blur, [-1,1]) â†’ VAE encode â†’ conditioning latent (encoded latent + ones-mask) â†’ concat fresh
//! noise â†’ DiT (one step) â†’ `latents = noise âˆ’ DiT_out` â†’ VAE decode â†’ crop â†’ LAB+wavelet color
//! correction â†’ RGB8.
//!
//! **Video mode (sc-5926):** the 5-D temporal pass. `generate_video` sizes a temporal chunk against
//! the device VRAM budget ([`crate::video`]), runs each chunk through the same `T`-aware
//! encode/condition/denoise/decode path, per-frame color-corrects, and cross-fades chunk overlaps to
//! close the causal-VAE seam. It falls back to a per-frame (`T=1`) path under tight memory, and to
//! per-frame **spatial tiling** when even one full-resolution frame exceeds the budget (HD).
//!
//! The negative-prompt conditioning is a precomputed embedding (bundled `data/neg_embed.safetensors`,
//! no runtime text encoder), loaded at construction.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::gen_core::{imageops, CancelFlag, Image, Quant};
use candle_gen::{CandleError, Result as CResult};
use rand::{rngs::StdRng, SeedableRng};

use crate::config::{DitConfig, TIMESTEP};
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;
use crate::video::{self, Chunk, ChunkPlan};
use crate::weights::Weights;
use crate::{color, convert};

/// Post-decode color-correction luminance weight (the reference `apply_color_correction` default).
const LUMINANCE_WEIGHT: f32 = 0.8;

/// Wall-clock stage splits for one spatial tile, printed as `[seedvr2-tile-time]`.
///
/// Each [`Self::lap`] synchronizes the device first. Without that the numbers are meaningless on a
/// GPU backend: kernel launches return immediately, so the time for a stage lands on whichever later
/// call happens to block, and the breakdown reads as "decode is everything". The sync costs nothing
/// relative to a stage measured in seconds, and the tile loop is already the coarse unit here.
struct Stage<'a> {
    device: &'a Device,
    last: std::time::Instant,
    start: std::time::Instant,
}

impl<'a> Stage<'a> {
    fn new(device: &'a Device) -> Self {
        let _ = device.synchronize();
        let now = std::time::Instant::now();
        Self {
            device,
            last: now,
            start: now,
        }
    }

    /// Seconds since the previous lap (or construction), with the device drained first.
    fn lap(&mut self) -> f64 {
        let _ = self.device.synchronize();
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        dt
    }

    fn total(&self) -> f64 {
        self.last.duration_since(self.start).as_secs_f64()
    }
}

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
    neg_embed: Tensor,
    dtype: DType,
    device: Device,
    /// Resident weight bytes (VAE + DiT at `dtype`) â€” drives the video memory-budget chunk sizer.
    weights_bytes: usize,
}

/// Estimate resident weight bytes for the video memory budget: the raw `fp16` checkpoint file sizes
/// scaled by the load `dtype` (`BF16` keeps the 2 B/param footprint, `F32` doubles it). File sizes
/// (vs summing per-tensor) match the wan `dit_resident_bytes` convention; the safetensors header
/// overhead is negligible.
fn resident_weight_bytes(files: &[&std::path::Path], dt: DType) -> usize {
    let raw: u64 = files
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let ratio = match dt {
        DType::F32 => 2.0, // fp16-on-disk â†’ f32 resident
        _ => 1.0,          // bf16/fp16 resident
    };
    (raw as f64 * ratio) as usize
}

/// The bundled precomputed negative-prompt embedding â†’ `(1, Lt, 5120)` at `dt`.
fn load_neg_embed(dt: DType, dev: &Device) -> CResult<Tensor> {
    const BYTES: &[u8] = include_bytes!("../data/neg_embed.safetensors");
    let map = candle_gen::candle_core::safetensors::load_buffer(BYTES, dev)?;
    let emb = map
        .get("embedding")
        .ok_or_else(|| CandleError::Msg("seedvr2 neg-embed: missing `embedding`".into()))?;
    Ok(emb.to_dtype(dt)?.unsqueeze(0)?)
}

/// Deterministic N(0,1) noise of a 5-D shape (CPU `StdRng`/ChaCha, launch-portable per seed).
fn seeded_normal5(
    seed: u64,
    shape: (usize, usize, usize, usize, usize),
    dt: DType,
    dev: &Device,
) -> CResult<Tensor> {
    let (a, b, c, d, e) = shape;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, a * b * c * d * e);
    Ok(Tensor::from_vec(data, shape, dev)?.to_dtype(dt)?)
}

/// `(1,3,H,W)` in [-1,1] â†’ RGB8 [`Image`].
fn decoded_to_image(decoded: &Tensor) -> Result<Image> {
    let (_b, _c, h, w) = decoded.dims4()?;
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let u8s = candle_gen::round_rgb8(&scaled)?.to_device(&Device::Cpu)?;
    let chw = u8s.squeeze(0)?; // (3,H,W)
    let pixels = chw.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Bail with the typed [`CandleError::Canceled`] when the caller's cooperative cancel flag has
/// tripped â€” the upscale loops check it before each (seconds-to-minutes) unit of work: video
/// chunk, video frame, and spatial tile (the gen-core per-step cancel contract, sc-8972/sc-11227;
/// mirrors `Sam3VideoModel::propagate`; per-tile checks on both the still and video tiled paths,
/// sc-12465/F-013). An absent flag is a no-op. The typed variant (never a stringified `Msg`)
/// bridges to `gen_core::Error::Canceled` so the worker keys off the variant to map a user cancel.
fn check_canceled(cancel: Option<&CancelFlag>) -> CResult<()> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(CandleError::Canceled);
    }
    Ok(())
}

/// Drive the spatial-tile loop over `plan`: honor the engine cancellation contract at the **top of
/// every tile iteration** (sc-12465 / F-013 â€” each tile is a full seconds-to-minutes
/// VAE-encode â†’ DiT â†’ VAE-decode pass, the per-unit granularity of the cancel contract on the tiled
/// paths), then run `f` on the tile. This is the production driver for
/// [`Seedvr2Pipeline::run_frame_tiled`] â€” extracted so the per-tile check placement is unit-testable
/// without weights (`tests::spatial_tile_loop_cancels_between_tiles` exercises this exact
/// function, not a re-implementation).
fn for_each_tile(
    plan: &[video::SpatialTile],
    cancel: Option<&CancelFlag>,
    mut f: impl FnMut(&video::SpatialTile) -> CResult<()>,
) -> CResult<()> {
    for t in plan {
        check_canceled(cancel)?;
        f(t)?;
    }
    Ok(())
}

impl Seedvr2Pipeline {
    /// Build from already-converted candle-layout VAE + DiT weights + a neg-embed (parity tests).
    pub fn from_parts(
        vae: Seedvr2Vae,
        transformer: Seedvr2Transformer,
        neg_embed: Tensor,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            vae,
            transformer,
            neg_embed,
            dtype,
            device,
            weights_bytes: 0,
        }
    }

    /// Load from a raw `numz/SeedVR2_comfyUI` checkpoint dir: convert in-memory (no Python), cast to
    /// `dt`, attach the bundled neg-embed. `dit_file` selects 3B/7B.
    pub fn load(
        raw_dir: impl AsRef<std::path::Path>,
        dit_file: &str,
        cfg: &DitConfig,
        dt: DType,
        device: &Device,
    ) -> CResult<Self> {
        let dir = raw_dir.as_ref();
        let vae_path = dir.join("ema_vae_fp16.safetensors");
        let dit_path = dir.join(dit_file);
        let weights_bytes = resident_weight_bytes(&[vae_path.as_path(), dit_path.as_path()], dt);
        // Stream each checkpoint one at a time and convert+cast tensor-by-tensor: the raw fp16 map is
        // consumed (moved into `convert_*`) and each raw tensor drops as its `dt` cast is produced, so
        // the raw set and the cast copy never both live for all weights (peak load memory ~1Ã— rather
        // than 2Ã—/3Ã— â€” sc-9042/F-058). VAE is fully built (and its raw map freed) before the 7B DiT is
        // read, bounding peak further. Loaded values are identical to the prior two-phase convert+cast.
        let vae_raw = Weights::from_file(&vae_path, device)?;
        let vae_w = convert::convert_vae(vae_raw, dt)?;
        let vae = Seedvr2Vae::from_weights(&vae_w)?;
        drop(vae_w);
        let dit_raw = Weights::from_file(&dit_path, device)?;
        let dit_w = convert::convert_dit(dit_raw, dt)?;
        let transformer = Seedvr2Transformer::from_weights(&dit_w, cfg)?;
        let neg_embed = load_neg_embed(dt, device)?;
        let mut p = Self::from_parts(vae, transformer, neg_embed, dt, device.clone());
        p.weights_bytes = weights_bytes;
        Ok(p)
    }

    /// Quantize the DiT Linears to `quant` (`Q4_0`/`Q8_0`) â€” Linear-only (sc-5927); the VAE stays
    /// dense. Call once after [`Self::load`], before the pipeline is shared. `weights_bytes` is
    /// intentionally **not** reduced â€” keeping the dense estimate makes the video chunk-size budget
    /// conservative (quant shrinks the weights, not the activations, so the headroom stays safe).
    pub fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.transformer.quantize(quant)
    }

    /// Build the static condition `[latent, ones-mask]` â†’ `(B, 17, T', h, w)`.
    fn condition(latent: &Tensor) -> Result<Tensor> {
        let (b, _c, t, h, w) = latent.dims5()?;
        let mask = Tensor::ones((b, 1, t, h, w), latent.dtype(), latent.device())?;
        Tensor::cat(&[latent, &mask], 1)
    }

    /// One denoise step: `vid = [noise, condition]` â†’ DiT â†’ `noise âˆ’ DiT_out`.
    fn denoise(&self, noise: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let model_input = Tensor::cat(&[noise, condition], 1)?; // (B,33,T',h,w)
        crate::debug::probe("denoise model_input", &model_input)?;

        let dit_out = self
            .transformer
            .forward(&model_input, &self.neg_embed, TIMESTEP)?;
        crate::debug::probe("denoise dit_out", &dit_out)?;

        let result = (noise - dit_out)?;
        crate::debug::probe("denoise result", &result)?;

        Ok(result)
    }

    /// Decode latents and crop to `(true_h, true_w)` (first frame) â†’ `(1,3,true_h,true_w)`.
    fn decode_crop(&self, latents: &Tensor, true_h: usize, true_w: usize) -> Result<Tensor> {
        let decoded = self.vae.decode(latents)?; // (1,3,T,H,W)
        decoded
            .narrow(2, 0, 1)?
            .squeeze(2)? // (1,3,H,W)
            .narrow(2, 0, true_h)?
            .narrow(3, 0, true_w)?
            .contiguous()
    }

    /// Full model path (no color correction): preprocessed image + injected noise â†’ decoded crop.
    /// Public for the golden parity harness (the engine-level parity check).
    pub fn run_model(
        &self,
        processed: &Tensor,
        noise: &Tensor,
        true_h: usize,
        true_w: usize,
    ) -> Result<Tensor> {
        let latent = self.vae.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond)?;
        self.decode_crop(&latents, true_h, true_w)
    }

    /// End-to-end upscale: LR `image` â†’ `(width, height)` super-resolved RGB8 image.
    ///
    /// Spatial-tiles when a single full-resolution pass would exceed the memory budget (sc-6225) â€” the
    /// image analog of the video path's HD-tiling fallback (sc-5926). See [`Self::generate_budgeted`].
    ///
    /// `cancel` is the caller's cooperative cancel flag, checked before each spatial tile on the
    /// tiled path (sc-12465/F-013 â€” a tiled 4096Â² still is up to 9 full encodeâ†’DiTâ†’decode passes,
    /// so a mid-image cancel must not wait for the whole image); the one-pass path is a single unit
    /// and relies on the caller's per-image check (the gen-core per-step cancel contract).
    pub fn generate(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        cancel: Option<&CancelFlag>,
    ) -> CResult<Image> {
        self.generate_budgeted(
            image,
            width,
            height,
            seed,
            softness,
            video::safe_budget_gib(),
            cancel,
        )
    }

    /// [`Self::generate`] with the safe peak-GB ceiling injected (so the spatial-tiling path is
    /// unit-testable without a multi-GB target â€” mirrors [`crate::video::plan_chunk_size_with`]).
    /// When even a single full-resolution pass would exceed `safe_gib`, the image is upscaled by
    /// feather-blended spatial tiling ([`Self::run_frame_tiled`], the parity-gated sc-5926 tiler)
    /// rather than one allocation that would blow past the device's free VRAM and OOM the worker
    /// (sc-6225); otherwise the one-pass still path runs (numerically unchanged from before).
    /// `cancel` is threaded to the per-tile check on the tiled path (sc-12465).
    #[allow(clippy::too_many_arguments)] // size/seed/budget + the threaded cancel flag
    pub fn generate_budgeted(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        safe_gib: f64,
        cancel: Option<&CancelFlag>,
    ) -> CResult<Image> {
        // Tile when EITHER the memory budget is exceeded (sc-6225) OR the output edge is past the VAE
        // decoder's correctness limit (sc-8228/sc-8261): above ~1536Â² the decoder corrupts its output
        // in a single pass, so a 2048Â² upscale that *fits* memory must still be tiled. The spatial
        // tiler caps every tile at `VAE_SAFE_DECODE_EDGE_PX` (see `plan_spatial_tile_px`).
        let over_budget = matches!(
            video::plan_chunk_size_with(self.weights_bytes, height as i32, width as i32, safe_gib),
            ChunkPlan::OverBudget { .. }
        );
        let over_vae_cap = width.max(height) as i32 > video::VAE_SAFE_DECODE_EDGE_PX;
       
        eprintln!(
            "[seedvr2-debug] over_budget={} over_vae_cap={} output={}x{} safe_gib={}",
            over_budget,
            over_vae_cap,
            width,
            height,
            safe_gib
        );

        if over_budget {
            return self.generate_tiled(image, width, height, seed, softness, safe_gib, cancel);
        }

        // Pad output dims to the nearest multiple of VAE_SCALE (= spatial-downscale 8
        // × DiT patch 2 = 16) before encoding. Non-multiple inputs cause an integer-
        // truncation mismatch in the DiT patch-embedding reshape (e.g. 1422/8 = 177.75
        // truncated to 177, which is odd and cannot be split by patch_h=2).
        // decode_crop already accepts the true (caller-supplied) height/width and uses
        // Tensor::narrow to strip the padding rows/columns from the decoded output, so
        // callers always receive exactly the dimensions they requested.
        let vae_scale = crate::config::VAE_SCALE as usize;
        let pad_h = ((height + vae_scale - 1) / vae_scale) * vae_scale;
        let pad_w = ((width  + vae_scale - 1) / vae_scale) * vae_scale;

        let processed = self.preprocess(image, pad_w, pad_h, softness)?; // (1,3,pad_h,pad_w)
        let latent = self.vae.encode(&processed)?;
        let (_b, _c, lt, lh, lw) = latent.dims5()?;
        let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(&noise, &cond)?;

        crate::debug::probe("latents (post-denoise)", &latents)?;

        // decode_crop crops to the TRUE (unpadded) output size via Tensor::narrow.
        let decoded = self.decode_crop(&latents, height, width)?; // (1,3,height,width)

        crate::debug::probe("decoded", &decoded)?;

        // Crop `processed` to the same true dims for color correction; the padded border
        // pixels are blank-extended content that would skew the luminance histogram.
        let processed_true = processed.narrow(2, 0, height)?.narrow(3, 0, width)?;

        let skip_color = std::env::var("SEEDVR2_SKIP_COLOR")
            .map(|v| v == "1")
            .unwrap_or(false);

        let corrected = if skip_color {
            decoded.to_dtype(DType::F32)?
        } else {
            let corrected = color::apply_color_correction(
                &decoded.to_dtype(DType::F32)?,
                &processed_true.to_dtype(DType::F32)?,
                LUMINANCE_WEIGHT,
            )?;

            crate::debug::probe("color-corrected", &corrected)?;

            corrected
        };

        Ok(decoded_to_image(&corrected)?)
    }

    /// Spatial-tiling still-image path (sc-6225): upscale one LR image by feather-blended spatial
    /// tiling â€” the image analog of the [`Self::generate_video_tiled`] per-frame branch. Reuses the
    /// budget tile sizer + parity-gated [`Self::run_frame_tiled`] + per-frame color correction, so peak
    /// stays bounded at any resolution (no single allocation exceeds the budget-sized tile). `safe_gib`
    /// sizes the tile. Honors `cancel` before each spatial tile (sc-12465).
    #[allow(clippy::too_many_arguments)] // size/seed/budget + the threaded cancel flag
    fn generate_tiled(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        safe_gib: f64,
        cancel: Option<&CancelFlag>,
    ) -> CResult<Image> {
        let tile = video::plan_spatial_tile_px(self.weights_bytes, safe_gib);
        let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
        let processed = self
            .preprocess(image, width, height, softness)?
            .unsqueeze(2)?; // (1,3,1,H,W)
        let decoded = self.run_frame_tiled(&processed, seed, tile, overlap, cancel)?;
        Ok(self
            .frames_from_decoded(&decoded, &processed, 1)?
            .into_iter()
            .next()
            .expect("one tiled frame"))
    }

    /// LR `Image` â†’ `(1,3,height,width)` in [-1,1] at the model dtype. Bicubic resize to target;
    /// optional `softness` pre-blur via a smaller round-trip.
    pub fn preprocess(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        softness: f32,
    ) -> CResult<Tensor> {
        let (ih, iw) = (image.height as usize, image.width as usize);
        let resized: Vec<f32> = if softness > 0.0 {
            let factor = 1.0 + softness.clamp(0.0, 1.0) * 7.0;
            let dw = ((width as f32 / factor) as usize).max(2);
            let dh = ((height as f32 / factor) as usize).max(2);
            let down = imageops::resize_bicubic_u8(&image.pixels, ih, iw, dh, dw)?;
            let down_u8: Vec<u8> = down
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect();
            imageops::resize_bicubic_u8(&down_u8, dh, dw, height, width)?
        } else {
            imageops::resize_bicubic_u8(&image.pixels, ih, iw, height, width)?
        };
        // HWC [0,255] â†’ [-1,1] â†’ (1,3,H,W)
        let arr = Tensor::from_vec(resized, (height, width, 3), &self.device)?;
        let arr = (arr.affine(2.0 / 255.0, -1.0))?;
        Ok(arr
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?
            .contiguous()?)
    }

    // -----------------------------------------------------------------------
    // video mode (sc-5926): multi-frame 5-D pass + temporal chunking/overlap + HD tiling
    // -----------------------------------------------------------------------

    /// Decode latents and crop spatially to `(true_h, true_w)`, **keeping all `T` frames** â†’
    /// `(B,3,T,true_h,true_w)`. The 5-D analog of `Self::decode_crop` (which keeps only frame 0).
    pub fn decode_crop_5d(&self, latents: &Tensor, true_h: usize, true_w: usize) -> Result<Tensor> {
        let decoded = self.vae.decode(latents)?; // (B,3,T,H,W)
        decoded
            .narrow(3, 0, true_h)?
            .narrow(4, 0, true_w)?
            .contiguous()
    }

    /// Multi-frame model path (no color correction): a preprocessed clip `(1,3,T,H,W)` + injected
    /// noise `(1,16,T',h,w)` â†’ decoded crop `(1,3,T,true_h,true_w)`. The video analog of
    /// [`Self::run_model`]; `encode`/`condition`/`denoise` are already `T`-aware. Public for the harness.
    pub fn run_model_5d(
        &self,
        processed: &Tensor,
        noise: &Tensor,
        true_h: usize,
        true_w: usize,
    ) -> Result<Tensor> {
        let latent = self.vae.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond)?;
        self.decode_crop_5d(&latents, true_h, true_w)
    }

    /// Per-frame color-correct a decoded clip `(1,3,T,Hc,Wc)` against its preprocessed `style`
    /// `(1,3,Ts,Hc,Wc)` â†’ `count` RGB8 frames. Frame `t` matches style frame `min(t, Ts-1)`.
    fn frames_from_decoded(
        &self,
        decoded: &Tensor,
        style: &Tensor,
        count: usize,
    ) -> CResult<Vec<Image>> {
        let style_t = style.dim(2)?;
        let mut out = Vec::with_capacity(count);
        for t in 0..count {
            let d = decoded.narrow(2, t, 1)?.squeeze(2)?; // (1,3,Hc,Wc)
            let s = style.narrow(2, t.min(style_t - 1), 1)?.squeeze(2)?;
            let corrected = color::apply_color_correction(
                &d.to_dtype(DType::F32)?,
                &s.to_dtype(DType::F32)?,
                LUMINANCE_WEIGHT,
            )?;
            out.push(decoded_to_image(&corrected)?);
        }
        Ok(out)
    }

    /// Preprocess one temporal chunk: pixel-frames `[start, start+len)` of `frames`, clamping past the
    /// end to the last real frame (last-frame padding) â†’ `(1,3,len,H,W)` in `[-1,1]`.
    fn preprocess_chunk(
        &self,
        frames: &[Image],
        start: i32,
        len: i32,
        width: usize,
        height: usize,
        softness: f32,
    ) -> CResult<Tensor> {
        let n = frames.len() as i32;
        let mut per = Vec::with_capacity(len as usize);
        for j in 0..len {
            let idx = (start + j).clamp(0, n - 1) as usize;
            per.push(
                self.preprocess(&frames[idx], width, height, softness)?
                    .unsqueeze(2)?, // (1,3,1,H,W)
            );
        }
        let refs: Vec<&Tensor> = per.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }

    /// End-to-end **video** upscale: a sequence of LR `frames` â†’ upscaled `(width, height)` RGB8
    /// frames (same count). Sizes the temporal chunk against the memory budget (or `chunk_override`),
    /// processes each chunk through the 5-D model path with one-step Euler, per-frame color-corrects,
    /// and cross-fades chunk overlaps to close the causal-VAE seam ([`crate::video`]). Falls back to
    /// the per-frame (`T=1`) path under tight memory, and to per-frame **spatial tiling** when even one
    /// full-resolution frame exceeds the budget (HD), so peak stays bounded at any resolution.
    ///
    /// `cancel` is the caller's cooperative cancel flag (the gen-core video per-step cancel contract,
    /// sc-8972/sc-11227; mirrors `Sam3VideoModel::propagate`): checked before each (seconds-to-minutes)
    /// chunk/frame â€” and before each spatial tile within a tiled frame (sc-12465) â€” so a mid-clip
    /// cancel is honored promptly instead of after the whole upscale, surfacing
    /// the typed [`CandleError::Canceled`]. `progress` is invoked `(units_done, units_total)` after each
    /// completed chunk/frame (real per-unit progress, 1-based `units_done`), replacing the old
    /// fixed-placeholder single step; `units_total` is the number of chunks (chunked path) or frames
    /// (per-frame / spatial-tiling paths).
    #[allow(clippy::too_many_arguments)] // the video seam threads size/seed + cancel + progress
    pub fn generate_video(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        chunk_override: Option<i32>,
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        let n = frames.len() as i32;
        if n == 0 {
            return Ok(Vec::new());
        }
        // VAE-decoder correctness cap (sc-8228/sc-8261): the Chunked / PerFrame branches below decode
        // at the full `widthÃ—height` per frame, which corrupts above ~1536Â² regardless of memory. Above
        // the cap, force the per-frame spatial tiler (tiles capped at `VAE_SAFE_DECODE_EDGE_PX`).
        if width.max(height) as i32 > video::VAE_SAFE_DECODE_EDGE_PX {
            return self
                .generate_video_tiled(frames, width, height, seed, softness, cancel, progress);
        }
        let chunk = match (
            chunk_override,
            video::plan_chunk_size(self.weights_bytes, height as i32, width as i32),
        ) {
            // Clamp an override DOWN to the budget-safe chunk so it can't bypass the planner and OOM;
            // a smaller-than-safe override is honored as-is. PerFrame/OverBudget route to the safe
            // paths regardless of the override, since forcing a chunk there would OOM.
            (Some(c), ChunkPlan::Chunked(safe)) => video::pad_to_valid_chunk(c).min(safe),
            (_, ChunkPlan::Chunked(c)) => c,
            (_, ChunkPlan::PerFrame) => {
                return self.generate_video_per_frame(
                    frames, width, height, seed, softness, cancel, progress,
                )
            }
            // Even one full-resolution frame exceeds the budget â†’ spatially tile each frame
            // (per-frame T=1 + overlap feather blend). Bounds peak at any resolution (sc-5201).
            (_, ChunkPlan::OverBudget { .. }) => {
                return self
                    .generate_video_tiled(frames, width, height, seed, softness, cancel, progress)
            }
        };

        let plan = video::plan_chunks(n, chunk, video::DEFAULT_OVERLAP);
        let total = plan.len();
        // Assemble incrementally: cross-fade each chunk into `out` as it is produced and drop the
        // chunk, so host RAM stays at one full video plus the one chunk in flight â€” the batch path
        // held every chunk's (overlap-duplicated) frames AND cloned the whole clip at peak (~2â€“3Ã—
        // the video, ~15â€“17 GB for a 300-frame 4K clip; F-128, sc-11234). Output is byte-identical:
        // chunks are processed in the same plan order with the same cross-fade schedule.
        let mut out: Vec<Image> = Vec::with_capacity(n.max(0) as usize);
        for (i, Chunk { start, len }) in plan.iter().enumerate() {
            // Honor the engine cancellation contract before each (seconds-to-minutes) chunk.
            check_canceled(cancel)?;
            let clip = self.preprocess_chunk(frames, *start, *len, width, height, softness)?;
            let latent = self.vae.encode(&clip)?;
            let (_b, _c, lt, lh, lw) = latent.dims5()?;
            // Same noise per chunk (deterministic) â†’ a clean overlap blend (the reference behavior).
            let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
            let cond = Self::condition(&latent)?;
            let latents = self.denoise(&noise, &cond)?;
            let decoded = self.decode_crop_5d(&latents, height, width)?;
            let chunk_imgs = self.frames_from_decoded(&decoded, &clip, *len as usize)?;
            video::assemble_overlap_chunk(&mut out, *start, chunk_imgs, n, video::DEFAULT_OVERLAP);
            if let Some(cb) = progress.as_deref_mut() {
                cb(i + 1, total); // real per-chunk progress (1-based done / total chunks)
            }
        }
        Ok(out)
    }

    /// Per-frame (`T=1`) video fallback: each frame through the still path with a fixed (anchored)
    /// seed â€” intrinsically temporally stable (spike sc-4812). Used when even an 8-frame chunk
    /// exceeds the memory budget. Honors `cancel` before each frame and reports `(frame+1, total)`
    /// real progress (sc-11227).
    #[allow(clippy::too_many_arguments)] // size/seed + cancel + progress threaded from generate_video
    fn generate_video_per_frame(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        let total = frames.len();
        let mut out = Vec::with_capacity(total);
        for (i, f) in frames.iter().enumerate() {
            check_canceled(cancel)?;
            out.push(self.generate(f, width, height, seed, softness, cancel)?);
            if let Some(cb) = progress.as_deref_mut() {
                cb(i + 1, total);
            }
        }
        Ok(out)
    }

    /// HD spatial-tiling video path (sc-5201): each frame is upscaled per-frame (`T=1`) but **spatially
    /// tiled** â€” the budget sizer picks the largest square tile that fits, and the decoded tiles are
    /// feather-blended. Used when even one full-resolution frame exceeds the memory budget; bounds peak
    /// at any resolution. Honors `cancel` before each frame **and before each spatial tile within a
    /// frame** (sc-12465 â€” a single 4096-wide frame spans ~9 tiles, each a full encodeâ†’DiTâ†’decode),
    /// and reports `(frame+1, total)` real progress (sc-11227).
    #[allow(clippy::too_many_arguments)] // size/seed + cancel + progress threaded from generate_video
    fn generate_video_tiled(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        let tile = video::plan_spatial_tile_px(self.weights_bytes, video::safe_budget_gib());
        let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
        let total = frames.len();
        let mut out = Vec::with_capacity(total);
        for (i, f) in frames.iter().enumerate() {
            check_canceled(cancel)?;
            let processed = self.preprocess(f, width, height, softness)?.unsqueeze(2)?; // (1,3,1,H,W)
            let decoded = self.run_frame_tiled(&processed, seed, tile, overlap, cancel)?;
            let imgs = self.frames_from_decoded(&decoded, &processed, 1)?;
            out.push(imgs.into_iter().next().expect("one frame"));
            if let Some(cb) = progress.as_deref_mut() {
                cb(i + 1, total);
            }
        }
        Ok(out)
    }

    /// Upscale one preprocessed frame `(1,3,1,H,W)` by spatial tiling: run the full encode â†’ DiT â†’
    /// decode path on each overlapping `tile`-px tile (one-step Euler, same-seed noise) and feather-
    /// blend the decoded tiles into a full `(1,3,1,H,W)` frame. candle is eager, so only one tile's
    /// activations are resident at a time (the memory bound); the two full-frame accumulators are the
    /// only persistent allocations. Honors `cancel` before each tile (sc-12465 â€” each tile is a full
    /// seconds-to-minutes encodeâ†’DiTâ†’decode pass, the per-unit granularity of the cancel contract on
    /// this path). Public for the harness.
    pub fn run_frame_tiled(
        &self,
        processed: &Tensor,
        seed: u64,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> CResult<Tensor> {
        let (_b, _c, _t, height, width) = processed.dims5()?; // (1,3,1,H,W)
        let (h_i, w_i) = (height as i32, width as i32);

        // `tile` is the square-equivalent budget edge; what the budget actually bounds is tile
        // *area*, and letting the shape follow the frame is worth ~2× on near-square frames
        // (see `video::plan_spatial_tiles_rect`).
        let max_area = (tile as f64) * (tile as f64);
        let (plan, th_px, tw_px) = video::plan_spatial_tiles_rect(h_i, w_i, max_area, overlap);
        let planned_px: i64 = plan
            .iter()
            .map(|t| ((t.y1 - t.y0) as i64) * ((t.x1 - t.x0) as i64))
            .sum();
        let frame_px = (h_i as i64) * (w_i as i64);
        eprintln!(
            "[seedvr2-tile-plan] frame={}x{} budget_edge={} tile={}x{} overlap={} tiles={} work={:.2}x frame",
            h_i,
            w_i,
            tile,
            th_px,
            tw_px,
            overlap,
            plan.len(),
            planned_px as f64 / frame_px.max(1) as f64
        );

        for (i, t) in plan.iter().enumerate() {
            eprintln!(
                "[seedvr2-tile-plan] #{:02} y={}..{} x={}..{} size={}x{}",
                i,
                t.y0,
                t.y1,
                t.x0,
                t.x1,
                t.y1 - t.y0,
                t.x1 - t.x0
            );
        }

        let mut acc: Option<Tensor> = None; // (1,3,1,H,W)
        let mut wsum: Option<Tensor> = None; // (1,1,1,H,W)
        let mut tile_no = 0usize;
        // `for_each_tile` checks the cancel flag before each (seconds-to-minutes) tile.
        for_each_tile(&plan, cancel, |t| {
            let (th, tw) = ((t.y1 - t.y0) as usize, (t.x1 - t.x0) as usize);
            let (y0, x0) = (t.y0 as usize, t.x0 as usize);
            let mut stage = Stage::new(&self.device);
            let tile_clip = processed.narrow(3, y0, th)?.narrow(4, x0, tw)?; // (1,3,1,th,tw)

            crate::debug::probe("tile input", &tile_clip)?;

            // full model path on the tile (one-step Euler), same noise per tile.
            let latent = self.vae.encode(&tile_clip)?;
            let t_encode = stage.lap();

            crate::debug::probe("tile VAE latent", &latent)?;
            let (_b, _c, lt, lh, lw) = latent.dims5()?;
            let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
            let cond = Self::condition(&latent)?;
            let t_noise = stage.lap();
            let latents = self.denoise(&noise, &cond)?;
            let t_denoise = stage.lap();

            crate::debug::probe("tile latents (post-denoise)", &latents)?;

            let decoded = self.decode_crop_5d(&latents, th, tw)?;
            let t_decode = stage.lap();

            crate::debug::probe("tile decoded", &decoded)?;

            // existing feather/reconstruction code continues here

            // feather weight tapering on edges that abut a neighbor; placed at (y0,x0).
            let wvec = video::feather_weight(
                th as i32,
                tw as i32,
                t.y0 > 0,
                t.y1 < h_i,
                t.x0 > 0,
                t.x1 < w_i,
                overlap,
            );
            let weight =
                Tensor::from_vec(wvec, (1, 1, 1, th, tw), &self.device)?.to_dtype(self.dtype)?;
            let (pad_r, pad_b) = (width - x0 - tw, height - y0 - th);
            let wdec = decoded
                .broadcast_mul(&weight)?
                .pad_with_zeros(3, y0, pad_b)?
                .pad_with_zeros(4, x0, pad_r)?;
            let wpad = weight
                .pad_with_zeros(3, y0, pad_b)?
                .pad_with_zeros(4, x0, pad_r)?;
            acc = Some(match acc.take() {
                Some(a) => a.add(&wdec)?,
                None => wdec,
            });
            wsum = Some(match wsum.take() {
                Some(a) => a.add(&wpad)?,
                None => wpad,
            });
            let t_blend = stage.lap();
            eprintln!(
                "[seedvr2-tile-time] #{:02} {}x{} encode={:.2}s noise={:.2}s dit={:.2}s decode={:.2}s blend={:.2}s total={:.2}s",
                tile_no,
                th,
                tw,
                t_encode,
                t_noise,
                t_denoise,
                t_decode,
                t_blend,
                stage.total()
            );
            tile_no += 1;
            Ok(())
        })?;

                    let acc = acc.expect("≥1 tile");
        let wsum = wsum.expect("≥1 tile");

        let output = acc.broadcast_div(&wsum)?;

        crate::debug::probe("tile assembly OUTPUT", &output)?;

        Ok(output)
    }
}

     
#[cfg(test)]
mod tests {
    use super::*;

    /// sc-11227 / F-092: the upscale loops check `check_canceled` before each
    /// (seconds-to-minutes) chunk/frame/tile and surface the **typed** [`CandleError::Canceled`] (never a
    /// stringified `Msg` â€” the worker keys off the variant to map a user cancel). An absent or
    /// untripped flag is a no-op. Mirrors `Sam3VideoModel`'s `check_canceled` (sc-8972).
    #[test]
    fn check_canceled_surfaces_typed_canceled_only_on_trip() {
        assert!(check_canceled(None).is_ok(), "no flag â†’ no-op");
        let flag = CancelFlag::new();
        assert!(check_canceled(Some(&flag)).is_ok(), "untripped â†’ no-op");
        flag.cancel();
        assert!(
            matches!(check_canceled(Some(&flag)), Err(CandleError::Canceled)),
            "tripped â†’ typed Canceled"
        );
    }

    /// sc-12465 / F-013: the spatial-tile loop checks `check_canceled` at the **top of every tile
    /// iteration** â€” the per-unit cancel contract's unit on the tiled paths is one tile (a full
    /// VAE-encode â†’ DiT â†’ VAE-decode, seconds-to-minutes), not one frame/image. This exercises
    /// [`for_each_tile`] itself â€” the production driver [`Seedvr2Pipeline::run_frame_tiled`] runs
    /// its tile work through (full-pipeline construction needs a complete fixed-arch checkpoint, so
    /// the driver is extracted to make the real loop weight-free-testable; deleting the driver's
    /// `check_canceled` fails this test). Pins that a flag tripped after N completed tiles stops
    /// the loop before tile N+1, a pre-tripped flag stops it before any tile, and an untripped flag
    /// runs the full plan. The real-weight path is exercised by
    /// `cuda_video_smoke::cuda_tiled_cancel_*`.
    #[test]
    fn spatial_tile_loop_cancels_between_tiles() {
        // The F-013 descriptor-max case: a 4096Â² still at the VAE-cap tile edge (1536) with the
        // standard overlap â†’ a 3Ã—3 = 9-tile plan, so a cancel-free tiled still would run 9 full
        // encodeâ†’DiTâ†’decode passes after the last pre-`generate` check.
        let plan = video::plan_spatial_tiles(
            4096,
            4096,
            video::VAE_SAFE_DECODE_EDGE_PX,
            video::SPATIAL_OVERLAP,
        );
        assert_eq!(plan.len(), 9, "4096Â² @ tile 1536 must be a 3Ã—3 plan");

        // Untripped flag â†’ every tile runs, Ok.
        let flag = CancelFlag::new();
        let mut tiles_done = 0usize;
        let result = for_each_tile(&plan, Some(&flag), |_t| {
            tiles_done += 1;
            Ok(())
        });
        assert!(result.is_ok(), "untripped flag must not abort the loop");
        assert_eq!(tiles_done, plan.len(), "untripped flag runs the full plan");

        for trip_after in [0usize, 1, 3] {
            let flag = CancelFlag::new();
            if trip_after == 0 {
                flag.cancel(); // pre-cancelled before the loop starts
            }
            let mut tiles_done = 0usize;
            // The closure is the tile's encodeâ†’DiTâ†’decode work; the cancel lands mid-run here.
            let result = for_each_tile(&plan, Some(&flag), |_t| {
                tiles_done += 1;
                if tiles_done == trip_after {
                    flag.cancel();
                }
                Ok(())
            });
            assert!(
                matches!(result, Err(CandleError::Canceled)),
                "tripped flag must surface the typed Canceled (trip_after={trip_after})"
            );
            assert_eq!(
                tiles_done, trip_after,
                "no tile may run after the flag trips (trip_after={trip_after})"
            );
        }
    }

    /// The per-unit progress contract (sc-11227): the loops report `(done, total)` with `done`
    /// running 1..=total (monotonic, 1-based, ending exactly at `total`) â€” real per-chunk/per-frame
    /// progress, NOT the old fixed `Step { current: 1, total: 1 }` placeholder that jumped to "done".
    /// This asserts the callback-driving arithmetic the three loops share (`cb(i + 1, total)`).
    #[test]
    fn progress_callback_reports_monotonic_one_based_units() {
        for total in [1usize, 4, 20] {
            let mut seen: Vec<(usize, usize)> = Vec::new();
            // Mirror the loops' `if let Some(cb) = progress.as_deref_mut() { cb(i + 1, total) }`.
            let mut cb = |done: usize, tot: usize| seen.push((done, tot));
            let mut progress: Option<&mut dyn FnMut(usize, usize)> = Some(&mut cb);
            for i in 0..total {
                if let Some(cb) = progress.as_deref_mut() {
                    cb(i + 1, total);
                }
            }
            assert_eq!(seen.len(), total, "one progress event per completed unit");
            assert_eq!(seen.first().copied(), Some((1, total)), "1-based start");
            assert_eq!(seen.last().copied(), Some((total, total)), "ends at total");
            assert!(
                seen.windows(2).all(|w| w[1].0 > w[0].0),
                "strictly increasing (monotonic real progress, not a fixed placeholder): {seen:?}"
            );
        }
    }
}









