//! Opt-in tensor sanity probes and stage timing.
//!
//! Off by default: [`probe`] returns immediately, so there are no GPU->CPU copies and no output.
//! `SEEDVR2_DEBUG=1` logs whole-tensor NaN / inf / min / max for each probed stage;
//! `SEEDVR2_DEBUG=2` additionally panics on the first stage that contains a NaN or inf.
//! `SEEDVR2_TIMING=1` enables [`StageTimer`], the per-block breakdown inside one model pass.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use std::sync::OnceLock;

/// 0 = off, 1 = log, 2 = log + panic on NaN/inf.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("SEEDVR2_DEBUG").ok().as_deref() {
        None | Some("") | Some("0") => 0,
        Some("2") | Some("strict") => 2,
        Some(_) => 1,
    })
}

/// True when any `SEEDVR2_DEBUG` mode is active (lets callers skip building expensive debug-only values).
pub(crate) fn enabled() -> bool {
    mode() > 0
}

/// Report whole-tensor stats for `t` under `name`. No-op unless `SEEDVR2_DEBUG` is set.
pub(crate) fn probe(name: &str, t: &Tensor) -> Result<()> {
    let m = mode();
    if m == 0 {
        return Ok(());
    }
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let (mut nan, mut inf) = (0usize, 0usize);
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in &v {
        if x.is_nan() {
            nan += 1;
        } else if x.is_infinite() {
            inf += 1;
        } else {
            min = min.min(x);
            max = max.max(x);
        }
    }
    eprintln!("[seedvr2-debug] {} shape={:?} nan={} inf={} min={} max={}", name, t.dims(), nan, inf, min, max);
    if m == 2 && (nan > 0 || inf > 0) {
        panic!("[seedvr2-debug] first bad stage: {name}");
    }
    Ok(())
}

/// True when `SEEDVR2_TIMING` is set to anything but `0`/empty — enables [`StageTimer`].
fn timing_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SEEDVR2_TIMING").ok().as_deref(),
            None | Some("") | Some("0")
        )
    })
}

/// Per-stage wall-clock splits inside one model pass, printed as one `[seedvr2-stage]` line.
///
/// Off unless `SEEDVR2_TIMING` is set. Every lap synchronizes the device, which is the only way to
/// get honest numbers from an async backend (kernel launches return immediately, so without it each
/// stage's cost lands on whichever later call happens to block) — but it also serializes the
/// pipeline, so it is not something to leave on in production.
///
/// Each stage's output shape is printed with its time, because on this pipeline "which stage" and
/// "at what resolution" are the same question: cost is dominated by whatever runs after the last
/// upsample.
pub(crate) struct StageTimer {
    device: Device,
    laps: Vec<(String, f64, Vec<usize>)>,
    last: std::time::Instant,
}

impl StageTimer {
    pub(crate) fn new(device: &Device) -> Self {
        if timing_enabled() {
            let _ = device.synchronize();
        }
        Self {
            device: device.clone(),
            laps: Vec::new(),
            last: std::time::Instant::now(),
        }
    }

    pub(crate) fn lap(&mut self, name: &str, out: &Tensor) {
        if !timing_enabled() {
            return;
        }
        let _ = self.device.synchronize();
        let now = std::time::Instant::now();
        self.laps.push((
            name.to_string(),
            now.duration_since(self.last).as_secs_f64(),
            out.dims().to_vec(),
        ));
        self.last = now;
    }

    pub(crate) fn report(&self, label: &str) {
        if !timing_enabled() || self.laps.is_empty() {
            return;
        }
        let total: f64 = self.laps.iter().map(|(_, d, _)| d).sum();
        let parts: Vec<String> = self
            .laps
            .iter()
            .map(|(n, d, shape)| {
                format!("{n}={d:.3}s({:.0}%){shape:?}", 100.0 * d / total.max(1e-9))
            })
            .collect();
        eprintln!(
            "[seedvr2-stage] {label} total={total:.3}s  {}",
            parts.join("  ")
        );
    }
}
