//! Opt-in tensor sanity probes.
//!
//! Off by default: [`probe`] returns immediately, so there are no GPU->CPU copies and no output.
//! `SEEDVR2_DEBUG=1` logs whole-tensor NaN / inf / min / max for each probed stage;
//! `SEEDVR2_DEBUG=2` additionally panics on the first stage that contains a NaN or inf.

use candle_gen::candle_core::{DType, Result, Tensor};
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
