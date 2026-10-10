//! ROCm conv2d correctness probe (diagnostic, no model weights needed).
//!
//! Runs the exact conv2d shapes the SeedVR2 VAE encoder hits for a 948x1422 frame (single pass) and
//! a 1072x948 tile, on the ROCm device, and compares against a CPU F32 reference computed on a few
//! output-row bands. Reports NaN/inf counts, max error, the worst columns (to see if errors line up
//! with the 8/16px streak pattern), whether the same conv is deterministic run-to-run, and whether
//! the row-chunked path (what `conv3d.rs` uses above its im2col budget) matches.
//!
//! Run:
//!   cargo test -p candle-gen-seedvr2 --features rocm --test rocm_conv_check -- --ignored --nocapture
//! Optional env:
//!   SEEDVR2_CONV_FILTER=down2     only cases whose name contains this substring
//!   SEEDVR2_CONV_DTYPES=f32       comma list of f16,f32 (default: f16,f32)

use candle_gen::candle_core::{DType, Device, Result, Tensor};

/// Same budget as `conv3d.rs::IM2COL_BUDGET`.
const BUDGET: usize = 128 * 1024 * 1024;
/// Output rows compared per band against the CPU reference.
const BAND_ROWS: usize = 2;

struct Case {
    name: &'static str,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    pad: usize,
    stride: usize,
    /// Apply the VAE downsampler's asymmetric (0,1) H/W pad before the conv.
    asym: bool,
}

const CASES: &[Case] = &[
    Case { name: "conv_in 3->128 @1422x948", cin: 3, cout: 128, h: 1422, w: 948, pad: 1, stride: 1, asym: false },
    Case { name: "block0 res 128->128 @1422x948", cin: 128, cout: 128, h: 1422, w: 948, pad: 1, stride: 1, asym: false },
    Case { name: "down0 s2 128->128 @1422x948", cin: 128, cout: 128, h: 1422, w: 948, pad: 0, stride: 2, asym: true },
    Case { name: "block1 res0 128->256 @711x474", cin: 128, cout: 256, h: 711, w: 474, pad: 1, stride: 1, asym: false },
    Case { name: "block1 res 256->256 @711x474", cin: 256, cout: 256, h: 711, w: 474, pad: 1, stride: 1, asym: false },
    Case { name: "down1 s2 256->256 @711x474", cin: 256, cout: 256, h: 711, w: 474, pad: 0, stride: 2, asym: true },
    Case { name: "block2 res0 256->512 @355x237", cin: 256, cout: 512, h: 355, w: 237, pad: 1, stride: 1, asym: false },
    Case { name: "block2 res 512->512 @355x237", cin: 512, cout: 512, h: 355, w: 237, pad: 1, stride: 1, asym: false },
    Case { name: "down2 s2 512->512 @355x237", cin: 512, cout: 512, h: 355, w: 237, pad: 0, stride: 2, asym: true },
    Case { name: "block3 res 512->512 @177x118", cin: 512, cout: 512, h: 177, w: 118, pad: 1, stride: 1, asym: false },
    Case { name: "conv_in 3->128 @1072x948 (tile)", cin: 3, cout: 128, h: 1072, w: 948, pad: 1, stride: 1, asym: false },
    Case { name: "block0 res 128->128 @1072x948 (tile)", cin: 128, cout: 128, h: 1072, w: 948, pad: 1, stride: 1, asym: false },
];

/// Copy of `conv3d.rs::row_chunked_conv2d` (private there): explicit symmetric pad, then output-row
/// bands each under `budget` im2col elements, concatenated along H.
fn row_chunked(x: &Tensor, w: &Tensor, padding: usize, stride: usize, budget: usize, contig: bool) -> Result<Tensor> {
    let n = x.dim(0)?;
    let c = x.dim(1)?;
    let (_o, _i, kh, kw) = w.dims4()?;
    let xp = if padding > 0 {
        x.pad_with_zeros(2, padding, padding)?.pad_with_zeros(3, padding, padding)?
    } else {
        x.clone()
    };
    let hp = xp.dim(2)?;
    let wp = xp.dim(3)?;
    let h_out = hp.saturating_sub(kh) / stride + 1;
    let w_out = wp.saturating_sub(kw) / stride + 1;
    let cols_per_out_row = (n * w_out * c * kh * kw).max(1);
    let band_rows = (budget / cols_per_out_row).max(1);
    let mut parts = Vec::new();
    let mut ro = 0;
    while ro < h_out {
        let rb = (h_out - ro).min(band_rows);
        let band = xp.narrow(2, ro * stride, (rb - 1) * stride + kh)?;
        let band = if contig { band.contiguous()? } else { band };
        parts.push(band.conv2d(w, 0, stride, 1, 1)?);
        ro += rb;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

/// Same bands as `row_chunked` (production form, no `.contiguous()`), but returns the per-band
/// outputs un-concatenated so the caller can concatenate on the CPU instead of on the GPU.
fn row_chunked_parts(x: &Tensor, w: &Tensor, padding: usize, stride: usize, budget: usize) -> Result<Vec<Tensor>> {
    let n = x.dim(0)?;
    let c = x.dim(1)?;
    let (_o, _i, kh, kw) = w.dims4()?;
    let xp = if padding > 0 {
        x.pad_with_zeros(2, padding, padding)?.pad_with_zeros(3, padding, padding)?
    } else {
        x.clone()
    };
    let hp = xp.dim(2)?;
    let wp = xp.dim(3)?;
    let h_out = hp.saturating_sub(kh) / stride + 1;
    let w_out = wp.saturating_sub(kw) / stride + 1;
    let band_rows = (budget / (n * w_out * c * kh * kw).max(1)).max(1);
    let mut parts = Vec::new();
    let mut ro = 0;
    while ro < h_out {
        let rb = (h_out - ro).min(band_rows);
        let band = xp.narrow(2, ro * stride, (rb - 1) * stride + kh)?;
        parts.push(band.conv2d(w, 0, stride, 1, 1)?);
        ro += rb;
    }
    Ok(parts)
}
fn to_cpu_f32(t: &Tensor) -> Result<Tensor> {
    t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()
}

fn vec_of(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

fn nonfinite(v: &[f32]) -> usize {
    v.iter().filter(|x| !x.is_finite()).count()
}

struct Report {
    nonfinite: usize,
    max_err: f32,
    ref_max: f32,
    worst_cols: Vec<(usize, f32)>,
}

/// Compare `out` (1,O,Hout,Wout, on CPU F32) to CPU-reference bands.
fn compare_bands(out: &Tensor, refs: &[(usize, Tensor)]) -> Result<Report> {
    let full = vec_of(out)?;
    let nonfinite_total = nonfinite(&full);
    drop(full);
    let (_n, o, _h, wout) = out.dims4()?;
    let mut max_err = 0f32;
    let mut ref_max = 0f32;
    let mut col_err = vec![0f32; wout];
    for (r0, rb) in refs {
        let got = vec_of(&out.narrow(2, *r0, BAND_ROWS)?.contiguous()?)?;
        let exp = vec_of(rb)?;
        for c in 0..o {
            for r in 0..BAND_ROWS {
                for x in 0..wout {
                    let i = (c * BAND_ROWS + r) * wout + x;
                    let e = if got[i].is_finite() { (got[i] - exp[i]).abs() } else { f32::INFINITY };
                    max_err = max_err.max(e);
                    ref_max = ref_max.max(exp[i].abs());
                    col_err[x] = col_err[x].max(e);
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..wout).collect();
    idx.sort_by(|a, b| col_err[*b].partial_cmp(&col_err[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let worst_cols = idx.into_iter().take(6).map(|i| (i, col_err[i])).collect();
    Ok(Report { nonfinite: nonfinite_total, max_err, ref_max, worst_cols })
}

fn run_case(dev: &Device, dtype: DType, case: &Case) -> Result<bool> {
    let cpu = Device::Cpu;
    let k = 3usize;
    let std_w = (1.0 / ((case.cin * k * k) as f64).sqrt()) as f32;
    let x = Tensor::randn(0f32, 1.0, (1, case.cin, case.h, case.w), &cpu)?;
    let w = Tensor::randn(0f32, std_w, (case.cout, case.cin, k, k), &cpu)?;

    // CPU reference sees the same values after the dtype round-trip.
    let mut x_ref = x.to_dtype(dtype)?.to_dtype(DType::F32)?;
    let w_ref = w.to_dtype(dtype)?.to_dtype(DType::F32)?;
    let mut x_dev = x.to_dtype(dtype)?.to_device(dev)?;
    let w_dev = w.to_dtype(dtype)?.to_device(dev)?;
    if case.asym {
        x_dev = x_dev.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        x_ref = x_ref.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
    }
    let (hin, win) = (x_dev.dim(2)?, x_dev.dim(3)?);
    let h_out = (hin + 2 * case.pad - k) / case.stride + 1;
    let w_out = (win + 2 * case.pad - k) / case.stride + 1;

    // Padded CPU input, built once; bands narrow into it.
    let x_ref_p = if case.pad > 0 {
        x_ref.pad_with_zeros(2, case.pad, case.pad)?.pad_with_zeros(3, case.pad, case.pad)?
    } else {
        x_ref
    };
    let band_starts = [0, (h_out / 2).min(h_out - BAND_ROWS), h_out - BAND_ROWS];
    let mut refs = Vec::new();
    for &r0 in &band_starts {
        let start = r0 * case.stride;
        let len = (BAND_ROWS - 1) * case.stride + k;
        let rb = x_ref_p.narrow(2, start, len)?.contiguous()?.conv2d(&w_ref, 0, case.stride, 1, 1)?;
        refs.push((r0, rb));
    }

    // GPU: direct conv (twice, for determinism) and the row-chunked production path.
    let direct_a = x_dev.conv2d(&w_dev, case.pad, case.stride, 1, 1)?;
    let direct_b = x_dev.conv2d(&w_dev, case.pad, case.stride, 1, 1)?;
    let chunked = row_chunked(&x_dev, &w_dev, case.pad, case.stride, BUDGET, false)?;
    let chunked_c = row_chunked(&x_dev, &w_dev, case.pad, case.stride, BUDGET, true)?;
    assert_eq!(direct_a.dims(), &[1, case.cout, h_out, w_out]);

    let a = to_cpu_f32(&direct_a)?;
    let b = to_cpu_f32(&direct_b)?;
    let c = to_cpu_f32(&chunked)?;
    let cc = to_cpu_f32(&chunked_c)?;

    let rep_a = compare_bands(&a, &refs)?;
    let rep_c = compare_bands(&c, &refs)?;
    let rep_cc = compare_bands(&cc, &refs)?;
    let parts = row_chunked_parts(&x_dev, &w_dev, case.pad, case.stride, BUDGET)?;
    let cpu_parts: Vec<Tensor> = parts.iter().map(to_cpu_f32).collect::<Result<_>>()?;
    let part_refs: Vec<&Tensor> = cpu_parts.iter().collect();
    let cpu_cat = Tensor::cat(&part_refs, 2)?;
    let rep_pc = compare_bands(&cpu_cat, &refs)?;

    // Run-to-run determinism on the same bands.
    let mut det = 0f32;
    for &r0 in &band_starts {
        let va = vec_of(&a.narrow(2, r0, BAND_ROWS)?.contiguous()?)?;
        let vb = vec_of(&b.narrow(2, r0, BAND_ROWS)?.contiguous()?)?;
        for (p, q) in va.iter().zip(vb.iter()) {
            det = det.max(if p.is_finite() && q.is_finite() { (p - q).abs() } else { f32::INFINITY });
        }
    }

    let tol = if dtype == DType::F32 { 1e-3f32 } else { 2e-2f32 };
    let bad = |r: &Report| r.nonfinite > 0 || r.max_err > tol * r.ref_max.max(1.0);
    let (bad_a, bad_c) = (bad(&rep_a), bad(&rep_c));
    let cols = |r: &Report| {
        r.worst_cols.iter().map(|(i, e)| format!("{i}(m8={},m16={}):{e:.3}", i % 8, i % 16)).collect::<Vec<_>>().join(" ")
    };
    eprintln!(
        "[conv-check] {:?} {:<38} direct: {} nonfinite={} max_err={:.5} (ref_max={:.2}) | chunked: {} nonfinite={} max_err={:.5} | run-to-run diff={:.5}",
        dtype, case.name,
        if bad_a { "BAD" } else { "ok " }, rep_a.nonfinite, rep_a.max_err, rep_a.ref_max,
        if bad_c { "BAD" } else { "ok " }, rep_c.nonfinite, rep_c.max_err, det,
    );
    eprintln!("[conv-check]    chunked+contiguous(): {} nonfinite={} max_err={:.5}", if rep_cc.nonfinite > 0 || rep_cc.max_err > tol * rep_cc.ref_max.max(1.0) { "BAD" } else { "ok " }, rep_cc.nonfinite, rep_cc.max_err);
    if bad_a {
        eprintln!("[conv-check]    direct worst cols: {}", cols(&rep_a));
    }
    if bad_c {
        eprintln!("[conv-check]    chunked worst cols: {}", cols(&rep_c));
    }
    eprintln!("[conv-check]    bands={} chunked, per-band GPU conv + CPU cat: {} nonfinite={} max_err={:.5}", cpu_parts.len(), if rep_pc.nonfinite > 0 || rep_pc.max_err > tol * rep_pc.ref_max.max(1.0) { "BAD" } else { "ok " }, rep_pc.nonfinite, rep_pc.max_err);
    Ok(!bad_a && !bad_c && det == 0.0)
}

#[test]
#[ignore = "ROCm conv2d correctness probe (diagnostic)"]
fn rocm_conv2d_matches_cpu() {
    let dev = candle_gen::default_device().expect("ROCm device");
    let filter = std::env::var("SEEDVR2_CONV_FILTER").ok();
    let dtypes: Vec<DType> = std::env::var("SEEDVR2_CONV_DTYPES")
        .unwrap_or_else(|_| "f16,f32".to_string())
        .split(',')
        .map(|s| match s.trim() {
            "f16" => DType::F16,
            "f32" => DType::F32,
            other => panic!("unsupported dtype `{other}`; use f16 or f32"),
        })
        .collect();
    eprintln!("[conv-check] device={dev:?} dtypes={dtypes:?} budget={BUDGET}");

    let mut failures = Vec::new();
    for dtype in dtypes {
        for case in CASES {
            if let Some(f) = &filter {
                if !case.name.contains(f.as_str()) {
                    continue;
                }
            }
            match run_case(&dev, dtype, case) {
                Ok(true) => {}
                Ok(false) => failures.push(format!("{dtype:?} {}", case.name)),
                Err(e) => {
                    eprintln!("[conv-check] {dtype:?} {} ERROR: {e}", case.name);
                    failures.push(format!("{dtype:?} {} (error)", case.name));
                }
            }
        }
    }
    assert!(failures.is_empty(), "conv2d mismatches/NaN on ROCm:\n  {}", failures.join("\n  "));
}


// ---------------------------------------------------------------------------------------------
// Localizer: which step of the multi-band path goes wrong (pad, narrow view, or conv on a band)?
//   SEEDVR2_CONV_FILTER=block2 cargo test ... --test rocm_conv_check rocm_chunk_localize -- --ignored --nocapture
// ---------------------------------------------------------------------------------------------

fn band_stats(out: &Tensor, rows: &[(usize, Tensor)], nr: usize) -> Result<(usize, f32)> {
    let nf = nonfinite(&vec_of(out)?);
    let mut me = 0f32;
    for (r0, rb) in rows {
        let got = vec_of(&out.narrow(2, *r0, nr)?.contiguous()?)?;
        let exp = vec_of(rb)?;
        for (g, e) in got.iter().zip(exp.iter()) {
            let d = if g.is_finite() { (g - e).abs() } else { f32::INFINITY };
            me = me.max(d);
        }
    }
    Ok((nf, me))
}

fn localize_case(dev: &Device, dtype: DType, case: &Case) -> Result<()> {
    let cpu = Device::Cpu;
    let k = 3usize;
    let std_w = (1.0 / ((case.cin * k * k) as f64).sqrt()) as f32;
    let x = Tensor::randn(0f32, 1.0, (1, case.cin, case.h, case.w), &cpu)?;
    let w = Tensor::randn(0f32, std_w, (case.cout, case.cin, k, k), &cpu)?;
    let mut x_ref = x.to_dtype(dtype)?.to_dtype(DType::F32)?;
    let w_ref = w.to_dtype(dtype)?.to_dtype(DType::F32)?;
    let mut x_dev = x.to_dtype(dtype)?.to_device(dev)?;
    let w_dev = w.to_dtype(dtype)?.to_device(dev)?;
    if case.asym {
        x_dev = x_dev.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        x_ref = x_ref.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
    }
    let (pad_gpu, pad_ref) = if case.pad > 0 {
        (
            x_dev.pad_with_zeros(2, case.pad, case.pad)?.pad_with_zeros(3, case.pad, case.pad)?,
            x_ref.pad_with_zeros(2, case.pad, case.pad)?.pad_with_zeros(3, case.pad, case.pad)?,
        )
    } else {
        (x_dev.clone(), x_ref.clone())
    };
    let (hp, wp) = (pad_gpu.dim(2)?, pad_gpu.dim(3)?);

    // (1) GPU zero-pad vs CPU zero-pad, whole tensor.
    let pg = vec_of(&to_cpu_f32(&pad_gpu)?)?;
    let pr = vec_of(&pad_ref)?;
    let mut bad = 0usize;
    let mut first: Option<usize> = None;
    let mut max_d = 0f32;
    for (i, (g, r)) in pg.iter().zip(pr.iter()).enumerate() {
        let d = if g.is_finite() { (g - r).abs() } else { f32::INFINITY };
        if d > 1e-3 {
            bad += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
        max_d = max_d.max(d);
    }
    drop(pg);
    drop(pr);
    match first {
        Some(i) => eprintln!(
            "[localize] {:?} {} PAD BAD: {} elems differ, max_diff={}, first at c={} row={} col={} (padded {}x{})",
            dtype, case.name, bad, max_d, i / (hp * wp), (i / wp) % hp, i % wp, hp, wp
        ),
        None => eprintln!("[localize] {:?} {} PAD ok (max_diff={})", dtype, case.name, max_d),
    }

    // (2) Per band: narrow view / narrow+contiguous / fresh upload, each vs CPU reference rows.
    let h_out = (hp - k) / case.stride + 1;
    let w_out = (wp - k) / case.stride + 1;
    let band_rows = (BUDGET / (w_out * case.cin * k * k).max(1)).max(1);
    let mut ro = 0usize;
    let mut bi = 0usize;
    while ro < h_out {
        let rb = (h_out - ro).min(band_rows);
        let in_start = ro * case.stride;
        let in_len = (rb - 1) * case.stride + k;
        let nr = rb.min(2);
        let mut rows = Vec::new();
        for r in [0usize, rb - nr] {
            let s = in_start + r * case.stride;
            let l = (nr - 1) * case.stride + k;
            rows.push((r, pad_ref.narrow(2, s, l)?.contiguous()?.conv2d(&w_ref, 0, case.stride, 1, 1)?));
        }
        let view = pad_gpu.narrow(2, in_start, in_len)?;
        let a = to_cpu_f32(&view.conv2d(&w_dev, 0, case.stride, 1, 1)?)?;
        let b = to_cpu_f32(&view.contiguous()?.conv2d(&w_dev, 0, case.stride, 1, 1)?)?;
        let fresh = pad_ref.narrow(2, in_start, in_len)?.contiguous()?.to_dtype(dtype)?.to_device(dev)?;
        let c = to_cpu_f32(&fresh.conv2d(&w_dev, 0, case.stride, 1, 1)?)?;
        let (sa, sb, sc) = (band_stats(&a, &rows, nr)?, band_stats(&b, &rows, nr)?, band_stats(&c, &rows, nr)?);
        eprintln!(
            "[localize] {:?} {} band#{} out_rows {}..{} (in_start={}, in_len={}): narrow-view nf={} err={:.4} | narrow+contig nf={} err={:.4} | fresh-upload nf={} err={:.4}",
            dtype, case.name, bi, ro, ro + rb, in_start, in_len, sa.0, sa.1, sb.0, sb.1, sc.0, sc.1
        );
        ro += rb;
        bi += 1;
    }
    Ok(())
}

#[test]
#[ignore = "ROCm band localizer (diagnostic)"]
fn rocm_chunk_localize() {
    let dev = candle_gen::default_device().expect("ROCm device");
    let filter = std::env::var("SEEDVR2_CONV_FILTER").unwrap_or_else(|_| "block2 res0".to_string());
    for dtype in [DType::F32, DType::F16] {
        for case in CASES {
            if case.name.contains(filter.as_str()) {
                localize_case(&dev, dtype, case).expect("localize_case");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// View-primitive probe: each op the Conv3d/VAE path uses to slice/copy tensors, run on ROCm and on
// the CPU with identical inputs. Copies must match exactly; convs within tolerance.
//   cargo test -p candle-gen-seedvr2 --features rocm --test rocm_conv_check rocm_view_primitives -- --ignored --nocapture
// ---------------------------------------------------------------------------------------------

fn prim_check(
    dev: &Device,
    dtype: DType,
    name: &str,
    x_cpu: &Tensor,
    tol: f32,
    op: &dyn Fn(&Tensor) -> Result<Tensor>,
) -> Result<bool> {
    let x_ref = x_cpu.to_dtype(dtype)?.to_dtype(DType::F32)?;
    let x_dev = x_cpu.to_dtype(dtype)?.to_device(dev)?;
    let want = op(&x_ref)?.to_dtype(DType::F32)?;
    let got = match op(&x_dev).and_then(|t| to_cpu_f32(&t)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[prim] {:?} {:<52} ERROR: {}", dtype, name, e);
            return Ok(false);
        }
    };
    if want.dims() != got.dims() {
        eprintln!("[prim] {:?} {:<52} BAD shape want={:?} got={:?}", dtype, name, want.dims(), got.dims());
        return Ok(false);
    }
    let wv = vec_of(&want)?;
    let gv = vec_of(&got)?;
    let nf = nonfinite(&gv);
    let mut max_err = 0f32;
    let mut ref_max = 0f32;
    for (g, w) in gv.iter().zip(wv.iter()) {
        let d = if g.is_finite() { (g - w).abs() } else { f32::INFINITY };
        max_err = max_err.max(d);
        ref_max = ref_max.max(w.abs());
    }
    let ok = nf == 0 && max_err <= tol * ref_max.max(1.0);
    eprintln!(
        "[prim] {:?} {:<52} {} nonfinite={} max_err={:.5}",
        dtype, name, if ok { "ok " } else { "BAD" }, nf, max_err
    );
    Ok(ok)
}

fn conv_with(w_cpu: &Tensor, x: &Tensor, pad: usize) -> Result<Tensor> {
    let w = w_cpu.to_dtype(x.dtype())?.to_device(x.device())?;
    x.conv2d(&w, pad, 1, 1, 1)
}

#[test]
#[ignore = "ROCm view-primitive probe (diagnostic)"]
fn rocm_view_primitives() {
    let dev = candle_gen::default_device().expect("ROCm device");
    let cpu = Device::Cpu;
    let mut bad: Vec<String> = Vec::new();
    for dtype in [DType::F32, DType::F16] {
        for &(c, h, w) in &[(128usize, 64usize, 48usize), (256, 112, 239)] {
            let tag = format!("C{c} {h}x{w}");
            let x4 = Tensor::randn(0f32, 1.0, (1, c, h, w), &cpu).unwrap();
            let x4b = Tensor::randn(0f32, 1.0, (4, c, h, w), &cpu).unwrap();
            let x5 = Tensor::randn(0f32, 1.0, (1, c, 3, h, w), &cpu).unwrap();
            let w_full = Tensor::randn(0f32, 0.03, (c, c, 3, 3), &cpu).unwrap();
            let w_half = Tensor::randn(0f32, 0.03, (c, c / 2, 3, 3), &cpu).unwrap();
            let mut run = |name: String, x: &Tensor, tol: f32, op: &dyn Fn(&Tensor) -> Result<Tensor>| {
                let full = format!("{tag} {name}");
                match prim_check(&dev, dtype, &full, x, tol, op) {
                    Ok(true) => {}
                    _ => bad.push(format!("{:?} {}", dtype, full)),
                }
            };
            // 4D views.
            run("narrow(H, off0).contiguous".into(), &x4, 0.0, &|x| x.narrow(2, 0, h / 2)?.contiguous());
            run("narrow(H, off5).contiguous".into(), &x4, 0.0, &|x| x.narrow(2, 5, h / 2)?.contiguous());
            run("narrow(H, off5) -> conv2d (no contiguous)".into(), &x4, 2e-2, &|x| {
                conv_with(&w_full, &x.narrow(2, 5, h / 2)?, 0)
            });
            run("narrow(C, off) (contiguous block w/ offset)".into(), &x4, 0.0, &|x| x.narrow(1, c / 4, c / 2)?.contiguous());
            run("narrow(C, off) -> conv2d".into(), &x4, 2e-2, &|x| conv_with(&w_half, &x.narrow(1, c / 4, c / 2)?, 1));
            run("narrow(N, off1,len2).contiguous (batch split)".into(), &x4b, 0.0, &|x| x.narrow(0, 1, 2)?.contiguous());
            run("narrow(N, off1,len2) -> conv2d (batch split)".into(), &x4b, 2e-2, &|x| conv_with(&w_full, &x.narrow(0, 1, 2)?, 1));
            // 5D temporal-path ops.
            run("broadcast_as(first frame x2).contiguous".into(), &x5, 0.0, &|x| {
                x.narrow(2, 0, 1)?.broadcast_as((1, c, 2, h, w))?.contiguous()
            });
            run("cat([rep, x], dim2) (causal pad)".into(), &x5, 0.0, &|x| {
                let rep = x.narrow(2, 0, 1)?.broadcast_as((1, c, 2, h, w))?.contiguous()?;
                Tensor::cat(&[&rep, x], 2)
            });
            for kd in 0..3usize {
                run(format!("tap kd={kd}: narrow(T).permute.contiguous"), &x5, 0.0, &move |x| {
                    x.narrow(2, kd, 1)?.permute((0, 2, 1, 3, 4))?.contiguous()
                });
            }
            for kd in 0..3usize {
                run(format!("tap kd={kd}: ...contiguous.reshape -> conv2d"), &x5, 2e-2, &|x| {
                    let m = x.narrow(2, kd, 1)?.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((1, c, h, w))?;
                    conv_with(&w_full, &m, 1)
                });
            }
            for kd in 0..3usize {
                run(format!("tap kd={kd}: index_select(T).permute.contiguous"), &x5, 0.0, &|x| {
                    let idx = Tensor::from_vec(vec![kd as u32], 1, x.device())?;
                    x.index_select(&idx, 2)?.permute((0, 2, 1, 3, 4))?.contiguous()
                });
            }
            for kd in 0..3usize {
                run(format!("tap kd={kd}: permute-first, narrow(T) -> conv2d"), &x5, 2e-2, &|x| {
                    let m = x.permute((0, 2, 1, 3, 4))?.contiguous()?.narrow(1, kd, 1)?.reshape((1, c, h, w))?;
                    conv_with(&w_full, &m, 1)
                });
            }
        }
    }
    assert!(bad.is_empty(), "view primitives wrong on ROCm:\n  {}", bad.join("\n  "));
}

// ---------------------------------------------------------------------------------------------
// Per-op offset probe: every backend op that takes a Layout, fed views with a NON-ZERO start offset
// (contiguous channel/batch blocks and strided H views), ROCm vs CPU.
//   cargo test -p candle-gen-seedvr2 --features rocm --test rocm_conv_check rocm_op_offsets -- --ignored --nocapture
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "ROCm per-op offset probe (diagnostic)"]
fn rocm_op_offsets() {
    use candle_gen::candle_nn::ops::softmax_last_dim;
    let dev = candle_gen::default_device().expect("ROCm device");
    let cpu = Device::Cpu;
    let mut bad: Vec<String> = Vec::new();
    for dtype in [DType::F32, DType::F16] {
        let tol = if dtype == DType::F32 { 1e-4f32 } else { 2e-2f32 };
        for &(c, h, w) in &[(64usize, 48usize, 40usize), (128, 112, 99)] {
            let tag = format!("C{c} {h}x{w}");
            let x = Tensor::randn(0f32, 1.0, (1, c, h, w), &cpu).unwrap();
            let x3 = Tensor::randn(0f32, 1.0, (4, 64, 48), &cpu).unwrap();
            let mut run = |name: &str, x: &Tensor, op: &dyn Fn(&Tensor) -> Result<Tensor>| {
                let full = format!("{tag} {name}");
                match prim_check(&dev, dtype, &full, x, tol, op) {
                    Ok(true) => {}
                    _ => bad.push(format!("{:?} {}", dtype, full)),
                }
            };
            // `blk` = contiguous channel block with offset; `hv` = strided H view with offset.
            macro_rules! unary_on_blk { ($n:expr, $f:expr) => { run(concat!("unary ", $n, " on narrow(C,off)"), &x, &|x| { let v = x.narrow(1, c / 4, c / 2)?; $f(&v) }); } }
            macro_rules! unary_on_hv  { ($n:expr, $f:expr) => { run(concat!("unary ", $n, " on narrow(H,off)"), &x, &|x| { let v = x.narrow(2, 5, h / 2)?; $f(&v) }); } }
            unary_on_blk!("neg", |v: &Tensor| v.neg());
            unary_on_blk!("exp", |v: &Tensor| v.exp());
            unary_on_blk!("silu", |v: &Tensor| v.silu());
            unary_on_blk!("tanh", |v: &Tensor| v.tanh());
            unary_on_blk!("sqr", |v: &Tensor| v.sqr());
            unary_on_blk!("affine", |v: &Tensor| v.affine(2.0, 1.0));
            unary_on_blk!("powf", |v: &Tensor| v.powf(2.0));
            unary_on_blk!("elu", |v: &Tensor| v.elu(1.0));
            unary_on_hv!("neg", |v: &Tensor| v.neg());
            unary_on_hv!("silu", |v: &Tensor| v.silu());
            unary_on_hv!("affine", |v: &Tensor| v.affine(2.0, 1.0));
            // binary (both operands offset views, different offsets)
            run("binary add blk+blk", &x, &|x| x.narrow(1, c / 8, c / 2)?.add(&x.narrow(1, c / 2, c / 2)?));
            run("binary mul blk*blk", &x, &|x| x.narrow(1, c / 8, c / 2)?.mul(&x.narrow(1, c / 2, c / 2)?));
            run("binary add hv+hv", &x, &|x| x.narrow(2, 3, h / 2)?.add(&x.narrow(2, 9, h / 2)?));
            // compare + where_cond
            run("cmp gt blk", &x, &|x| x.narrow(1, c / 8, c / 2)?.gt(&x.narrow(1, c / 2, c / 2)?));
            run("where_cond blk", &x, &|x| {
                let a = x.narrow(1, c / 8, c / 2)?;
                let b = x.narrow(1, c / 2, c / 2)?;
                a.gt(&b)?.where_cond(&a, &b)
            });
            // reductions + softmax
            run("sum(1) on blk", &x, &|x| x.narrow(1, c / 4, c / 2)?.sum(1));
            run("max(2) on blk", &x, &|x| x.narrow(1, c / 4, c / 2)?.max(2));
            run("sum(3) on hv", &x, &|x| x.narrow(2, 5, h / 2)?.sum(3));
            run("softmax_last_dim on blk", &x, &|x| softmax_last_dim(&x.narrow(1, c / 4, c / 2)?));
            // cast
            run("to_dtype on blk", &x, &|x| {
                let t = if x.dtype() == DType::F32 { DType::F16 } else { DType::F32 };
                x.narrow(1, c / 4, c / 2)?.to_dtype(t)
            });
            // index_select: source offset and ids offset
            run("index_select(dim2) src blk, ids offset view", &x, &|x| {
                let ids = Tensor::from_vec((0u32..16).collect::<Vec<_>>(), 16, x.device())?.narrow(0, 3, 5)?;
                x.narrow(1, c / 4, c / 2)?.index_select(&ids, 2)
            });
            // matmul with offset operands
            run("matmul batch-offset lhs/rhs", &x3, &|x| {
                let a = x.narrow(0, 1, 2)?;
                let b = x.narrow(0, 2, 2)?.transpose(1, 2)?.contiguous()?;
                a.matmul(&b)
            });
            run("matmul batch-offset, strided rhs", &x3, &|x| {
                let a = x.narrow(0, 1, 2)?;
                let b = x.narrow(0, 2, 2)?.transpose(1, 2)?;
                a.matmul(&b)
            });
        }
    }
    assert!(bad.is_empty(), "ops wrong on offset views:\n  {}", bad.join("\n  "));
}