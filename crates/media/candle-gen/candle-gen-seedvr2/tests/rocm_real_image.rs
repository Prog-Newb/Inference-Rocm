use candle_gen::candle_core::DType;
use candle_gen::gen_core::Image;
use candle_gen_seedvr2::config::DitConfig;
use candle_gen_seedvr2::pipeline::Seedvr2Pipeline;

const DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";

#[test]
#[ignore = "real SeedVR2 ROCm image test"]
fn seedvr2_rocm_real_image_2x() {
    let ckpt =
        std::env::var("SEEDVR2_CKPT").expect("SEEDVR2_CKPT is not set");

    let input =
        std::env::var("SEEDVR2_INPUT").expect("SEEDVR2_INPUT is not set");

    let device = candle_gen::default_device().expect("ROCm device");
    let dtype = match std::env::var("SEEDVR2_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        Ok("f32") => DType::F32,
        Err(_) => DType::F16,
        Ok(other) => panic!("unsupported SEEDVR2_DTYPE `{other}`; use f16, bf16, or f32"),
    };

    eprintln!("[seedvr2-rocm] device={device:?} dtype={dtype:?}");
    eprintln!("[seedvr2-rocm] input={input}");
    eprintln!("[seedvr2-rocm] checkpoint={ckpt}");

    let img = image::open(&input)
        .expect("open input image")
        .to_rgb8();

    let src_w = img.width() as usize;
    let src_h = img.height() as usize;

    let lr = Image {
        width: src_w as u32,
        height: src_h as u32,
        pixels: img.into_raw(),
    };

    let tgt_w = src_w * 2;
    let tgt_h = src_h * 2;

    eprintln!(
        "[seedvr2-rocm] input {}x{} -> target {}x{}",
        src_w, src_h, tgt_w, tgt_h
    );

    let cfg = DitConfig::seedvr2_3b();

    let t_load = std::time::Instant::now();
    let pipe =
        Seedvr2Pipeline::load(&ckpt, DIT_FILE, &cfg, dtype, &device)
            .expect("load SeedVR2 pipeline");

    eprintln!("[seedvr2-rocm] model loaded in {:?}", t_load.elapsed());

    let t_gen = std::time::Instant::now();

    let out = pipe
        .generate(&lr, tgt_w, tgt_h, 42, 0.0, None)
        .expect("SeedVR2 generate");

    eprintln!(
        "[seedvr2-rocm] generated {}x{} in {:?}",
        out.width,
        out.height,
        t_gen.elapsed()
    );

    assert_eq!(out.width, tgt_w as u32);
    assert_eq!(out.height, tgt_h as u32);

    let output = std::env::var("SEEDVR2_OUTPUT").unwrap_or_else(|_| {
        r"C:\SceneWorks-AMD\data\models\seedvr2_test_2x.png".to_string()
    });

    image::RgbImage::from_raw(out.width, out.height, out.pixels)
        .expect("construct output image")
        .save(&output)
        .expect("save output PNG");

    eprintln!("[seedvr2-rocm] OUTPUT: {output}");
}
