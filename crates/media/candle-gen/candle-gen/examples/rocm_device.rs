use candle_gen::{default_device, default_dtype};
use candle_gen::candle_core::{DType, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ROCm cmp/where test ===");

    let device = default_device()?;
    let dtype = default_dtype();

    println!("Device: {device:?}");
    println!("DType:  {dtype:?}");

    let a = Tensor::new(vec![0.0f32, 1.0, 2.0, 3.0], &device)?
        .to_dtype(dtype)?;

    let b = Tensor::new(vec![0.5f32, 1.0, 1.5, 3.5], &device)?
        .to_dtype(dtype)?;

    let mask = a.gt(&b)?;
    let mask_cpu = mask.to_vec1::<u8>()?;

    println!("CMP mask: {mask_cpu:?}");

    let on_true = Tensor::new(vec![10.0f32, 10.0, 10.0, 10.0], &device)?
        .to_dtype(dtype)?;

    let on_false = Tensor::new(vec![20.0f32, 20.0, 20.0, 20.0], &device)?
        .to_dtype(dtype)?;

    let selected = mask
        .where_cond(&on_true, &on_false)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    println!("WHERE result: {selected:?}");

    let scaled = Tensor::new(
        vec![0.0f32, 1.2, 127.5, 128.4, 254.6],
        &device,
    )?;

    let rgb8 = candle_gen::round_rgb8(&scaled)?.to_vec1::<u8>()?;

    println!("ROUND RGB8: {rgb8:?}");

    println!("=== ROCm cmp/where test complete ===");
    Ok(())
}
