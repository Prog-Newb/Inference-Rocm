//! Supported AMD runtime (Windows, ROCm/HIP): Candle on the ROCm backend.
//!
//! The *public surface* mirrors `runtime-cuda` (same module and item names) so SceneWorks' worker -- written
//! against `runtime_cuda::...` -- compiles unchanged against this bundle when built through the AMD build
//! configuration (`amd/`). The *catalog* is deliberately honest and smaller: only providers that have been
//! validated on ROCm are registered (currently SeedVR2), with no LLM or audio lane. Everything else is linked
//! for compilation only and is never advertised, so an AMD worker cannot claim work it cannot run.
//! Add a provider to [`media_registry`] once it has been validated on ROCm.

pub use candle_gen_catalog::{media, vae_tiling};
pub use candle_llm as llm;
pub use runtime_catalog::{
    core_llm, gen_core, memory_strategy, RuntimeCatalog, RuntimeCatalogSnapshot,
    VideoDecodeMemoryProfile,
};

/// Resolve a provider-owned conservative VAE decode profile (same contract as `runtime-cuda`).
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<VideoDecodeMemoryProfile> {
    candle_gen_catalog::conservative_video_decode_memory_profile(provider_id, width, height, frames)
}

/// The Candle backend crates, re-exported for source compatibility with `runtime-cuda`. Only the providers
/// registered in [`catalog`] (SeedVR2) are validated on ROCm; the rest are compile-surface only.
pub mod providers {
    pub use candle_gen_catalog::providers::*;
}

/// AMD GPUs have no NVFP4 tensor-core tier, so this bundle surfaces no advanced quant tiers.
pub fn nvfp4_quant_tiers() -> &'static [gen_core::Quant] {
    &[]
}

/// Platform label for this bundle; matches `RuntimeCatalog::platform`.
pub const PLATFORM: &str = "rocm";
/// The single tensor backend every provider in this bundle uses.
pub const BACKEND: &str = "candle";
/// Target triples this bundle is supported on (ROCm for Windows).
pub const SUPPORTED_TARGET_TRIPLES: &[&str] = &["x86_64-pc-windows-msvc"];
/// Native (non-Cargo) prerequisites required to build and run this bundle.
pub const NATIVE_PREREQUISITES: &[&str] = &[
    "AMD ROCm/HIP SDK 7.2+ runtime libraries (HIP, hipBLAS/hipBLASLt, rocBLAS, MIOpen, hipRTC)",
    "supported AMD Radeon GPU and driver (RDNA3 / gfx1100 validated)",
];

/// The catalog requires a declared snapshot preparer on every runtime. This bundle advertises no LLM or audio
/// lane, so its preparer never claims a snapshot and fails loudly if it is ever driven.
fn preparer_backend() -> &'static str {
    BACKEND
}

fn cannot_prepare(_spec: &core_llm::PrepareSpec) -> bool {
    false
}

fn never_prepare(_spec: &core_llm::PrepareSpec) -> core_llm::Result<core_llm::PrepareReport> {
    Err(core_llm::Error::Msg(
        "the ROCm runtime does not prepare LLM/audio snapshots".to_string(),
    ))
}

fn snapshot_preparers() -> core_llm::Result<core_llm::SnapshotPreparerRegistry> {
    core_llm::SnapshotPreparerRegistryBuilder::new()
        .register(core_llm::SnapshotPreparerRegistration {
            backend: preparer_backend,
            can_prepare: cannot_prepare,
            prepare: never_prepare,
        })
        .build()
}

/// Providers validated on ROCm. Add to this registry as providers are validated.
fn media_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    candle_gen_seedvr2::provider_registry()
}

/// Build the validated ROCm runtime composition: SeedVR2 media; no LLM or audio lane advertised.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    RuntimeCatalog::try_new(
        PLATFORM,
        BACKEND,
        media_registry(),
        core_llm::TextLlmRegistryBuilder::new().build(),
        snapshot_preparers(),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn rocm_catalog_is_explicit_and_exposes_only_seedvr2() {
        let snapshot = super::catalog().expect("rocm catalog validates").snapshot();
        assert_eq!(snapshot.platform, "rocm");
        assert_eq!(snapshot.backend, "candle");
        let ids: Vec<&str> = snapshot.generator_ids.iter().map(|s| s.as_str()).collect();
        for id in ["seedvr2", "seedvr2_3b", "seedvr2_7b"] {
            assert!(ids.contains(&id), "missing {id} in {ids:?}");
        }
        assert_eq!(ids.len(), 3, "only SeedVR2 is advertised by the ROCm bundle: {ids:?}");
        assert!(snapshot.text_llm_ids.is_empty());
        assert!(snapshot.audio_generator_ids.is_empty());
        assert!(super::nvfp4_quant_tiers().is_empty());
    }

    /// The compile-surface contract: the names SceneWorks' worker uses on `runtime_cuda` resolve here too.
    #[test]
    fn exposes_runtime_cuda_compatible_surface() {
        let _ = super::providers::seedvr2::provider_registry;
        let _ = super::media::default_device;
        let _ = super::vae_tiling;
        let _ = super::llm::text_registry;
        assert_eq!(super::BACKEND, "candle");
    }
}
