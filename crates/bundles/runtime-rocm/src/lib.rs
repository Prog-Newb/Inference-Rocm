//! Supported AMD runtime (Windows, ROCm/HIP): an explicit Candle ROCm media catalog.
//!
//! Deliberately smaller than `runtime-cuda`: only providers that have been validated on ROCm are
//! composed in (currently SeedVR2). There is no LLM or audio lane here, so nothing is advertised that
//! cannot run on an AMD GPU. Providers are added to [`providers`] and [`media_registry`] as they are
//! validated; the NVIDIA, macOS and CPU bundles are unaffected.

pub use candle_gen as media;
pub use runtime_catalog::{
    core_llm, gen_core, memory_strategy, RuntimeCatalog, RuntimeCatalogSnapshot,
    VideoDecodeMemoryProfile,
};

/// The Candle backend crates this platform owns (ROCm-validated only).
pub mod providers {
    pub use candle_gen_seedvr2 as seedvr2;
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

/// No ROCm provider currently publishes a conservative video-decode memory profile.
pub fn conservative_video_decode_memory_profile(
    _provider_id: &str,
    _width: u32,
    _height: u32,
    _frames: u32,
) -> Option<VideoDecodeMemoryProfile> {
    None
}

fn media_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    candle_gen_seedvr2::provider_registry()
}

/// The catalog requires a declared snapshot preparer on every runtime. The ROCm bundle ships no LLM
/// or audio lane, so its preparer never claims a snapshot and fails loudly if it is ever driven.
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
/// Build the validated ROCm runtime composition: SeedVR2 media, no LLM/snapshot-preparer/audio lanes.
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
        assert_eq!(ids.len(), 3, "only SeedVR2 is composed into the ROCm bundle: {ids:?}");
        assert!(snapshot.text_llm_ids.is_empty());
        assert!(snapshot.audio_generator_ids.is_empty());
    }
}