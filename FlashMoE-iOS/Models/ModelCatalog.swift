/*
 * ModelCatalog.swift — Curated registry of downloadable pre-packed Flash-MoE models
 *
 * Each entry maps to a public HuggingFace repo containing pre-packed weights
 * ready for the Flash-MoE engine (config.json, model_weights.bin, packed_experts/).
 *
 * Download URLs: https://huggingface.co/{repoId}/resolve/main/{filename}
 */

import Foundation

// MARK: - Data Types

struct RepoFile: Identifiable, Codable, Sendable {
    var id: String { filename }
    let filename: String
    let sizeBytes: UInt64
}

struct CatalogEntry: Identifiable, Codable, Sendable {
    let id: String
    let displayName: String
    let repoId: String
    let description: String
    let totalSizeBytes: UInt64
    let quantization: String
    let expertLayers: Int
    let files: [RepoFile]

    var totalSizeGB: Double {
        Double(totalSizeBytes) / (1024.0 * 1024.0 * 1024.0)
    }

    func downloadURL(for file: RepoFile) -> URL {
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(file.filename)")!
    }
}

// MARK: - Curated Model Catalog

enum ModelCatalog {

    /// Pre-packed models available for download.
    /// New models are added in app updates.
    static let models: [CatalogEntry] = [
        // -- Qwen 3.5 35B-A3B 4-bit (full quality) --
        CatalogEntry(
            id: "qwen3.5-35b-a3b-q4",
            displayName: "Qwen 3.5 35B-A3B",
            repoId: "alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE",
            description: "Compact 35B MoE model. 3B active params per token. Good for 8GB devices.",
            totalSizeBytes: 19_500_000_000,
            quantization: "4-bit",
            expertLayers: 40,
            files: makeFileList(
                configFiles: [
                    ("config.json", 3_809),
                    ("model_weights.json", 251_539),
                    ("model_weights.bin", 1_378_869_376),
                    ("vocab.bin", 3_360_287),
                    ("tokenizer.json", 19_989_343),
                    ("tokenizer.bin", 8_201_040),
                ],
                expertLayers: 40,
                expertLayerSize: 452_984_832
            )
        ),

        // -- Qwen 3.5 35B-A3B Tiered (hot=4-bit, cold=2-bit, ~12GB smaller) --
        CatalogEntry(
            id: "qwen3.5-35b-a3b-tiered",
            displayName: "Qwen 3.5 35B-A3B Tiered",
            repoId: "alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE",
            description: "Tiered quantization: hot experts 4-bit, cold 2-bit. ~12GB experts (vs 18GB full). Faster with slight quality trade-off.",
            totalSizeBytes: 13_424_643_082,
            quantization: "tiered (4-bit/2-bit)",
            expertLayers: 40,
            files: makeTieredFileList()
        ),
    ]

    // MARK: - Helpers

    private static func makeTieredFileList() -> [RepoFile] {
        let configFiles: [(String, UInt64)] = [
            ("config.json", 3_809),
            ("model_weights.json", 251_539),
            ("model_weights.bin", 1_378_869_376),
            ("vocab.bin", 3_360_287),
            ("tokenizer.json", 19_989_343),
            ("tokenizer.bin", 8_201_040),
        ]
        // Variable-size tiered expert layers (hot=4-bit, cold=2-bit)
        let layerSizes: [UInt64] = [
            337_379_328, 349_175_808, 342_097_920, 331_087_872, 320_077_824,
            301_989_888, 301_989_888, 289_406_976, 285_474_816, 294_125_568,
            305_922_048, 306_708_480, 297_271_296, 293_339_136, 282_329_088,
            288_620_544, 287_834_112, 292_552_704, 280_756_224, 287_834_112,
            282_329_088, 283_115_520, 301_989_888, 305_135_616, 294_125_568,
            294_125_568, 281_542_656, 292_552_704, 296_484_864, 298_844_160,
            289_406_976, 291_766_272, 301_989_888, 302_776_320, 305_135_616,
            300_417_024, 298_057_728, 304_349_184, 301_989_888, 309_854_208,
        ]
        var files = configFiles.map { RepoFile(filename: $0.0, sizeBytes: $0.1) }
        files.append(RepoFile(filename: "packed_experts_tiered/tiered_manifest.json", sizeBytes: 1_005_120))
        for (i, size) in layerSizes.enumerated() {
            files.append(RepoFile(
                filename: String(format: "packed_experts_tiered/layer_%02d.bin", i),
                sizeBytes: size
            ))
        }
        return files
    }

    private static func makeFileList(
        configFiles: [(String, UInt64)],
        expertLayers: Int,
        expertLayerSize: UInt64
    ) -> [RepoFile] {
        var files = configFiles.map { RepoFile(filename: $0.0, sizeBytes: $0.1) }
        for i in 0..<expertLayers {
            files.append(RepoFile(
                filename: String(format: "packed_experts/layer_%02d.bin", i),
                sizeBytes: expertLayerSize
            ))
        }
        return files
    }
}
