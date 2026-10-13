import Foundation
import Testing
@testable import MLXLoRAStudio

@Suite("Choice verifier")
struct ChoiceVerifierTests {
    @Test("chat folder with a report gets the small-dataset SFT recipe")
    func chatFolderAppliesSFTRecipe() throws {
        let folder = try makeChatFolder(
            trainRows: 155,
            validRows: 26,
            longestTokens: 211
        )
        defer { try? FileManager.default.removeItem(at: folder) }

        let assessment = ChoiceVerifier.assessDataset(folder.path, currentMode: .dpo)
        let preset = try #require(assessment.preset)
        #expect(assessment.verdict.status == .acceptable)
        #expect(assessment.verdict.detail.contains("155 train rows"))
        #expect(assessment.verdict.detail.contains("211"))
        #expect(preset.trainMode == .sft)
        #expect(preset.maskPrompt)
        #expect(preset.epochs == 5)
        #expect(preset.maxSeqLength == 512)
        #expect(preset.batchSize == 1)
        #expect(preset.gradientAccumulationSteps == 8)
        #expect(preset.dropout == 0.05)
        #expect(preset.rank == 8)
        #expect(preset.scale == 20)
        #expect(preset.valBatches == 26)
        #expect(!preset.test)
        #expect(preset.learningRateSchedule == .cosineDecay)
        #expect(preset.lrWarmupSteps == 10)
        #expect(preset.clearDatasetSystemPrompt)
        #expect(preset.chatFeature.isEmpty)

        var config = TrainingConfig()
        config.trainMode = .dpo
        config.datasetSystemPrompt = "leftover"
        config.quantization = .fourBit
        config.applyDatasetPreset(preset)
        #expect(config.trainMode == .sft)
        #expect(config.maskPrompt)
        #expect(config.epochs == 5)
        #expect(config.maxSeqLength == 512)
        #expect(config.datasetSystemPrompt.isEmpty)
        #expect(config.quantization == .fourBit)
    }

    @Test("an existing compatible mode is kept")
    func preferenceKeepsCPO() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let row = #"{"prompt":"q","chosen":"a","rejected":"b"}"# + "\n"
        try Data(row.utf8).write(to: folder.appending(path: "train.jsonl"))

        let assessment = ChoiceVerifier.assessDataset(folder.path, currentMode: .cpo)
        #expect(assessment.preset?.trainMode == .cpo)
        #expect(assessment.verdict.status == .acceptable)
    }

    @Test("a folder without train.jsonl is rejected")
    func missingTrainFileIsRejected() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data("notes".utf8).write(to: folder.appending(path: "README.md"))

        let assessment = ChoiceVerifier.assessDataset(folder.path, currentMode: .sft)
        #expect(assessment.verdict.status == .rejected)
        #expect(assessment.preset == nil)
    }

    @Test("an MTPLX vision package is not a trainable text model")
    func mtplxVisionModelIsRejected() throws {
        let folder = try makeModelFolder(vision: true, mtplx: true, bits: 6, group: 64)
        defer { try? FileManager.default.removeItem(at: folder) }

        let trainable = ChoiceVerifier.assessModel(folder.path, role: .trainableText)
        #expect(trainable.verdict.status == .rejected)
        #expect(trainable.verdict.title == "Not a trainable text model")
        #expect(trainable.quantization == nil)

        let original = ChoiceVerifier.assessModel(folder.path, role: .originalVLM)
        #expect(original.verdict.status == .acceptable)
        #expect(original.quantization == Quantization.none)
    }

    @Test("an already quantized text model turns load-time quantization off")
    func quantizedTextModelUsesNone() throws {
        let folder = try makeModelFolder(vision: false, mtplx: false, bits: 6, group: 64)
        defer { try? FileManager.default.removeItem(at: folder) }

        let assessment = ChoiceVerifier.assessModel(folder.path, role: .trainableText)
        #expect(assessment.verdict.status == .acceptable)
        #expect(assessment.quantization == Quantization.none)
        #expect(assessment.verdict.detail.contains("group size 64"))
    }

    private func makeChatFolder(trainRows: Int, validRows: Int, longestTokens: Int) throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let row = #"{"messages":[{"role":"user","content":"How did the approach differ?"},{"role":"assistant","content":"The new method operates openly."}]}"# + "\n"
        try Data(String(repeating: row, count: trainRows).utf8).write(to: folder.appending(path: "train.jsonl"))
        try Data(String(repeating: row, count: validRows).utf8).write(to: folder.appending(path: "valid.jsonl"))
        let report: [String: Any] = [
            "format": "chat",
            "counts": ["train": trainRows, "valid": validRows, "test": 0],
            "longest_row_tokens": longestTokens,
            "max_seq_length": 2048
        ]
        let data = try JSONSerialization.data(withJSONObject: report)
        try data.write(to: folder.appending(path: "report.json"))
        return folder
    }

    private func makeModelFolder(vision: Bool, mtplx: Bool, bits: Int, group: Int) throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var config: [String: Any] = [
            "model_type": vision ? "qwen3_5" : "qwen3",
            "architectures": [vision ? "Qwen3_5ForConditionalGeneration" : "Qwen3ForCausalLM"],
            "quantization": ["bits": bits, "group_size": group, "mode": "affine"]
        ]
        if vision {
            config["vision_config"] = ["depth": 1]
        }
        if mtplx {
            config["mtplx_mtp_contract"] = ["mtp_quant_mode": "affine"]
            try Data("mtp".utf8).write(to: folder.appending(path: "mtp.safetensors"))
        }
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: folder.appending(path: "config.json"))
        try Data("weights".utf8).write(to: folder.appending(path: "model.safetensors"))
        return folder
    }
}
