import Foundation

enum ChoiceStatus: String, Equatable, Sendable {
    case acceptable
    case caution
    case rejected
}

struct ChoiceVerdict: Equatable, Sendable {
    var status: ChoiceStatus
    var title: String
    var detail: String
    var appliedSummary: String?
    var chips: [String]

    init(
        status: ChoiceStatus,
        title: String,
        detail: String,
        appliedSummary: String? = nil,
        chips: [String] = []
    ) {
        self.status = status
        self.title = title
        self.detail = detail
        self.appliedSummary = appliedSummary
        self.chips = chips
    }
}

struct ModelAssessment: Equatable, Sendable {
    var verdict: ChoiceVerdict
    /// Set when the checkpoint is already quantized and load-time quantization must stay off.
    var quantization: Quantization?
    var quantizationSummary: String?
}

struct DatasetTrainingPreset: Equatable, Sendable {
    var trainMode: TrainMode
    var sftLossType: SFTLossType
    var prefersLoRA: Bool
    var maskPrompt: Bool
    var maxSeqLength: Int
    var batchSize: Int
    var gradientAccumulationSteps: Int
    var epochs: Int
    var iters: Int
    var learningRate: Double
    var learningRateSchedule: LearningRateScheduleKind
    var lrWarmupSteps: Int
    var lrDecayFraction: Double
    var lrFinal: Double
    var stepsPerReport: Int
    var stepsPerEval: Int
    var saveEvery: Int
    var valBatches: Int
    var test: Bool
    var rank: Int
    var scale: Double
    var dropout: Double
    var numLayers: Int
    var gradCheckpoint: Bool
    var efficientLongContext: Bool
    var qatEnable: Bool
    var promptFeature: String
    var completionFeature: String
    var chosenFeature: String
    var rejectedFeature: String
    var chatFeature: String
    var textFeature: String
    var systemFeature: String
    var answerFeature: String
    var ftpoContextFeature: String
    var ftpoRejectedFeature: String
    var ftpoChosenFeature: String
    var clearDatasetSystemPrompt: Bool
    var summary: String
    var chips: [String]
}

struct DatasetAssessment: Equatable, Sendable {
    var verdict: ChoiceVerdict
    var preset: DatasetTrainingPreset?
}

enum ModelChoiceRole: Sendable {
    case trainableText
    case originalVLM
}

enum ChoiceVerifier {
    static func assessModel(_ rawPath: String, role: ModelChoiceRole) -> ModelAssessment {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .caution,
                title: "No model selected",
                detail: "Choose a text MLX folder or a Hugging Face model repo."
            ))
        }

        if trimmed == TrainingConfig.defaultModel {
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .acceptable,
                title: "Default base model",
                detail: "Qwen3-0.6B is a full-precision Hugging Face repo. 4-bit is the usual load setting for it."
            ))
        }

        let expanded = expand(trimmed)
        if FileManager.default.fileExists(atPath: expanded) {
            return assessLocalModel(at: URL(fileURLWithPath: expanded, isDirectory: true), displayPath: trimmed, role: role)
        }

        if looksLikeHuggingFaceID(trimmed) {
            return assessRemoteModel(trimmed)
        }

        return ModelAssessment(verdict: ChoiceVerdict(
            status: .rejected,
            title: "Model path is not available",
            detail: "The app cannot see this folder. Browse to it so macOS grants access, or pick a Hugging Face repo id such as owner/name."
        ))
    }

    static func assessDataset(_ rawPath: String, currentMode: TrainMode) -> DatasetAssessment {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return DatasetAssessment(verdict: ChoiceVerdict(
                status: .caution,
                title: "No dataset selected",
                detail: "Choose a folder with train.jsonl, or a Hugging Face dataset repo."
            ))
        }

        if TrainingConfig.builtInDatasets.contains(trimmed) {
            return DatasetAssessment(verdict: ChoiceVerdict(
                status: .acceptable,
                title: "Default dataset",
                detail: "This Hugging Face dataset is the bundled match for an algorithm in Studio. Its columns already fit that trainer."
            ))
        }

        let expanded = expand(trimmed)
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory) {
            return assessLocalDataset(
                at: URL(fileURLWithPath: expanded, isDirectory: isDirectory.boolValue),
                isDirectory: isDirectory.boolValue,
                currentMode: currentMode
            )
        }

        if looksLikeHuggingFaceID(trimmed) {
            return DatasetAssessment(verdict: ChoiceVerdict(
                status: .caution,
                title: "Remote dataset",
                detail: "Studio will download this Hugging Face repo when the run starts. Settings are applied automatically for a local folder whose rows can be checked."
            ))
        }

        return DatasetAssessment(verdict: ChoiceVerdict(
            status: .rejected,
            title: "Dataset path is not available",
            detail: "The app cannot see this folder. Browse to it so macOS grants access. A local dataset folder needs train.jsonl."
        ))
    }

    // MARK: - Models

    private static func assessRemoteModel(_ identifier: String) -> ModelAssessment {
        let lowered = identifier.lowercased()
        if lowered.contains("gguf") {
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "GGUF is not a training format",
                detail: "Studio trains MLX safetensors models. Convert this repo to MLX, or pick an mlx-community checkpoint."
            ))
        }
        if let note = prequantizedNameNote(lowered) {
            return ModelAssessment(
                verdict: ChoiceVerdict(
                    status: .acceptable,
                    title: "Already quantized",
                    detail: note
                ),
                quantization: Quantization.none,
                quantizationSummary: "Quantization set to None so this checkpoint is not quantized again."
            )
        }
        return ModelAssessment(verdict: ChoiceVerdict(
            status: .caution,
            title: "Remote model",
            detail: "Studio will download this repo when the run starts. Use 4-bit or 8-bit for a full-precision model, and None if the repo is already quantized."
        ))
    }

    private static func assessLocalModel(at url: URL, displayPath: String, role: ModelChoiceRole) -> ModelAssessment {
        let folder = resolvedModelFolder(url)
        if let gguf = ggufFile(in: folder) {
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "GGUF is not a training format",
                detail: "\(gguf) cannot be fine-tuned here. Studio needs an MLX folder with config.json and safetensors weights."
            ))
        }

        let configURL = folder.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "Not an MLX model folder",
                detail: "The model folder needs config.json next to its weight files. \(displayPath) does not have one."
            ))
        }
        guard hasSafetensors(in: folder) else {
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "Model weights are missing",
                detail: "config.json is present, but there is no model.safetensors or safetensors index."
            ))
        }

        guard let config = jsonObject(at: configURL) else {
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "Model config could not be read",
                detail: "config.json is not valid JSON, or macOS blocked access. Browse to the folder and try again."
            ))
        }

        let vision = isVisionPackage(config, folder: folder)
        let serving = isMTPLXPackage(config, folder: folder)
        let quantNote = quantizationDescription(config)

        if vision && role == .trainableText {
            let kind = serving ? "MTPLX vision model" : "vision-language model"
            return ModelAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "Not a trainable text model",
                detail: "This is a \(kind). Studio fine-tunes a text MLX model. Put a matching text-only checkpoint in Model, and use this folder as the original VLM if you want the vision tower merged back afterwards."
            ))
        }

        if vision && role == .originalVLM {
            var detail = "The vision tower and processor in this folder can be kept for the final export. Train a text model that matches its language trunk."
            if serving {
                detail += " The MTP speculative-decoding head is not trained."
            }
            return ModelAssessment(
                verdict: ChoiceVerdict(status: .acceptable, title: "Usable as the original VLM", detail: detail),
                quantization: quantNote == nil ? nil : Quantization.none,
                quantizationSummary: quantNote.map { _ in "Quantization set to None because these weights are already quantized." }
            )
        }

        var detail = "MLX text checkpoint. LoRA can train on these weights."
        if let quantNote {
            detail = "Already quantized (\(quantNote)). Quantization is set to None so Studio does not quantize it again at a different group size."
        }
        return ModelAssessment(
            verdict: ChoiceVerdict(status: .acceptable, title: "Acceptable model", detail: detail),
            quantization: quantNote == nil ? nil : Quantization.none,
            quantizationSummary: quantNote == nil ? nil : "Quantization set to None."
        )
    }

    // MARK: - Datasets

    private static func assessLocalDataset(at url: URL, isDirectory: Bool, currentMode: TrainMode) -> DatasetAssessment {
        if !isDirectory {
            guard url.pathExtension.lowercased() == "jsonl" else {
                return DatasetAssessment(verdict: ChoiceVerdict(
                    status: .rejected,
                    title: "Not a dataset file",
                    detail: "Pick a folder that contains train.jsonl, or a single .jsonl file."
                ))
            }
            return assessJSONL(train: url, valid: nil, test: nil, report: nil, currentMode: currentMode)
        }

        if let train = locateTrainJSONL(in: url) {
            let valid = train.deletingLastPathComponent().appendingPathComponent("valid.jsonl")
            let test = train.deletingLastPathComponent().appendingPathComponent("test.jsonl")
            let report = reportURL(near: train, root: url)
            return assessJSONL(
                train: train,
                valid: FileManager.default.fileExists(atPath: valid.path) ? valid : nil,
                test: FileManager.default.fileExists(atPath: test.path) ? test : nil,
                report: report,
                currentMode: currentMode
            )
        }

        if hasParquetSplit(in: url) || hasOutputFull(in: url) {
            return DatasetAssessment(verdict: ChoiceVerdict(
                status: .acceptable,
                title: "Dataset folder needs a quick conversion",
                detail: "Studio can turn output_full.jsonl or the train parquet split into train.jsonl when the run starts. Open the folder after that conversion if you want settings chosen from the rows."
            ))
        }

        return DatasetAssessment(verdict: ChoiceVerdict(
            status: .rejected,
            title: "Not a training dataset",
            detail: "This folder has no train.jsonl. Studio looks for train.jsonl, and can also use valid.jsonl and test.jsonl beside it."
        ))
    }

    private static func assessJSONL(
        train: URL,
        valid: URL?,
        test: URL?,
        report: URL?,
        currentMode: TrainMode
    ) -> DatasetAssessment {
        let rows = sampleRows(from: train, limit: 8)
        if rows.isEmpty {
            return DatasetAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "Training file is empty",
                detail: "\(train.lastPathComponent) has no JSON rows."
            ))
        }

        let reportObject = report.flatMap { jsonObject(at: $0) }
        let kind = detectKind(rows: rows, report: reportObject)
        guard let kind else {
            let keys = rows.first.map { $0.keys.sorted().joined(separator: ", ") } ?? "none"
            return DatasetAssessment(verdict: ChoiceVerdict(
                status: .rejected,
                title: "Dataset columns are not recognized",
                detail: "The first row has keys \(keys). Studio can train chat messages, prompt and completion, raw text, preference pairs, or prompt and answer rows."
            ))
        }

        let trainRows = rowCount(reportObject, key: "train") ?? countRows(at: train)
        let validRows = valid == nil ? 0 : (rowCount(reportObject, key: "valid") ?? countRows(at: valid!))
        let testRows = test == nil ? 0 : (rowCount(reportObject, key: "test") ?? countRows(at: test!))
        let longest = rowCountValue(reportObject?["longest_row_tokens"]) ?? rows.map(estimatedTokens).max()
        let preset = makePreset(
            kind: kind,
            currentMode: currentMode,
            trainRows: trainRows,
            validRows: validRows,
            testRows: testRows,
            longestTokens: longest
        )
        let lengthNote = longest.map { " Longest row is about \($0) tokens." } ?? ""
        let splitNote = "\(trainRows) train rows" + (validRows > 0 ? " and \(validRows) validation rows." : ".")
        return DatasetAssessment(
            verdict: ChoiceVerdict(
                status: .acceptable,
                title: "Acceptable dataset",
                detail: "\(kind.displayName). \(splitNote)\(lengthNote)"
            ),
            preset: preset
        )
    }

    // MARK: - Presets

    private static func makePreset(
        kind: DetectedDatasetKind,
        currentMode: TrainMode,
        trainRows: Int,
        validRows: Int,
        testRows: Int,
        longestTokens: Int?
    ) -> DatasetTrainingPreset {
        let mode = kind.compatibleModes.contains(currentMode) ? currentMode : kind.preferredMode
        let batchSize = 1
        let accumulation = 8
        let epochs = epochCount(trainRows: trainRows)
        let stepsPerEpoch = max(1, Int((Double(max(trainRows, 1)) / Double(batchSize * accumulation)).rounded(.up)))
        let totalSteps = epochs > 0 ? epochs * stepsPerEpoch : 1000
        let warmup = max(1, min(40, totalSteps / 10))
        let checkpointEvery = epochs > 0 ? max(10, stepsPerEpoch) : 100
        let reportEvery = epochs > 0 ? max(1, min(10, max(1, stepsPerEpoch / 4))) : 10
        let sequence = sequenceLength(forEstimatedTokens: longestTokens)
        let dropout = trainRows > 0 && trainRows < 500 ? 0.05 : 0.0
        let columns = kind.columns
        let summary = presetSummary(
            mode: mode,
            epochs: epochs,
            sequence: sequence,
            dropout: dropout,
            maskPrompt: kind.masksPrompt
        )
        return DatasetTrainingPreset(
            trainMode: mode,
            sftLossType: .nll,
            prefersLoRA: true,
            maskPrompt: kind.masksPrompt,
            maxSeqLength: sequence,
            batchSize: batchSize,
            gradientAccumulationSteps: accumulation,
            epochs: epochs,
            iters: epochs > 0 ? 1000 : 1000,
            learningRate: mode == .sft ? 0.00001 : 0.000005,
            learningRateSchedule: .cosineDecay,
            lrWarmupSteps: warmup,
            lrDecayFraction: 0.9,
            lrFinal: 0.000002,
            stepsPerReport: reportEvery,
            stepsPerEval: checkpointEvery,
            saveEvery: checkpointEvery,
            valBatches: validRows > 0 ? min(validRows, 50) : 25,
            test: testRows > 0,
            rank: 8,
            scale: 20,
            dropout: dropout,
            numLayers: 16,
            gradCheckpoint: true,
            efficientLongContext: sequence >= 4096,
            qatEnable: false,
            promptFeature: columns.prompt,
            completionFeature: columns.completion,
            chosenFeature: columns.chosen,
            rejectedFeature: columns.rejected,
            chatFeature: columns.chat,
            textFeature: columns.text,
            systemFeature: columns.system,
            answerFeature: columns.answer,
            ftpoContextFeature: columns.ftpoContext,
            ftpoRejectedFeature: columns.ftpoRejected,
            ftpoChosenFeature: columns.ftpoChosen,
            clearDatasetSystemPrompt: columns.system.isEmpty,
            summary: summary,
            chips: presetChips(mode: mode, epochs: epochs, sequence: sequence, maskPrompt: kind.masksPrompt, dropout: dropout)
        )
    }

    private static func epochCount(trainRows: Int) -> Int {
        if trainRows <= 0 || trainRows > 2000 { return 0 }
        if trainRows <= 400 { return 5 }
        return 3
    }

    private static func sequenceLength(forEstimatedTokens tokens: Int?) -> Int {
        guard let tokens, tokens > 0 else { return 2048 }
        let needed = tokens + 64
        for bucket in [512, 1024, 2048, 4096, 8192, 16384] where bucket >= needed {
            return bucket
        }
        return 16384
    }

    private static func presetSummary(mode: TrainMode, epochs: Int, sequence: Int, dropout: Double, maskPrompt: Bool) -> String {
        let length = epochs > 0 ? "\(epochs) epochs" : "1000 iterations"
        let mask = maskPrompt ? "mask prompt on" : "loss on the full text"
        let drop = dropout > 0 ? String(format: ", dropout %.2f", dropout) : ""
        return "Applied \(mode.title): \(mask), \(length), batch 1 × 8, max sequence \(sequence), LoRA rank 8\(drop), cosine schedule."
    }

    private static func presetChips(mode: TrainMode, epochs: Int, sequence: Int, maskPrompt: Bool, dropout: Double) -> [String] {
        var chips = [mode.title, epochs > 0 ? "\(epochs) epochs" : "1000 iters", "Max seq \(sequence)"]
        if maskPrompt { chips.append("Mask prompt") }
        chips.append("LoRA r8")
        if dropout > 0 { chips.append("Dropout 0.05") }
        return chips
    }

    // MARK: - Detection

    private static func detectKind(rows: [[String: Any]], report: [String: Any]?) -> DetectedDatasetKind? {
        let keys = Set(rows.flatMap(\.keys))
        if let format = (report?["format"] as? String)?.lowercased(), format == "chat",
           let messages = firstKey(in: keys, candidates: ["messages", "conversations", "conversation"]) {
            return .chat(messages: messages, system: firstKey(in: keys, candidates: ["system", "system_prompt"]))
        }
        if let context = firstKey(in: keys, candidates: ["context_with_chat_template", "context"]),
           let rejected = firstKey(in: keys, candidates: ["rejected_decoded", "rejected"]),
           let chosen = firstKey(in: keys, candidates: ["multi_chosen_decoded", "chosen"]) {
            return .ftpo(context: context, rejected: rejected, chosen: chosen)
        }
        if let messages = firstKey(in: keys, candidates: ["messages", "conversations", "conversation"]) {
            return .chat(messages: messages, system: firstKey(in: keys, candidates: ["system", "system_prompt"]))
        }
        if let chosen = firstKey(in: keys, candidates: ["chosen"]),
           let rejected = firstKey(in: keys, candidates: ["rejected"]) {
            return .preference(
                prompt: firstKey(in: keys, candidates: ["prompt", "question", "instruction"]),
                chosen: chosen,
                rejected: rejected
            )
        }
        if let prompt = firstKey(in: keys, candidates: ["prompt", "question", "instruction"]),
           let completion = firstKey(in: keys, candidates: ["completion", "response", "output"]) {
            return .promptCompletion(prompt: prompt, completion: completion)
        }
        if let prompt = firstKey(in: keys, candidates: ["prompt", "question"]),
           let answer = firstKey(in: keys, candidates: ["answer"]) {
            return .reward(prompt: prompt, answer: answer)
        }
        if let text = firstKey(in: keys, candidates: ["text", "content"]) {
            return .text(text)
        }
        if let prompt = firstKey(in: keys, candidates: ["prompt", "question", "instruction"]) {
            return .promptOnly(prompt)
        }
        return nil
    }

    private static func firstKey(in keys: Set<String>, candidates: [String]) -> String? {
        candidates.first { keys.contains($0) }
    }

    // MARK: - Files

    private static func locateTrainJSONL(in root: URL) -> URL? {
        var candidates: [URL] = []
        let generated = root.appendingPathComponent("generated-data", isDirectory: true)
        candidates.append(generated.appendingPathComponent("data", isDirectory: true))
        candidates.append(generated)
        candidates.append(root.appendingPathComponent("data", isDirectory: true))
        candidates.append(root)
        for folder in candidates {
            let train = folder.appendingPathComponent("train.jsonl")
            if FileManager.default.fileExists(atPath: train.path) {
                return train
            }
        }
        return nil
    }

    private static func reportURL(near train: URL, root: URL) -> URL? {
        let besideTrain = train.deletingLastPathComponent().appendingPathComponent("report.json")
        if FileManager.default.fileExists(atPath: besideTrain.path) { return besideTrain }
        let atRoot = root.appendingPathComponent("report.json")
        if FileManager.default.fileExists(atPath: atRoot.path) { return atRoot }
        return nil
    }

    private static func hasParquetSplit(in root: URL) -> Bool {
        let names = ["train-00000-of-00001.parquet"]
        let folders = [root, root.appendingPathComponent("data"), root.appendingPathComponent("generated-data")]
        return folders.contains { folder in
            names.contains { FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path) }
        }
    }

    private static func hasOutputFull(in root: URL) -> Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent("output_full.jsonl").path)
    }

    private static func resolvedModelFolder(_ url: URL) -> URL {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return url.deletingLastPathComponent()
        }
        return url
    }

    private static func hasSafetensors(in folder: URL) -> Bool {
        if FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.safetensors").path) { return true }
        if FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.safetensors.index.json").path) { return true }
        guard let files = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) else {
            return false
        }
        return files.contains { $0.pathExtension == "safetensors" }
    }

    private static func ggufFile(in folder: URL) -> String? {
        guard let files = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) else {
            return nil
        }
        return files.first { $0.pathExtension.lowercased() == "gguf" }?.lastPathComponent
    }

    private static func isVisionPackage(_ config: [String: Any], folder: URL) -> Bool {
        if config["vision_config"] != nil { return true }
        let architectures = config["architectures"] as? [String] ?? []
        if architectures.contains(where: { $0.contains("ConditionalGeneration") }) { return true }
        if FileManager.default.fileExists(atPath: folder.appendingPathComponent("model-vision.safetensors").path) { return true }
        if weightIndexLooksLikeVision(in: folder) { return true }
        return false
    }

    private static func isMTPLXPackage(_ config: [String: Any], folder: URL) -> Bool {
        if config["mtplx_mtp_contract"] != nil || config["mlx_lm_extra_tensors"] != nil { return true }
        return FileManager.default.fileExists(atPath: folder.appendingPathComponent("mtp.safetensors").path)
            || FileManager.default.fileExists(atPath: folder.appendingPathComponent("mtplx_runtime.json").path)
    }

    private static func weightIndexLooksLikeVision(in folder: URL) -> Bool {
        let index = folder.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: index),
              let text = String(data: data.prefix(12_000), encoding: .utf8) else { return false }
        return text.contains("language_model.") || text.contains("vision_tower")
    }

    private static func quantizationDescription(_ config: [String: Any]) -> String? {
        let block = (config["quantization"] as? [String: Any]) ?? (config["quantization_config"] as? [String: Any])
        guard let block else { return nil }
        let bits = intValue(block["bits"])
        let group = intValue(block["group_size"])
        let mode = block["mode"] as? String
        var parts: [String] = []
        if let bits { parts.append("\(bits)-bit") }
        if let mode, !mode.isEmpty { parts.append(mode) }
        if let group { parts.append("group size \(group)") }
        guard !parts.isEmpty else { return "quantized" }
        return parts.joined(separator: ", ")
    }

    private static func prequantizedNameNote(_ lowered: String) -> String? {
        if lowered.contains("nvfp4") { return "The repo name says NVFP4. Quantization is set to None so it is not quantized again." }
        if lowered.contains("mxfp4") { return "The repo name says MXFP4. Quantization is set to None so it is not quantized again." }
        if lowered.contains("8bit") || lowered.contains("-q8") || lowered.contains("q8_") {
            return "The repo name says 8-bit. Quantization is set to None so it is not quantized again."
        }
        if lowered.contains("6bit") || lowered.contains("q6") {
            return "The repo name says 6-bit. Quantization is set to None so it is not quantized again."
        }
        if lowered.contains("4bit") || lowered.contains("q4") {
            return "The repo name says 4-bit. Quantization is set to None so it is not quantized again."
        }
        return nil
    }

    private static func sampleRows(from url: URL, limit: Int) -> [[String: Any]] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? handle.close() }
        let data = handle.readData(ofLength: 256 * 1024)
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        var rows: [[String: Any]] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, let object = jsonObject(from: Data(trimmed.utf8)) else { continue }
            rows.append(object)
            if rows.count == limit { break }
        }
        return rows
    }

    private static func countRows(at url: URL) -> Int {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        if size > 32_000_000 {
            return countNewlines(at: url, limit: 100_000)
        }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(whereSeparator: \.isNewline).reduce(into: 0) { count, line in
            if !line.trimmingCharacters(in: .whitespaces).isEmpty { count += 1 }
        }
    }

    private static func countNewlines(at url: URL, limit: Int) -> Int {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return 0 }
        defer { try? handle.close() }
        var count = 0
        while count < limit {
            let data = handle.readData(ofLength: 64 * 1024)
            if data.isEmpty { break }
            count += data.reduce(into: 0) { total, byte in
                if byte == 10 { total += 1 }
            }
        }
        return min(count, limit)
    }

    private static func rowCount(_ report: [String: Any]?, key: String) -> Int? {
        guard let counts = report?["counts"] as? [String: Any] else { return nil }
        return intValue(counts[key])
    }

    private static func rowCountValue(_ value: Any?) -> Int? {
        intValue(value)
    }

    private static func estimatedTokens(in row: [String: Any]) -> Int {
        var characters = 0
        func walk(_ value: Any, depth: Int) {
            guard depth < 6 else { return }
            if let text = value as? String {
                characters += text.count
            } else if let list = value as? [Any] {
                list.forEach { walk($0, depth: depth + 1) }
            } else if let object = value as? [String: Any] {
                object.values.forEach { walk($0, depth: depth + 1) }
            }
        }
        walk(row, depth: 0)
        return max(1, characters / 4)
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return jsonObject(from: data)
    }

    private static func jsonObject(from data: Data) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func intValue(_ value: Any?) -> Int? {
        if let number = value as? Int { return number }
        if let number = value as? Double { return Int(number) }
        if let number = value as? NSNumber { return number.intValue }
        return nil
    }

    private static func expand(_ path: String) -> String {
        NSString(string: path).expandingTildeInPath
    }

    private static func looksLikeHuggingFaceID(_ text: String) -> Bool {
        guard !text.hasPrefix("/"), !text.hasPrefix("~"), !text.hasPrefix("."), !text.contains(" ") else {
            return false
        }
        let parts = text.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && !parts[0].isEmpty && !parts[1].isEmpty
    }
}

private enum DetectedDatasetKind: Equatable {
    case chat(messages: String, system: String?)
    case promptCompletion(prompt: String, completion: String)
    case text(String)
    case preference(prompt: String?, chosen: String, rejected: String)
    case reward(prompt: String, answer: String)
    case promptOnly(String)
    case ftpo(context: String, rejected: String, chosen: String)

    var displayName: String {
        switch self {
        case .chat: "Chat messages"
        case .promptCompletion: "Prompt and completion"
        case .text: "Raw text"
        case .preference: "Preference pairs"
        case .reward: "Prompt and answer"
        case .promptOnly: "Prompts only"
        case .ftpo: "FTPO preference rows"
        }
    }

    var preferredMode: TrainMode {
        switch self {
        case .chat, .promptCompletion, .text: .sft
        case .preference(let prompt, _, _): prompt == nil ? .orpo : .dpo
        case .reward: .grpo
        case .promptOnly: .onlineDPO
        case .ftpo: .ftpo
        }
    }

    var compatibleModes: Set<TrainMode> {
        switch self {
        case .chat, .promptCompletion, .text:
            [.sft]
        case .preference(let prompt, _, _):
            prompt == nil ? [.orpo] : [.dpo, .cpo, .dsla]
        case .reward:
            [.grpo, .klpo]
        case .promptOnly:
            [.onlineDPO, .xpo, .rlhfReinforce, .ppo]
        case .ftpo:
            [.ftpo]
        }
    }

    var masksPrompt: Bool {
        switch self {
        case .text, .promptOnly: false
        default: true
        }
    }

    var columns: DatasetColumns {
        switch self {
        case .chat(let messages, let system):
            return DatasetColumns(chat: messages == "messages" ? "" : messages, system: system ?? "")
        case .promptCompletion(let prompt, let completion):
            return DatasetColumns(
                prompt: prompt == "prompt" ? "" : prompt,
                completion: completion == "completion" ? "" : completion
            )
        case .text(let key):
            return DatasetColumns(text: key == "text" ? "" : key)
        case .preference(let prompt, let chosen, let rejected):
            return DatasetColumns(
                prompt: prompt == "prompt" ? "" : (prompt ?? ""),
                chosen: chosen == "chosen" ? "" : chosen,
                rejected: rejected == "rejected" ? "" : rejected
            )
        case .reward(let prompt, let answer):
            return DatasetColumns(
                prompt: prompt == "prompt" ? "" : prompt,
                answer: answer == "answer" ? "" : answer
            )
        case .promptOnly(let prompt):
            return DatasetColumns(prompt: prompt == "prompt" ? "" : prompt)
        case .ftpo(let context, let rejected, let chosen):
            return DatasetColumns(ftpoContext: context, ftpoRejected: rejected, ftpoChosen: chosen)
        }
    }
}

private struct DatasetColumns {
    var prompt = ""
    var completion = ""
    var chosen = ""
    var rejected = ""
    var chat = ""
    var text = ""
    var system = ""
    var answer = ""
    var ftpoContext = "context_with_chat_template"
    var ftpoRejected = "rejected_decoded"
    var ftpoChosen = "multi_chosen_decoded"
}

extension TrainingConfig {
    mutating func applyDatasetPreset(_ preset: DatasetTrainingPreset) {
        trainMode = preset.trainMode
        sftLossType = preset.sftLossType
        if preset.prefersLoRA, trainType != .dora {
            trainType = .lora
        }
        maskPrompt = preset.maskPrompt
        maxSeqLength = preset.maxSeqLength
        batchSize = preset.batchSize
        gradientAccumulationSteps = preset.gradientAccumulationSteps
        epochs = preset.epochs
        iters = preset.iters
        learningRate = preset.learningRate
        learningRateSchedule = preset.learningRateSchedule
        lrWarmupSteps = preset.lrWarmupSteps
        lrDecayFraction = preset.lrDecayFraction
        lrFinal = preset.lrFinal
        stepsPerReport = preset.stepsPerReport
        stepsPerEval = preset.stepsPerEval
        saveEvery = preset.saveEvery
        valBatches = preset.valBatches
        test = preset.test
        rank = preset.rank
        scale = preset.scale
        dropout = preset.dropout
        numLayers = preset.numLayers
        gradCheckpoint = preset.gradCheckpoint
        efficientLongContext = preset.efficientLongContext
        qatEnable = preset.qatEnable
        promptFeature = preset.promptFeature
        completionFeature = preset.completionFeature
        chosenFeature = preset.chosenFeature
        rejectedFeature = preset.rejectedFeature
        chatFeature = preset.chatFeature
        textFeature = preset.textFeature
        systemFeature = preset.systemFeature
        answerFeature = preset.answerFeature
        ftpoContextFeature = preset.ftpoContextFeature
        ftpoRejectedFeature = preset.ftpoRejectedFeature
        ftpoChosenFeature = preset.ftpoChosenFeature
        if preset.clearDatasetSystemPrompt {
            datasetSystemPrompt = ""
        }
    }
}
