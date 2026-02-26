import CoreGraphics
import Foundation
@preconcurrency import MLX
import Hub

public struct Flux2KleinPipelineOutput {
  public let packedLatents: MLXArray
  public let decoded: MLXArray
  public let promptEmbeds: MLXArray
  public let textIds: MLXArray
  public let latentIds: MLXArray
  public let imageLatents: MLXArray?
  public let imageLatentIds: MLXArray?
}

public enum Flux2KleinPipelineError: Error {
  case promptEncoderReleased
  case missingTokenizer
  case missingNegativeTokens
  case invalidNegativeTokens
  case negativeInputIdsShapeMismatch(expected: [Int], got: [Int])
  case negativeAttentionMaskShapeMismatch(expected: [Int], got: [Int])
  case invalidLatentChannels(Int, Int)
  case invalidNumInferenceSteps(Int)
  case invalidImageCount(Int)
  case promptEncoderReloadUnavailable
}

public final class Flux2KleinPipeline {
  public let transformer: Flux2Transformer2DModel
  public let scheduler: FlowMatchEulerDiscreteScheduler
  public let vae: Flux2AutoencoderKL
  public private(set) var promptEncoder: Flux2KleinPromptEncoder?
  public let isDistilled: Bool

  private let pipeline: Flux2Pipeline

  // Retained for encoder-only reload (reloadPromptEncoder).
  private var _snapshot: URL?
  private var _dtype: DType?
  private var _hiddenStateLayers: [Int]?
  private var _maxLengthOverride: Int?

  public init(
    transformer: Flux2Transformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: Flux2AutoencoderKL,
    promptEncoder: Flux2KleinPromptEncoder,
    isDistilled: Bool = false
  ) {
    self.transformer = transformer
    self.scheduler = scheduler
    self.vae = vae
    self.promptEncoder = promptEncoder
    self.isDistilled = isDistilled
    self.pipeline = Flux2Pipeline(transformer: transformer, scheduler: scheduler, vae: vae)
  }

  public convenience init(
    snapshot: URL,
    dtype: DType = .bfloat16,
    hiddenStateLayers: [Int] = [9, 18, 27],
    maxLengthOverride: Int? = nil,
    hubApi: HubApi = .shared,
    loadTokenizer: Bool = true
  ) throws {
    let transformer = try Flux2Transformer2DModel.load(from: snapshot, dtype: dtype)
    let scheduler = try FlowMatchEulerDiscreteScheduler.load(from: snapshot)
    let vae = try Flux2AutoencoderKL.load(from: snapshot, dtype: dtype)
    let isDistilled = Self.loadIsDistilledFlag(snapshot: snapshot)
    let promptEncoder: Flux2KleinPromptEncoder
    if loadTokenizer {
      promptEncoder = try Flux2KleinPromptEncoder(
        snapshot: snapshot,
        dtype: dtype,
        hiddenStateLayers: hiddenStateLayers,
        maxLengthOverride: maxLengthOverride,
        hubApi: hubApi
      )
    } else {
      let textEncoder = try Flux2Qwen3TextEncoder.load(from: snapshot, dtype: dtype)
      promptEncoder = Flux2KleinPromptEncoder(
        textEncoder: textEncoder,
        tokenizer: nil,
        hiddenStateLayers: hiddenStateLayers
      )
    }
    self.init(
      transformer: transformer,
      scheduler: scheduler,
      vae: vae,
      promptEncoder: promptEncoder,
      isDistilled: isDistilled
    )
    self._snapshot = snapshot
    self._dtype = dtype
    self._hiddenStateLayers = hiddenStateLayers
    self._maxLengthOverride = maxLengthOverride
  }

  // MARK: - Synchronous API

  /// Generate an image synchronously.
  ///
  /// - Important: `wiredMemoryLimit` is currently not applied in the synchronous
  ///   API path because MLX's synchronous `Memory.withWiredLimit` wrapper is a
  ///   deprecated no-op. Use `generateTask(..., wiredMemoryLimit:)` or
  ///   `generateTask(..., wiredMemoryTicket:)` for effective wired-memory
  ///   coordination.
  public func generate(
    prompts: [String],
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    numImagesPerPrompt: Int = 1,
    latents: MLXArray? = nil,
    guidanceScale: Float = 1.0,
    modelTimestepScale: Float = 0.001,
    images: [MLXArray]? = nil,
    imageIdScale: Int = 10,
    progressHandler: GenerationProgressHandler? = nil,
    wiredMemoryLimit: Int? = nil,
    denoiseStepCallback: DenoiseStepCallback? = nil,
    evalInterval: Int? = nil
  ) throws -> Flux2KleinPipelineOutput {
    guard let promptEncoder = promptEncoder else {
      throw Flux2KleinPipelineError.promptEncoderReleased
    }
    guard promptEncoder.tokenizer != nil else {
      throw Flux2KleinPipelineError.missingTokenizer
    }

    let encoding = try promptEncoder.encodePrompts(
      prompts,
      numImagesPerPrompt: numImagesPerPrompt
    )

    let classifierFreeGuidanceEnabled = guidanceScale > 1.0 && !isDistilled

    let negativeEncoding: Flux2PromptEncoding?
    if classifierFreeGuidanceEnabled {
      let negatives = Array(repeating: "", count: prompts.count)
      negativeEncoding = try promptEncoder.encodePrompts(
        negatives,
        numImagesPerPrompt: numImagesPerPrompt
      )
    } else {
      negativeEncoding = nil
    }

    evalAndReleasePromptEncoder(promptEncoding: encoding, negativeEncoding: negativeEncoding)

    return try generateFromEncoding(
      promptEncoding: encoding,
      negativeEncoding: negativeEncoding,
      height: height,
      width: width,
      numInferenceSteps: numInferenceSteps,
      latents: latents,
      guidanceScale: classifierFreeGuidanceEnabled ? guidanceScale : 1.0,
      modelTimestepScale: modelTimestepScale,
      images: images,
      imageIdScale: imageIdScale,
      progressHandler: progressHandler,
      denoiseStepCallback: denoiseStepCallback,
      evalInterval: evalInterval
    )
  }

  /// Generate an image synchronously from pre-tokenized inputs.
  ///
  /// - Important: `wiredMemoryLimit` is currently not applied in the synchronous
  ///   API path because MLX's synchronous `Memory.withWiredLimit` wrapper is a
  ///   deprecated no-op. Use `generateTask(..., wiredMemoryLimit:)` or
  ///   `generateTask(..., wiredMemoryTicket:)` for effective wired-memory
  ///   coordination.
  public func generateTokens(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    negativeInputIds: MLXArray? = nil,
    negativeAttentionMask: MLXArray? = nil,
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    numImagesPerPrompt: Int = 1,
    latents: MLXArray? = nil,
    guidanceScale: Float = 1.0,
    modelTimestepScale: Float = 0.001,
    images: [MLXArray]? = nil,
    imageIdScale: Int = 10,
    progressHandler: GenerationProgressHandler? = nil,
    wiredMemoryLimit: Int? = nil,
    denoiseStepCallback: DenoiseStepCallback? = nil,
    evalInterval: Int? = nil
  ) throws -> Flux2KleinPipelineOutput {
    guard let promptEncoder = promptEncoder else {
      throw Flux2KleinPipelineError.promptEncoderReleased
    }

    let encoding = try promptEncoder.encodeTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      numImagesPerPrompt: numImagesPerPrompt
    )

    let classifierFreeGuidanceEnabled = guidanceScale > 1.0 && !isDistilled

    let negativeEncoding: Flux2PromptEncoding?
    if classifierFreeGuidanceEnabled {
      if negativeInputIds != nil || negativeAttentionMask != nil {
        guard let negativeInputIds, let negativeAttentionMask else {
          throw Flux2KleinPipelineError.invalidNegativeTokens
        }
        guard negativeInputIds.shape == inputIds.shape else {
          throw Flux2KleinPipelineError.negativeInputIdsShapeMismatch(
            expected: inputIds.shape,
            got: negativeInputIds.shape
          )
        }
        guard negativeAttentionMask.shape == attentionMask.shape else {
          throw Flux2KleinPipelineError.negativeAttentionMaskShapeMismatch(
            expected: attentionMask.shape,
            got: negativeAttentionMask.shape
          )
        }

        negativeEncoding = try promptEncoder.encodeTokens(
          inputIds: negativeInputIds,
          attentionMask: negativeAttentionMask,
          numImagesPerPrompt: numImagesPerPrompt
        )
      } else if let tokenizer = promptEncoder.tokenizer {
        let batchSize = inputIds.dim(0)
        let maxLength = inputIds.dim(1)
        let negatives = Array(repeating: "", count: batchSize)
        let tokens = try tokenizer.encode(
          prompts: negatives,
          maxLength: maxLength,
          addGenerationPrompt: true,
          enableThinking: false
        )
        negativeEncoding = try promptEncoder.encodeTokens(
          inputIds: tokens.inputIds,
          attentionMask: tokens.attentionMask,
          numImagesPerPrompt: numImagesPerPrompt
        )
      } else {
        throw Flux2KleinPipelineError.missingNegativeTokens
      }
    } else {
      negativeEncoding = nil
    }

    evalAndReleasePromptEncoder(promptEncoding: encoding, negativeEncoding: negativeEncoding)

    return try generateFromEncoding(
      promptEncoding: encoding,
      negativeEncoding: negativeEncoding,
      height: height,
      width: width,
      numInferenceSteps: numInferenceSteps,
      latents: latents,
      guidanceScale: classifierFreeGuidanceEnabled ? guidanceScale : 1.0,
      modelTimestepScale: modelTimestepScale,
      images: images,
      imageIdScale: imageIdScale,
      progressHandler: progressHandler,
      denoiseStepCallback: denoiseStepCallback,
      evalInterval: evalInterval
    )
  }

  // MARK: - Async streaming API

  /// Launch an asynchronous generation task that streams progress and produces a CGImage.
  ///
  /// Returns a `GenerationHandle<CGImage>` whose `.progress` stream yields
  /// metadata-only `GenerationProgress` events, and whose `.value()` awaits
  /// the final CGImage result.
  public func generateTask(
    prompts: [String],
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    numImagesPerPrompt: Int = 1,
    latents: MLXArray? = nil,
    guidanceScale: Float = 1.0,
    modelTimestepScale: Float = 0.001,
    images: [MLXArray]? = nil,
    imageIdScale: Int = 10,
    wiredMemoryLimit: Int? = nil,
    evalInterval: Int? = nil
  ) throws -> GenerationHandle<CGImage> {
    try generateTask(
      prompts: prompts,
      height: height,
      width: width,
      numInferenceSteps: numInferenceSteps,
      numImagesPerPrompt: numImagesPerPrompt,
      latents: latents,
      guidanceScale: guidanceScale,
      modelTimestepScale: modelTimestepScale,
      images: images,
      imageIdScale: imageIdScale,
      wiredMemoryTicket: wiredMemoryLimit.map(Flux2WiredMemory.requestTicket(limit:)),
      evalInterval: evalInterval
    )
  }

  /// Launch an asynchronous generation task using a pre-built wired-memory ticket.
  ///
  /// Progress events are metadata-only and may be emitted before all GPU work for
  /// a step is materialized because MLX evaluation is lazy.
  public func generateTask(
    prompts: [String],
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    numImagesPerPrompt: Int = 1,
    latents: MLXArray? = nil,
    guidanceScale: Float = 1.0,
    modelTimestepScale: Float = 0.001,
    images: [MLXArray]? = nil,
    imageIdScale: Int = 10,
    wiredMemoryTicket: WiredMemoryTicket?,
    evalInterval: Int? = nil
  ) throws -> GenerationHandle<CGImage> {
    let (progressStream, progressContinuation) = AsyncThrowingStream.makeStream(
      of: GenerationProgress.self
    )

    // Box non-Sendable values (MLXArray) for transfer into the detached task
    let paramsBox = SendableBox((pipeline: self, latents: latents, images: images))

    let task = Task.detached { [progressContinuation] () -> CGImage in
      guard let params = paramsBox.take() else {
        progressContinuation.finish(throwing: CancellationError())
        throw CancellationError()
      }

      do {
        let body = { () throws -> CGImage in
          let output = try params.pipeline.generateTaskBody(
            prompts: prompts,
            height: height,
            width: width,
            numInferenceSteps: numInferenceSteps,
            numImagesPerPrompt: numImagesPerPrompt,
            latents: params.latents,
            guidanceScale: guidanceScale,
            modelTimestepScale: modelTimestepScale,
            images: params.images,
            imageIdScale: imageIdScale,
            evalInterval: evalInterval,
            progressContinuation: progressContinuation
          )

          return try ImageConversion.cgImage(from: output.decoded)
        }

        let cgImage: CGImage
        if let wiredMemoryTicket {
          cgImage = try await wiredMemoryTicket.withWiredLimit {
            try body()
          }
        } else {
          cgImage = try body()
        }
        progressContinuation.finish()
        return cgImage
      } catch {
        progressContinuation.finish(throwing: error)
        throw error
      }
    }

    progressContinuation.onTermination = { @Sendable _ in
      task.cancel()
    }

    return GenerationHandle(progress: progressStream, task: task)
  }

  // MARK: - Private

  private func generateTaskBody(
    prompts: [String],
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    numImagesPerPrompt: Int,
    latents: MLXArray?,
    guidanceScale: Float,
    modelTimestepScale: Float,
    images: [MLXArray]?,
    imageIdScale: Int,
    evalInterval: Int? = nil,
    progressContinuation: AsyncThrowingStream<GenerationProgress, Error>.Continuation
  ) throws -> Flux2KleinPipelineOutput {
    guard let promptEncoder = promptEncoder else {
      throw Flux2KleinPipelineError.promptEncoderReleased
    }
    guard promptEncoder.tokenizer != nil else {
      throw Flux2KleinPipelineError.missingTokenizer
    }

    let encoding = try promptEncoder.encodePrompts(
      prompts,
      numImagesPerPrompt: numImagesPerPrompt
    )

    let classifierFreeGuidanceEnabled = guidanceScale > 1.0 && !isDistilled

    let negativeEncoding: Flux2PromptEncoding?
    if classifierFreeGuidanceEnabled {
      let negatives = Array(repeating: "", count: prompts.count)
      negativeEncoding = try promptEncoder.encodePrompts(
        negatives,
        numImagesPerPrompt: numImagesPerPrompt
      )
    } else {
      negativeEncoding = nil
    }

    evalAndReleasePromptEncoder(promptEncoding: encoding, negativeEncoding: negativeEncoding)

    return try generateFromEncoding(
      promptEncoding: encoding,
      negativeEncoding: negativeEncoding,
      height: height,
      width: width,
      numInferenceSteps: numInferenceSteps,
      latents: latents,
      guidanceScale: classifierFreeGuidanceEnabled ? guidanceScale : 1.0,
      modelTimestepScale: modelTimestepScale,
      images: images,
      imageIdScale: imageIdScale,
      progressHandler: { progress in
        let result = progressContinuation.yield(progress)
        if case .terminated = result {
          // Consumer stopped listening
        }
      },
      evalInterval: evalInterval
    )
  }

  private func evalAndReleasePromptEncoder(
    promptEncoding: Flux2PromptEncoding,
    negativeEncoding: Flux2PromptEncoding?
  ) {
    var toEval: [MLXArray] = [
      promptEncoding.promptEmbeds,
      promptEncoding.textIds,
    ]
    if let negativeEncoding {
      toEval.append(negativeEncoding.promptEmbeds)
      toEval.append(negativeEncoding.textIds)
    }
    MLX.eval(toEval)
    promptEncoder = nil
  }

  /// Reload only the prompt encoder from disk, avoiding a full pipeline rebuild.
  ///
  /// Call this when `promptEncoder` is `nil` (released after the previous generation)
  /// but the rest of the pipeline (transformer, VAE, scheduler) is still valid.
  /// Requires that the pipeline was created via the convenience `init(snapshot:...)`.
  public func reloadPromptEncoder() throws {
    guard let snapshot = _snapshot, let dtype = _dtype,
          let hiddenStateLayers = _hiddenStateLayers else {
      throw Flux2KleinPipelineError.promptEncoderReloadUnavailable
    }
    self.promptEncoder = try Flux2KleinPromptEncoder(
      snapshot: snapshot,
      dtype: dtype,
      hiddenStateLayers: hiddenStateLayers,
      maxLengthOverride: _maxLengthOverride
    )
  }

  /// Force-materialize all model weights so that the first denoise step isn't
  /// penalized by lazy MLX evaluation.  Call after loading or reloading the pipeline.
  public func evaluateWeights() {
    var arrays: [MLXArray] = []
    arrays.append(contentsOf: transformer.parameters().flattened().map(\.1))
    arrays.append(contentsOf: vae.parameters().flattened().map(\.1))
    if let promptEncoder {
      arrays.append(contentsOf: promptEncoder.textEncoder.parameters().flattened().map(\.1))
    }
    if !arrays.isEmpty { MLX.eval(arrays) }
  }

  /// Force-materialize only the prompt encoder weights after an encoder-only reload.
  public func evaluatePromptEncoderWeights() {
    guard let promptEncoder else { return }
    let arrays = promptEncoder.textEncoder.parameters().flattened().map(\.1)
    if !arrays.isEmpty { MLX.eval(arrays) }
  }

  private func generateFromEncoding(
    promptEncoding: Flux2PromptEncoding,
    negativeEncoding: Flux2PromptEncoding?,
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    latents: MLXArray?,
    guidanceScale: Float,
    modelTimestepScale: Float,
    images: [MLXArray]?,
    imageIdScale: Int,
    progressHandler: GenerationProgressHandler? = nil,
    denoiseStepCallback: DenoiseStepCallback? = nil,
    evalInterval: Int? = nil
  ) throws -> Flux2KleinPipelineOutput {
    guard numInferenceSteps > 0 else {
      throw Flux2KleinPipelineError.invalidNumInferenceSteps(numInferenceSteps)
    }

    let patchArea = vae.configuration.patchSizeArea
    let inChannels = transformer.configuration.inChannels
    guard inChannels % patchArea == 0 else {
      throw Flux2KleinPipelineError.invalidLatentChannels(inChannels, patchArea)
    }
    let numLatentChannels = inChannels / patchArea

    let batchSize = promptEncoding.promptEmbeds.dim(0)
    let prepared = try Flux2LatentPreparation.prepareLatents(
      batchSize: batchSize,
      numLatentChannels: numLatentChannels,
      height: height,
      width: width,
      vae: vae,
      dtype: promptEncoding.promptEmbeds.dtype,
      latents: latents
    )

    let preparedImages: Flux2PreparedImageLatents?
    if let images {
      if images.isEmpty {
        throw Flux2KleinPipelineError.invalidImageCount(images.count)
      }

      // Check attention memory budget before encoding reference images.
      let refDims: [(height: Int, width: Int)] = images.map { img in
        (height: img.dim(1), width: img.dim(2))
      }
      let numHeads = transformer.configuration.numAttentionHeads
      let textSeqLen = promptEncoding.textIds.dim(1)
      let estimatedBytes = Flux2AttentionBudget.attentionBytes(
        numHeads: numHeads,
        outputHeight: height,
        outputWidth: width,
        referenceImageDims: refDims,
        textSeqLen: textSeqLen
      )
      let maxBytes = Flux2AttentionBudget.defaultMaxBytes
      if estimatedBytes > maxBytes {
        throw Flux2AttentionBudgetError.attentionExceedsBudget(
          estimatedGB: Double(estimatedBytes) / (1024 * 1024 * 1024),
          maxGB: Double(maxBytes) / (1024 * 1024 * 1024),
          outputHeight: height,
          outputWidth: width,
          referenceCount: images.count
        )
      }

      preparedImages = try Flux2LatentPreparation.prepareImageLatents(
        images: images,
        batchSize: batchSize,
        vae: vae,
        dtype: promptEncoding.promptEmbeds.dtype,
        imageIdScale: imageIdScale
      )
    } else {
      preparedImages = nil
    }

    let sigmas = scheduler.flux2DefaultSigmas(numInferenceSteps: numInferenceSteps)
    let mu = scheduler.config.useDynamicShifting
      ? FlowMatchEulerDiscreteScheduler.flux2EmpiricalMu(
        imageSeqLen: prepared.latents.dim(1),
        numSteps: numInferenceSteps
      )
      : nil

    try scheduler.setTimesteps(numInferenceSteps: numInferenceSteps, sigmas: sigmas, mu: mu)
    scheduler.setBeginIndex(0)

    let resolvedEvalInterval = evalInterval ?? (numInferenceSteps < 5 ? 1 : 5)

    let denoised = try self.denoise(
      latents: prepared.latents,
      latentIds: prepared.ids,
      promptEncoding: promptEncoding,
      negativeEncoding: negativeEncoding,
      guidanceScale: guidanceScale,
      modelTimestepScale: modelTimestepScale,
      imageConditioning: preparedImages.map { (latents: $0.latents, ids: $0.ids) },
      evalInterval: resolvedEvalInterval,
      progressHandler: progressHandler,
      denoiseStepCallback: denoiseStepCallback
    )

    let decoded = try self.pipeline.decodeLatents(denoised, latentIds: prepared.ids)

    var toEval: [MLXArray] = [
      denoised,
      decoded,
      promptEncoding.promptEmbeds,
      promptEncoding.textIds,
      prepared.ids,
    ]
    if let preparedImages {
      toEval.append(preparedImages.latents)
      toEval.append(preparedImages.ids)
    }
    MLX.eval(toEval)

    return Flux2KleinPipelineOutput(
      packedLatents: denoised,
      decoded: decoded,
      promptEmbeds: promptEncoding.promptEmbeds,
      textIds: promptEncoding.textIds,
      latentIds: prepared.ids,
      imageLatents: preparedImages?.latents,
      imageLatentIds: preparedImages?.ids
    )
  }

  private func denoise(
    latents: MLXArray,
    latentIds: MLXArray,
    promptEncoding: Flux2PromptEncoding,
    negativeEncoding: Flux2PromptEncoding?,
    guidanceScale: Float,
    modelTimestepScale: Float,
    imageConditioning: (latents: MLXArray, ids: MLXArray)?,
    evalInterval: Int = 5,
    progressHandler: GenerationProgressHandler? = nil,
    denoiseStepCallback: DenoiseStepCallback? = nil
  ) throws -> MLXArray {
    let stepValues = scheduler.timestepsValues
    let batch = latents.dim(0)
    var current = latents
    let combinedIds: MLXArray
    let conditioningImageLatents: MLXArray?

    if let imageConditioning {
      conditioningImageLatents = imageConditioning.latents
      combinedIds = MLX.concatenated([latentIds, imageConditioning.ids], axis: 1)
    } else {
      conditioningImageLatents = nil
      combinedIds = latentIds
    }

    let totalSteps = stepValues.count
    for (stepIndex, step) in stepValues.enumerated() {
      // Check for cooperative cancellation
      try Task.checkCancellation()

      let timestep = MLX.full([batch], values: step).asType(current.dtype)
      var noisePred = predictNoise(
        latents: current,
        encoderHiddenStates: promptEncoding.promptEmbeds,
        timestep: timestep,
        imgIds: combinedIds,
        txtIds: promptEncoding.textIds,
        modelTimestepScale: modelTimestepScale,
        imageLatents: conditioningImageLatents
      )

      if guidanceScale > 1.0, let negativeEncoding {
        let negNoise = predictNoise(
          latents: current,
          encoderHiddenStates: negativeEncoding.promptEmbeds,
          timestep: timestep,
          imgIds: combinedIds,
          txtIds: negativeEncoding.textIds,
          modelTimestepScale: modelTimestepScale,
          imageLatents: conditioningImageLatents
        )
        noisePred = negNoise + MLXArray(guidanceScale) * (noisePred - negNoise)
      }

      let prev = try scheduler.step(
        modelOutput: noisePred,
        timestep: timestep,
        sample: current
      ).prevSample
      current = prev

      if evalInterval > 0, (stepIndex + 1) % evalInterval == 0 {
        MLX.eval(current)
      }

      progressHandler?(GenerationProgress(step: stepIndex + 1, totalSteps: totalSteps))
      try denoiseStepCallback?(stepIndex + 1, totalSteps, current, latentIds)
    }

    return current
  }

  /// Decode packed latents to pixel data via the VAE.
  ///
  /// This is a convenience that delegates to the internal `Flux2Pipeline.decodeLatents`.
  /// Use it from a ``DenoiseStepCallback`` to generate intermediate preview images.
  public func decodeLatents(
    _ packedLatents: MLXArray,
    latentIds: MLXArray
  ) throws -> MLXArray {
    try pipeline.decodeLatents(packedLatents, latentIds: latentIds)
  }

  private func predictNoise(
    latents: MLXArray,
    encoderHiddenStates: MLXArray,
    timestep: MLXArray,
    imgIds: MLXArray,
    txtIds: MLXArray,
    modelTimestepScale: Float,
    imageLatents: MLXArray?
  ) -> MLXArray {
    var transformerInput = latents
    if let imageLatents {
      transformerInput = MLX.concatenated([latents, imageLatents], axis: 1)
    }

    var transformerTimestep = timestep
    let scale = MLXArray(modelTimestepScale).asType(transformerTimestep.dtype)
    transformerTimestep = transformerTimestep * scale

    let noisePredAll = transformer(
      transformerInput,
      encoderHiddenStates: encoderHiddenStates,
      timestep: transformerTimestep,
      imgIds: imgIds,
      txtIds: txtIds,
      guidance: nil,
      attentionMask: .none
    )

    let tokenCount = latents.dim(1)
    return noisePredAll[0..., 0..<tokenCount, 0...]
  }

  // MARK: - Model detection

  private struct ModelIndex: Decodable {
    let isDistilled: Bool?

    enum CodingKeys: String, CodingKey {
      case isDistilled = "is_distilled"
    }
  }

  private static func loadIsDistilledFlag(snapshot: URL) -> Bool {
    let indexURL = snapshot.appendingPathComponent("model_index.json")
    guard let data = try? Data(contentsOf: indexURL) else {
      return false
    }

    do {
      let modelIndex = try JSONDecoder().decode(ModelIndex.self, from: data)
      return modelIndex.isDistilled ?? false
    } catch {
      return false
    }
  }
}
