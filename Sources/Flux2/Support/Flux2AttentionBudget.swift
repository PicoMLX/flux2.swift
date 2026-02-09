import Foundation

/// Estimates attention memory requirements and checks against a budget.
///
/// Ported from flux2.c `attention_bytes()` / `fit_refs_for_attention()` (commit d62636d).
/// In the joint attention layers the query/key length equals the total sequence length
/// across all inputs (output latent tokens + reference image latent tokens + text tokens).
/// The attention matrix has size `numHeads * totalSeq^2 * sizeof(float32)`.
public enum Flux2AttentionBudget {

  /// Default maximum attention buffer size (4 GB), matching the flux2.c default.
  public static let defaultMaxBytes: Int = 4 * 1024 * 1024 * 1024

  /// The spatial downsampling factor from pixel space to latent token space.
  /// For the Flux2 VAE: vaeScaleFactor (8) * patchSize (2) = 16.
  static let spatialDownsampleFactor = 16

  /// Computes the estimated attention buffer size in bytes.
  ///
  /// - Parameters:
  ///   - numHeads: Number of attention heads in the transformer.
  ///   - outputHeight: Output image height in pixels.
  ///   - outputWidth: Output image width in pixels.
  ///   - referenceImageDims: Array of `(height, width)` in pixels for each reference image.
  ///   - textSeqLen: Text sequence length (number of text tokens).
  /// - Returns: Estimated bytes for the attention matrix.
  public static func attentionBytes(
    numHeads: Int,
    outputHeight: Int,
    outputWidth: Int,
    referenceImageDims: [(height: Int, width: Int)],
    textSeqLen: Int
  ) -> Int {
    let outputTokens = (outputHeight / spatialDownsampleFactor)
      * (outputWidth / spatialDownsampleFactor)

    let refTokens = referenceImageDims.reduce(0) { total, dim in
      total + (dim.height / spatialDownsampleFactor) * (dim.width / spatialDownsampleFactor)
    }

    let totalSeq = outputTokens + refTokens + textSeqLen

    // numHeads * totalSeq^2 * sizeof(float32)
    return numHeads * totalSeq * totalSeq * 4
  }

  /// Checks whether the estimated attention buffer exceeds the budget.
  ///
  /// - Parameters:
  ///   - numHeads: Number of attention heads in the transformer.
  ///   - outputHeight: Output image height in pixels.
  ///   - outputWidth: Output image width in pixels.
  ///   - referenceImageDims: Array of `(height, width)` in pixels for each reference image.
  ///   - textSeqLen: Text sequence length (number of text tokens).
  ///   - maxBytes: Maximum allowed bytes (defaults to 4 GB).
  /// - Returns: `true` if the estimated size exceeds the budget.
  public static func wouldExceedBudget(
    numHeads: Int,
    outputHeight: Int,
    outputWidth: Int,
    referenceImageDims: [(height: Int, width: Int)],
    textSeqLen: Int,
    maxBytes: Int = defaultMaxBytes
  ) -> Bool {
    attentionBytes(
      numHeads: numHeads,
      outputHeight: outputHeight,
      outputWidth: outputWidth,
      referenceImageDims: referenceImageDims,
      textSeqLen: textSeqLen
    ) > maxBytes
  }
}

/// Error thrown when the attention memory budget would be exceeded.
public enum Flux2AttentionBudgetError: Error, LocalizedError {
  case attentionExceedsBudget(estimatedGB: Double, maxGB: Double, outputHeight: Int, outputWidth: Int, referenceCount: Int)

  public var errorDescription: String? {
    switch self {
    case .attentionExceedsBudget(let estimatedGB, let maxGB, let outputHeight, let outputWidth, let referenceCount):
      return "Attention memory budget exceeded: estimated \(String(format: "%.1f", estimatedGB)) GB "
        + "exceeds \(String(format: "%.1f", maxGB)) GB limit. "
        + "Output size: \(outputWidth)x\(outputHeight) with \(referenceCount) reference image(s). "
        + "Try reducing the output dimensions or using smaller reference images."
    }
  }
}
