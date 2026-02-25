import CoreGraphics
import Foundation

/// A handle to a running asynchronous generation task.
///
/// Provides:
/// - `progress`: An `AsyncThrowingStream` of `GenerationProgress` metadata events.
/// - `value()`: Awaits and returns the final `Output` (e.g., `CGImage`).
/// - `cancel()`: Cancels the underlying generation task.
///
/// The `Output` type must be `Sendable` to safely cross concurrency boundaries.
/// For raw pipeline output (which contains non-Sendable `MLXArray`), use the
/// synchronous `generate()` API.
public final class GenerationHandle<Output: Sendable>: Sendable {
  /// Stream of metadata-only progress updates.
  public let progress: AsyncThrowingStream<GenerationProgress, Error>

  private let task: Task<Output, Error>

  init(
    progress: AsyncThrowingStream<GenerationProgress, Error>,
    task: Task<Output, Error>
  ) {
    self.progress = progress
    self.task = task
  }

  /// Await the final generated output.
  public func value() async throws -> Output {
    try await task.value
  }

  /// Cancel the generation.
  public func cancel() {
    task.cancel()
  }
}
