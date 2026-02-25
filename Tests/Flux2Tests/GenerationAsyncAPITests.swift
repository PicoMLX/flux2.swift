import CoreGraphics
import Foundation
import MLX
import XCTest
@testable import Flux2

// MARK: - SendableBox tests

final class SendableBoxTests: XCTestCase {

  func testTakeReturnsValueOnce() {
    let box = SendableBox(42)
    XCTAssertEqual(box.take(), 42)
    XCTAssertNil(box.take(), "Second take should return nil")
  }

  func testTakeReturnsNilAfterConsumed() {
    let box = SendableBox("hello")
    _ = box.take()
    XCTAssertNil(box.take())
    XCTAssertNil(box.take())
  }
}

// MARK: - GenerationProgress tests

final class GenerationProgressTests: XCTestCase {

  func testFractionCompleted() {
    let progress = GenerationProgress(step: 5, totalSteps: 10)
    XCTAssertEqual(progress.step, 5)
    XCTAssertEqual(progress.totalSteps, 10)
    XCTAssertEqual(progress.fractionCompleted, 0.5, accuracy: 1e-10)
  }

  func testFractionCompletedZeroSteps() {
    let progress = GenerationProgress(step: 0, totalSteps: 0)
    XCTAssertEqual(progress.fractionCompleted, 0.0)
  }

  func testFractionCompletedComplete() {
    let progress = GenerationProgress(step: 20, totalSteps: 20)
    XCTAssertEqual(progress.fractionCompleted, 1.0, accuracy: 1e-10)
  }

  func testMonotonicProgress() {
    let totalSteps = 10
    var previousFraction = -1.0
    for step in 1...totalSteps {
      let progress = GenerationProgress(step: step, totalSteps: totalSteps)
      XCTAssertGreaterThan(
        progress.fractionCompleted, previousFraction,
        "Progress should be monotonically increasing"
      )
      previousFraction = progress.fractionCompleted
    }
  }
}

// MARK: - GenerationHandle tests

final class GenerationHandleTests: XCTestCase {

  func testCancelCancelsUnderlyingTask() async {
    let (stream, continuation) = AsyncThrowingStream<GenerationProgress, Error>.makeStream(
      of: GenerationProgress.self
    )
    let task = Task<Int, Error> {
      while !Task.isCancelled {
        try await Task.sleep(nanoseconds: 5_000_000)
      }
      throw CancellationError()
    }
    let handle = GenerationHandle(progress: stream, task: task)

    handle.cancel()
    continuation.finish(throwing: CancellationError())

    do {
      _ = try await handle.value()
      XCTFail("Expected CancellationError")
    } catch is CancellationError {
      // Expected
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  func testProgressCanDrainBeforeAwaitingValue() async throws {
    let (stream, continuation) = AsyncThrowingStream<GenerationProgress, Error>.makeStream(
      of: GenerationProgress.self
    )
    let task = Task<Int, Error> {
      continuation.yield(GenerationProgress(step: 1, totalSteps: 2))
      continuation.yield(GenerationProgress(step: 2, totalSteps: 2))
      continuation.finish()
      return 42
    }

    let handle = GenerationHandle(progress: stream, task: task)
    var steps: [Int] = []
    for try await progress in handle.progress {
      steps.append(progress.step)
    }

    XCTAssertEqual(steps, [1, 2])
    let value = try await handle.value()
    XCTAssertEqual(value, 42)
  }
}

// MARK: - API surface tests

final class GenerationAsyncAPISurfaceTests: XCTestCase {

  func testKleinGenerateTaskWiredMemoryOverloadsCompile() {
    let limitOverload = { (pipeline: Flux2KleinPipeline) throws in
      try pipeline.generateTask(
        prompts: ["cat"],
        height: 64,
        width: 64,
        numInferenceSteps: 1,
        wiredMemoryLimit: nil
      )
    }
    let ticketOverload = { (pipeline: Flux2KleinPipeline, ticket: WiredMemoryTicket?) throws in
      try pipeline.generateTask(
        prompts: ["cat"],
        height: 64,
        width: 64,
        numInferenceSteps: 1,
        wiredMemoryTicket: ticket
      )
    }

    XCTAssertNotNil(limitOverload as Any)
    XCTAssertNotNil(ticketOverload as Any)
  }

  func testDevGenerateTaskWiredMemoryOverloadsCompile() {
    let limitOverload = { (pipeline: Flux2DevPipeline) throws in
      try pipeline.generateTask(
        prompts: ["cat"],
        height: 64,
        width: 64,
        numInferenceSteps: 1,
        wiredMemoryLimit: nil
      )
    }
    let ticketOverload = { (pipeline: Flux2DevPipeline, ticket: WiredMemoryTicket?) throws in
      try pipeline.generateTask(
        prompts: ["cat"],
        height: 64,
        width: 64,
        numInferenceSteps: 1,
        wiredMemoryTicket: ticket
      )
    }

    XCTAssertNotNil(limitOverload as Any)
    XCTAssertNotNil(ticketOverload as Any)
  }
}

// MARK: - ImageConversion tests

final class ImageConversionTests: XCTestCase {

  func testInvalidDimensionsThrows() {
    // 3D tensor instead of 4D: shape [3, 4, 4] = 48 float32 elements = 192 bytes
    let badTensor = MLXArray(Data(repeating: 0, count: 3 * 4 * 4 * 4), [3, 4, 4], dtype: .float32)
    XCTAssertThrowsError(try ImageConversion.cgImage(from: badTensor)) { error in
      guard case ImageConversion.ImageConversionError.invalidDimensions(3) = error else {
        XCTFail("Expected invalidDimensions(3), got \(error)")
        return
      }
    }
  }

  func testInvalidChannelsThrows() {
    // 4 channels instead of 3: shape [1, 4, 2, 2] = 16 float32 elements = 64 bytes
    let badTensor = MLXArray(Data(repeating: 0, count: 1 * 4 * 2 * 2 * 4), [1, 4, 2, 2], dtype: .float32)
    XCTAssertThrowsError(try ImageConversion.cgImage(from: badTensor)) { error in
      guard case ImageConversion.ImageConversionError.invalidChannels(4) = error else {
        XCTFail("Expected invalidChannels(4), got \(error)")
        return
      }
    }
  }
}
