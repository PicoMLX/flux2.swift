import Foundation
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

// Need to import MLX for ImageConversion tests
import MLX
