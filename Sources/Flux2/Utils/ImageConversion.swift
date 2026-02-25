import CoreGraphics
import Foundation
@preconcurrency import MLX

/// Converts decoded MLXArray image tensors to CGImage.
///
/// Expects NCHW layout (batch, channels=3, height, width) with values in [-1, 1].
/// Adapted from CLI+Image.swift for use in the library target.
public enum ImageConversion {

  public enum ImageConversionError: Error {
    case invalidDimensions(Int)
    case emptyBatch
    case invalidChannels(Int)
    case cgImageCreationFailed
  }

  /// Convert the first image in a decoded NCHW tensor to a CGImage.
  ///
  /// - Parameter decoded: An MLXArray with shape `[N, 3, H, W]` and values in `[-1, 1]`.
  /// - Returns: A CGImage with 8-bit RGBA pixels.
  public static func cgImage(from decoded: MLXArray) throws -> CGImage {
    guard decoded.ndim == 4 else {
      throw ImageConversionError.invalidDimensions(decoded.ndim)
    }

    let batch = decoded.dim(0)
    guard batch >= 1 else {
      throw ImageConversionError.emptyBatch
    }

    let channels = decoded.dim(1)
    guard channels == 3 else {
      throw ImageConversionError.invalidChannels(channels)
    }

    let height = decoded.dim(2)
    let width = decoded.dim(3)

    // Normalize from [-1, 1] to [0, 255]
    var rgb = decoded[0].asType(.float32)
    rgb = (rgb / MLXArray(2.0)) + MLXArray(0.5)
    rgb = MLX.clip(rgb, min: 0.0, max: 1.0)
    rgb = rgb * MLXArray(255.0)
    // Transpose from CHW to HWC
    rgb = rgb.transposed(1, 2, 0)
    let flat = rgb.reshaped(-1).asType(.uint8).asArray(UInt8.self)

    // Convert RGB to RGBA
    var rgba = [UInt8](repeating: 255, count: width * height * 4)
    var rgbIndex = 0
    for pixel in 0..<(width * height) {
      rgba[pixel * 4] = flat[rgbIndex]
      rgba[pixel * 4 + 1] = flat[rgbIndex + 1]
      rgba[pixel * 4 + 2] = flat[rgbIndex + 2]
      rgbIndex += 3
    }

    let data = Data(rgba)
    guard let provider = CGDataProvider(data: data as CFData) else {
      throw ImageConversionError.cgImageCreationFailed
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
    guard let cgImage = CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: provider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent
    ) else {
      throw ImageConversionError.cgImageCreationFailed
    }

    return cgImage
  }
}
