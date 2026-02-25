/// A one-shot `@unchecked Sendable` box for transferring non-Sendable values
/// across concurrency boundaries (e.g., into a `Task.detached` closure).
///
/// The value is consumed on first access and the box becomes empty.
/// This is safe because the value is transferred exactly once, maintaining
/// exclusive ownership semantics.
///
/// Usage:
/// ```swift
/// let box = SendableBox(nonSendableValue)
/// Task.detached {
///   let value = box.take()  // transfers ownership
/// }
/// ```
public final class SendableBox<T>: @unchecked Sendable {
  private var value: T?

  public init(_ value: T) {
    self.value = value
  }

  /// Consume and return the boxed value. Returns `nil` if already taken.
  public func take() -> T? {
    let v = value
    value = nil
    return v
  }
}
