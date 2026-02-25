import Foundation
import MLX

enum Flux2WiredMemory {
  private static let requestPolicy = WiredSumPolicy(
    id: UUID(uuidString: "A2D8E6E2-3D06-4F8C-9E23-2D606A93A6B1")!
  )

  static func requestTicket(limit: Int) -> WiredMemoryTicket {
    WiredMemoryTicket(
      size: max(0, limit),
      policy: requestPolicy,
      manager: .shared,
      kind: .active
    )
  }
}
