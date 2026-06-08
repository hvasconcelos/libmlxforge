import CMLXForge
import XCTest

@testable import MLXForge

final class MLXForgeTests: XCTestCase {
  func testAbiAndVersion() {
    XCTAssertEqual(mlxforge_abi_version(), MLXFORGE_ABI_VERSION)
    XCTAssertFalse(String(cString: mlxforge_version()).isEmpty)
  }

  func testBadSpecThrows() {
    XCTAssertThrowsError(try Engine("")) { error in
      XCTAssertTrue(error is MLXForgeError)
    }
  }

  func testGenerationIfModelPresent() async throws {
    guard let dir = ProcessInfo.processInfo.environment["MLXFORGE_MODEL_DIR"], !dir.isEmpty else {
      throw XCTSkip("MLXFORGE_MODEL_DIR not set; skipping model generation test")
    }
    let engine = try await Engine.load(dir)
    var s = Sampling.greedy
    s.maxTokens = 16
    let out = try await engine.complete(
      [ChatMessage(role: "user", content: "What is the capital of France?")], sampling: s)
    XCTAssertFalse(out.isEmpty)
  }
}
