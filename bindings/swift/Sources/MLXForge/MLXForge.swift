import CMLXForge
import Foundation

/// An error carrying the message reported by the C ABI's `char** err`.
public struct MLXForgeError: Error, CustomStringConvertible {
  public let message: String
  public var description: String { message }
}

/// Sampling parameters. The default is deterministic greedy decoding; the C ABI
/// normalizes the "disabled" sentinels, so a default value is always valid.
public struct Sampling {
  public var temperature: Float = 0
  public var topK: Int32 = 0
  public var topP: Float = 0
  public var minP: Float = 0
  public var repetitionPenalty: Float = 0
  public var frequencyPenalty: Float = 0
  public var presencePenalty: Float = 0
  public var seed: UInt64 = 0
  public var maxTokens: Int32 = 0

  public init() {}
  public static var greedy: Sampling { Sampling() }

  fileprivate var c: mlxforge_sampling {
    mlxforge_sampling(
      temperature: temperature, top_k: topK, top_p: topP, min_p: minP,
      repetition_penalty: repetitionPenalty, frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty, seed: seed, max_tokens: maxTokens)
  }
}

public struct ChatMessage {
  public let role: String
  public let content: String
  public init(role: String, content: String) {
    self.role = role
    self.content = content
  }
}

/// A batched MLX LLM engine. Many `chat()`/`text()` calls may run concurrently
/// on one engine — they share its continuous-batching scheduler.
public final class Engine {
  private let handle: OpaquePointer

  public init(_ spec: String, maxWaiting: Int32 = 0) throws {
    var opts = mlxforge_engine_opts(max_waiting: maxWaiting)
    var err: UnsafeMutablePointer<CChar>?
    guard let h = mlxforge_engine_create(spec, &opts, &err) else {
      let message = err.map { String(cString: $0) } ?? "engine create failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    handle = h
  }

  deinit { mlxforge_engine_free(handle) }

  public var ready: Bool { mlxforge_engine_ready(handle) != 0 }
  public var modelName: String { String(cString: mlxforge_engine_model_name(handle)) }

  /// Poll until the worker has finished loading the model.
  public func waitReady() async {
    while !ready { try? await Task.sleep(nanoseconds: 10_000_000) }
  }

  /// Construct an engine and resolve once the model has finished loading.
  public static func load(_ spec: String, maxWaiting: Int32 = 0) async throws -> Engine {
    let engine = try Engine(spec, maxWaiting: maxWaiting)
    await engine.waitReady()
    return engine
  }

  /// Stream a chat completion as decoded text chunks.
  public func chat(_ messages: [ChatMessage], sampling: Sampling = .greedy) throws
    -> AsyncThrowingStream<String, Error>
  {
    Engine.stream(try submitChat(messages, sampling))
  }

  /// Stream a raw-text completion (no chat template).
  public func text(_ prompt: String, sampling: Sampling = .greedy) throws
    -> AsyncThrowingStream<String, Error>
  {
    var s = sampling.c
    var err: UnsafeMutablePointer<CChar>?
    guard let req = mlxforge_submit_text(handle, prompt, &s, &err) else {
      let message = err.map { String(cString: $0) } ?? "submit failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    return Engine.stream(req)
  }

  /// Run a chat to completion and return the full string.
  public func complete(_ messages: [ChatMessage], sampling: Sampling = .greedy) async throws
    -> String
  {
    var out = ""
    for try await chunk in try chat(messages, sampling: sampling) { out += chunk }
    return out
  }

  private func submitChat(_ messages: [ChatMessage], _ sampling: Sampling) throws -> OpaquePointer {
    // Own the C strings for the duration of the submit call.
    var owned: [UnsafeMutablePointer<CChar>] = []
    defer { owned.forEach { free($0) } }
    var msgs: [mlxforge_msg] = []
    msgs.reserveCapacity(messages.count)
    for m in messages {
      let role = strdup(m.role)!
      let content = strdup(m.content)!
      owned.append(role)
      owned.append(content)
      msgs.append(mlxforge_msg(role: role, content: content))
    }
    var s = sampling.c
    var err: UnsafeMutablePointer<CChar>?
    let req = msgs.withUnsafeBufferPointer { buf in
      mlxforge_submit_chat(handle, buf.baseAddress, buf.count, &s, &err)
    }
    guard let req else {
      let message = err.map { String(cString: $0) } ?? "submit failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    return req
  }

  // Drive the blocking C-ABI poll on a background queue, yielding chunks; the
  // request is freed when the stream ends. (MLX work stays on the engine's own
  // worker thread — this thread only touches the request's token queue.)
  private static func stream(_ req: OpaquePointer) -> AsyncThrowingStream<String, Error> {
    AsyncThrowingStream { continuation in
      DispatchQueue.global(qos: .userInitiated).async {
        while true {
          var text: UnsafeMutablePointer<CChar>?
          let rc = mlxforge_request_next(req, &text)
          if rc == 0, let t = text {
            continuation.yield(String(cString: t))
            mlxforge_string_free(t)
          } else {
            break
          }
        }
        mlxforge_request_free(req)
        continuation.finish()
      }
    }
  }
}
