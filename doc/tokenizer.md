# Tokenizer

mlxforge ships its **own from-scratch byte-level BPE tokenizer** in C++
(`src/tokenizer/`), built only on `nlohmann/json` — there is no Rust, no
`tokenizers-cpp`, no SentencePiece. It reproduces the HuggingFace "fast"
tokenizer pipeline for a `tokenizer.json` whose `model.type == "BPE"` with a
`ByteLevel` decoder, which is what Llama-3.2 uses.

Why bother re-implementing it? Tokenization is part of the
numerical-correctness contract. If our ids drift from `mlx-lm`'s by even one
token, the prompt the model sees differs from the golden reference and every
downstream comparison is meaningless — but it would *look* fine (no crash, just
subtly different text). So encode/decode are validated to produce **byte-identical
ids** to the HF tokenizer against committed mlx-lm golden fixtures
(`reference/fixtures/tokenizer_corpus.json`). See the
[golden-reference discipline](./supported-models.md#the-golden-reference-discipline).

## Where it lives

| File | Responsibility |
| --- | --- |
| `tokenizer/bpe.{h,cpp}` | `BpeTokenizer` — the byte-level BPE engine: pre-tokenizer regex, GPT-2 byte↔unicode alphabet, merge loop, special-token segmentation. Pure / `const` / thread-safe. |
| `tokenizer/tokenizer.{h,cpp}` | `Tokenizer` — the user-facing wrapper: BOS handling, the Llama-3.2 chat template, and the `StreamingDetokenizer`. |
| `tokenizer/unicode_tables.h` | Unicode category lookups (`is_letter`, `is_number`, `is_whitespace`) the pre-tokenizer regex needs. |

`BpeTokenizer` carries no mutable state after construction, so encode/decode need
no mutex — the worker and HTTP request threads can call it concurrently. (See the
threading model in [architecture.md](./architecture.md).)

## The encode pipeline

`BpeTokenizer::encode` mirrors the HF fast pipeline, stage by stage:

```
text
 │  1. special-token segmentation   (longest-literal match)
 ▼
[plain segment]  [special id]  [plain segment]  …
 │  2. pre-tokenizer split          (llama3 tiktoken-style regex)
 ▼
[pre-token]  [pre-token]  …
 │  3. GPT-2 byte→unicode           (each byte → a printable codepoint)
 ▼
[merge-alphabet string]
 │  4. BPE merge loop               (with ignore_merges fast path)
 ▼
[token id]  [token id]  …
```

### 1. Special-token segmentation

The input is first scanned for special added-token literals (e.g.
`<|begin_of_text|>`, `<|eot_id|>`, `<|start_header_id|>`). Matches are emitted
directly as their ids; the gaps between them are plain text fed to the rest of
the pipeline.

- Specials are tried **longest-literal-first** (`special_tokens_` is sorted by
  descending length), so `<|eot_id|>` wins over any shorter prefix.
- A `special_first_bytes_` set is a cheap pre-filter: the scan only attempts a
  match at positions whose byte could begin some special literal.

This is how a chat-templated prompt — which is full of `<|…|>` control tokens —
gets its control tokens mapped to exact ids while the human text between them is
BPE-encoded normally.

### 2. Pre-tokenizer split (the regex)

Each plain segment is split into pre-tokens by the Llama-3.2 tiktoken-style
`Split` regex. Rather than pull in a regex engine, `match_piece` in `bpe.cpp`
hand-rolls it, replicating the alternatives **in order** (regex alternation is
first-match-wins):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)        contractions, case-insensitive
| [^\r\n\p{L}\p{N}]?\p{L}+          optional non-letter lead, then letters
| \p{N}{1,3}                        1–3 digits
|  ?[^\s\p{L}\p{N}]+[\r\n]*         optional space, symbols, trailing newlines
| \s*[\r\n]+                        whitespace ending at the last newline
| \s+(?!\S)                         a whitespace run, less one if non-space follows
| \s+                               any remaining whitespace
```

The fiddly parts are the three whitespace alternatives (5–7), which is why the
golden corpus deliberately exercises whitespace runs, newlines, and trailing
spaces. Unicode `\p{L}`/`\p{N}`/`\s` membership comes from `unicode_tables.h`.

### 3. GPT-2 byte→unicode alphabet

The vocab is expressed in the reversible GPT-2 ByteLevel alphabet: printable
ASCII and a couple of Latin-1 ranges map to themselves; every other byte maps to
a codepoint at U+0100+. So a space byte (`0x20`) is the token character `Ġ`, a
newline is `Ċ`, and so on. Each pre-token's **original bytes** (sliced by offset,
so raw bytes survive even invalid UTF-8) are mapped through this alphabet before
BPE runs. The `ByteMap` is built once as a function-local static (`byte_map()`).

### 4. BPE merge loop

For each mapped pre-token, `bpe_piece`:

1. **`ignore_merges` fast path** — if the whole pre-token is itself a vocab
   entry, emit its id directly without merging. Llama-3.2 sets this, and it is
   required for byte-exact parity.
2. Otherwise split into single-codepoint symbols and repeatedly merge the
   **lowest-rank** adjacent pair (merge rank = line order in `model.merges`),
   merging all occurrences of that pair left-to-right each pass, until no
   ranked pair remains — the GPT-2 / HF reference behavior.
3. Look up each surviving symbol in the vocab to get its id.

## Decode

`BpeTokenizer::decode` reverses the alphabet: each id → its token string →
codepoints → back to raw bytes via the inverse byte map, concatenated. Special
ids are skipped (matching mlx-lm's `skip_special_tokens` default) and
out-of-range ids are ignored. Because the byte map is a bijection, decode is the
exact inverse of the byte→unicode step.

## The `Tokenizer` wrapper

`BpeTokenizer` is the engine; `Tokenizer` (`tokenizer/tokenizer.{h,cpp}`) is what
the rest of the codebase calls. It adds the three things that are *policy* rather
than BPE mechanics:

- **BOS handling.** `BpeTokenizer::encode` does **not** run the BOS
  post-processor (matching the Rust `Encode`). The wrapper prepends the
  configured `bos_id` (default `128000` = `<|begin_of_text|>`; `-1` adds none),
  matching `mlx-lm`'s `tok.encode`. Decode pre-filters special ids symmetrically.
- **Chat template.** `apply_chat_template` / `render_chat_template` render the
  Llama-3.2 header format (`<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`)
  and inject the default "Cutting Knowledge Date / Today Date" system preamble.
  The format is selected from `config.json`'s `model_type` via
  `chat_format_from_model_type` → `ChatFormat`, keeping the (shared) forward pass
  decoupled from the (per-family) prompt formatting. Only `Llama3` exists today;
  the `enum` is the seam for new families.
- **Streaming detokenization.** `StreamingDetokenizer` is fed one new id at a
  time and returns only the text that has become **complete UTF-8** — it never
  emits a broken multi-byte character or a partial byte-BPE sequence mid-stream
  (which is what SSE token streaming needs). `utf8_complete_len` finds the
  longest complete-UTF-8 prefix; `finish()` flushes the tail.

## Supported families and the routing seam

`BpeTokenizer::is_supported` returns `true` only for `model.type == "BPE"` **and**
a `ByteLevel` decoder. `Tokenizer::from_file` calls it and **throws** otherwise —
in particular a Metaspace / SentencePiece tokenizer is rejected rather than
silently mistokenized, because the byte-level pipeline and the hand-rolled
splitter are only correct for byte-level BPE.

Re-onboarding a non-Llama family therefore means (a) routing its tokenizer to an
appropriate backend behind this same `is_supported` check, and (b) adding its
chat format to `ChatFormat` / `render_chat_template`. See
[Adding a new model family](./supported-models.md#adding-a-new-model-family).

## Validation

The tokenizer is gated by `tests/tokenizer/` against committed golden ids dumped
from `mlx-lm`:

- `reference/fixtures/tokenizer_corpus.json` — `(text, ids)` pairs over a corpus
  that stresses the pre-tokenizer's edge cases: whitespace runs, newlines,
  contractions, digits, CJK, accented Latin, emoji / ZWJ sequences, code, and
  inline special tokens. `bpe_test.cpp` asserts `encode` is byte-identical and
  that `decode(encode(x)) == x` for special-token-free strings.
- The corpus is regenerated by `reference/dump_ref.py` (run via the throwaway
  venv described in [contributing.md](./contributing.md)). It is **model-gated**:
  the tests self-skip with a `MESSAGE` when the model isn't present locally, so a
  green `ctest` without the model has *not* exercised tokenizer parity — download
  the model to run it.

When debugging a parity mismatch, narrow it to a single corpus entry (the test
`INFO`s the input), then bisect by stage: print the pre-token boundaries
(`match_piece`), the byte→unicode mapping, and the merge sequence against the HF
tokenizer's `tokenizer.encode(...).tokens`.
