# 言語に依らない見本（vectors）

Python の参照実装（`tools/contracts/`）を読まずに、ここだけで同じ結果を確かめるための見本。すべて JSON。
再生成：`python tools/contracts/make_vectors.py`（リポジトリの直下で）。`canonicalize.json` の JCS の出力は手で書いた値（実装の出力ではない）。

| ファイル | 内容 | 手順 |
|---|---|---|
| `canonicalize.json` | 構文解析の段と JCS | `input_utf8`（UTF-8 に符号化）か `input_hex`（生のバイト）を strict_load → JCS。`accept` は `jcs_utf8` と一致し、そのバイト列の sha256 が `jcs_sha256`。`reject` は `code` で拒否 |
| `quantize.json` | 量子化 | `value_bits`（IEEE-754 binary64、ビッグエンディアン）を `q = round_ties_to_even(value × 2^(−exponent))`。範囲 `envelope`＝±(2^53−1)、`int64`＝int64 |
| `ids.json` | `session_id`（実験）・`observation_id` | 入力から id を計算 |
| `envelope-ids.json` | 封筒の `content_id`・`artifact_id` | `lookup`（artifact_id → `{content_id, observation_id, seq}`）を使い、各 `case.envelope` から計算して `expect` と一致。`relations` は 2 つの case の id が同じか違うか |
| `blob/blob-refs.json` | Blob の参照（`common/blob.schema.json`、0.7.0。IC-00 の凍結の外） | `ref` を blob.schema に当て（外れたら `schema` のみ）、`data_hex`（あれば）で `check_blob`：S34・S24/S25・S36・`count_negative`。`data_hex` が無い見本は参照だけ |
| `check-envelope.json` | `check_envelope` | `case.envelope` と `case.lookup` から、違反の符号の集合が `expect.codes`（空なら受理） |

拒否の符号：`E_UTF8`・`E_SYNTAX`・`E_SURROGATE`・`E_FLOAT_TOKEN`（小数点・指数）・`E_CONSTANT`（NaN・Infinity）・`E_DUP_KEY`・`E_INT_RANGE`・
`E_QUANT_NONFINITE`・`E_QUANT_RANGE`・`E_ID_INPUT`。`check_envelope` の符号：`schema`・`S4`・`S5`・`S8`・`S16`・`parent_unknown`・`calibration_unknown`・
`calibration_mismatch`・`content_id_mismatch`・`artifact_id_mismatch`・`lookup_incomplete`。

注意（Rust）：`-0` は整数のトークンで値は 0 として受理する（`serde_json` は `-0` を浮動小数にするので、トークンの字面で判定すること）。
