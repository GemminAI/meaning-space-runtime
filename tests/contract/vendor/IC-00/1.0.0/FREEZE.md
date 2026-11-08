# IC-00 1.0.0 — 凍結の記録

- **凍結の日**：2026-11-06
- **決定**：D-95（IC-00 を 1.0.0 として凍結する）、D-96（凍結の条件の修正と `$id` の形。Tomo 11-06 04:47）。`OKF/Decisions.md`
- **設計書**：`SensOS 受け渡し契約 設計書 v0.7.2.md`（vault の `日報/202611/20261106/`。sensos には入れない）
  - sha256：`f4a865ccf58027c64ed4b9197bc9bd0ebb76fa8f63bd69e1bf7c7a7dd137b911`
- **タグ**：`contracts/ic00-v1.0.0`（注釈つき。この FREEZE.md を含む commit に付ける）
- 下の sha256 は `tests/contracts/test_freeze.py` が実際のファイルと照合する。凍結の後にファイルが変われば、その試験が落ちる（変更は下の「変更の規則」に従い、新しい版として行う）。

## 凍結の範囲（設計書 §2.5 の表の 7 項目）

| # | 範囲 | 当たるファイル |
|---|---|---|
| 1 | 封筒の構造と JSON Schema | スキーマ（下） |
| 2 | `content_id`・`artifact_id` の作り方 | 設計書 §2.1、`tools/contracts/ids.py`、見本 `envelope-ids.json`・`ids.json` |
| 3 | JCS・量子化・数のトークンの検査・重複キーの拒否 | `jcs.py`・`quantize.py`・`strict_load.py`、見本 `canonicalize.json`・`quantize.json` |
| 4 | `producer`・`parents`・`session_id`・`observation_id` の識別の規則 | 設計書 §2.1〜§2.4、`ids.py`・`check.py`、見本 `ids.json`・`envelope-ids.json` |
| 5 | 不変条件と検査の範囲（S4・S5・S8・S16、スキーマ、id の計算し直し、親の既知） | `check.py`、見本 `check-envelope.json` |
| 6 | 受理・拒否の見本 | 見本（下） |
| 7 | 参照実装との整合 | 参照実装（下）、設計書 付録 B |

### スキーマ（`common/`）

| ファイル | sha256 |
|---|---|
| `docs/architecture/contracts/common/artifact.schema.json` | `16f715bcd8e65e5c07085160c4b91b565f8564eeea00ac8fcd9162d17dabc8e2` |
| `docs/architecture/contracts/common/provenance.schema.json` | `dcde3f691b52241d5e347df94a36b6793d07d779ad5a990a1e3f7ae3a4a4dbea` |
| `docs/architecture/contracts/common/uncertainty.schema.json` | `8ce4a7a2f52e9913eacf180d07b08a63b7cc574024b39c3aa0323dca1448a517` |

### 参照実装（`tools/contracts/` の IC-00 の範囲）

正規化・量子化・厳密な読み込み・id・`check_envelope`（とその `errors.py`・`__init__.py`）。

| ファイル | sha256 |
|---|---|
| `tools/contracts/__init__.py` | `3f8f5d3f91c991164fe4f924ec150d8afda0992fcf94e4839771aab9930c5b6d` |
| `tools/contracts/errors.py` | `63ecc07b6069e620e093c78ba5a1398e24b3440cd7c7ad28fd4203ef38fa1df5` |
| `tools/contracts/strict_load.py` | `5343c06789ce8b7b17481f47a0b354654d8e89f144d5fcfd1742cd88987c2ca6` |
| `tools/contracts/jcs.py` | `6f98c150c1398462524fea0907c779ed71192ba1e54ed0fb2f6b0dc62803f7e0` |
| `tools/contracts/quantize.py` | `de149111667554c00d4ff5ca62e070f2c210c44465b1fc39c25bd94b8fd7a56c` |
| `tools/contracts/ids.py` | `157b0d19d288989dd65ae8b40483af086e802332e281bb79084991ed89e4be85` |
| `tools/contracts/check.py` | `1797e561da83dda1c101b1d174e244f910212f5c4f60bb36b6eb1544b982686a` |

`schemas.py`（スキーマの読み込みの補助。新しい契約を足すたびに一覧が変わるので、凍結の対象にしない）と `make_vectors.py`（見本の生成器。blob の見本も作る）は範囲に入れない。

### 見本（`common/vectors/`）

| ファイル | sha256 |
|---|---|
| `docs/architecture/contracts/common/vectors/canonicalize.json` | `bc31183bc895f61c7a7f4152380f884348058b53236d1c31d3a47c274513131b` |
| `docs/architecture/contracts/common/vectors/quantize.json` | `a80a217b1841e3c8c49d8a942eafa646de6368b2fe4494c84c2ce8245f15b2ec` |
| `docs/architecture/contracts/common/vectors/ids.json` | `b54cc6fe0e049188727f409eeaf5839bdc9c51b7686166849d6f1c553922f180` |
| `docs/architecture/contracts/common/vectors/envelope-ids.json` | `463f2d235d3df138c1665689f3811a9e84ec113e9233c835e01509e441324a36` |
| `docs/architecture/contracts/common/vectors/check-envelope.json` | `06002711aebd0c8fcb06e60bf4fdf75768f3b7d7aa9c5ecae3986779c7ba671e` |

## 凍結の範囲の外

- IC-01〜IC-06 の契約（payload の各契約）。
- `common/blob.schema.json`（0.7.0、草案）と `common/vectors/blob/`。
- `msr/` の契約（geometry-measurement・calibration、0.7.0、草案）。
- `tools/contracts/gate.py`・`blob.py`・`ic02.py`（凍結の後も変えてよい検査器）。

## 凍結の条件の確かめ（設計書 §2.5、D-96）

1. **スキーマの差分**：6a34b99（0.6.2）から、`common/` の 3 スキーマの差分は版の文字列（`$id`、`contract.version` の const、provenance への `$ref`）と、`artifact.schema.json` の description の中の設計書の版だけ。構造・制約は一字も変えていない。provenance・uncertainty の description に設計書の版は無い。
2. **Python の参照実装の試験**：`tests/contracts` は凍結の確かめの時点で 272 件がすべて通過（この FREEZE の試験を足した後は件数が増える。報告を参照）。見本：canonicalize 46（受理 16・拒否 30）、quantize 29（21・8）、ids 10（5・5）、envelope-ids 17 件と関係 16、check-envelope 37（受理 7・拒否 30）。見本のファイルは生成器の出力と一致する。
3. **独立の確かめ**（設計書 付録 B.2。参照実装のコードを読まずに §2.1 の前像だけから書いた計算器）：
   - envelope-ids：17／17 一致、関係 16 すべて期待どおり。すべての封筒の `contract` は `{"id": "IC-00", "version": "1.0.0"}`。
   - check-envelope：受理の 7／7 一致。拒否の 30 件で一致しないのは、拒否が期待される理由（id の不一致・未知の親・未整列・スキーマ違反など）のものだけ。
   - 対照：`contract.version` を 0.6.2 に戻すと一致は 0 件（版が前像に入っており、確かめが空振りしていない）。
4. 見本の id は description に依らない（見本は作り直していない。上の sha256 が ab3baed の見本と同じ）。

## ほかの言語の実装（D-96）

Rust などの実装は凍結の条件に入れない（sensos に Rust の参照実装は無い）。作る側が凍結した見本を全件通してから使う。見本は言語に依らない規範で、通らなければその実装の誤りとし、契約は変えない。

## 変更の規則（設計書 §2.5）

| 変更 | 扱い | 例 |
|---|---|---|
| 誤記・説明の明確化（規範の意味も、id も、受理・拒否も変えない） | Patch（1.0.x） | 言い回しの修正、例の追加 |
| 後方互換の拡張（既存の成果物の受理・意味・id を変えない） | Minor（1.x.0） | `producer.name` の正規の値の追加、任意の欄の追加で、無いときの意味が今と同じもの |
| 既存の成果物の解釈か id の作り方に影響する変更 | Major（2.0.0） | 前像の欄の追加・削除、整列の規則、量子化の規則、受理していたものを拒否する変更 |

- **既存の IC-00 の成果物の意味を、後から変えない**。Major を上げても、1.x の成果物は 1.x の規則で読み、id は 1.x の規則で計算し直す（封筒の `contract.version` がどちらの規則かを決める）。
- Minor でも、受け取る側（Gate）が新しい値を知らなければ拒否するもの（`producer.name` の追加）は、Gate の更新を先に行う。
- `$id` の形は `sensos:contracts/<フォルダ>/<名前>/<版>`、`payload_schema.id` は `$id` から版を除き `/` を `.` にした名前（D-96）。
