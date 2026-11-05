# msr/ — IC-02（MSR の計測）と calibration の契約

| ファイル | 版 | 中身 |
|---|---|---|
| `geometry-measurement.schema.json` | **0.7.0** | 計測の payload（`mode: single` / `pair`）。`series` は系列の名前をキーにした整数の Blob（§3.3.2） |
| `calibration.schema.json` | **0.7.0** | calibration の payload。0.6.1 からの変更は `$id` の版の文字列だけ（欄・制約は同じ。設計書 §3.3.1 の変更は規範でない細目と、ファイル・添え書きの扱いだけで、スキーマに影響しない） |
| `archive/0.6.1/`・`archive/0.6.2/` | — | 付録 A の旧スキーマ（0.6.1 の calibration、0.6.2 の計測）。**そのまま残す**（書き換えない。`contract.lock` の `archived` に sha256） |

Blob の参照の型（`intBlob`・`floatBlob`）は `../common/blob.schema.json`（0.7.0）。`series` の各系列は `intBlob` への `$ref`。
`payload_schema` の `{id, version}` は `msr.geometry-measurement` `0.7.0`、`msr.calibration` `0.7.0`（`$id` の版。IC-00 の `contract.version` とは別。設計書 §2.6、S39）。

定義の出典：meaning-space-runtime の `feature/msr-ic02`（`13d0a51`）の `src/msr/geometry/metrics.py` の冒頭。この README が**規範**で、コードの冒頭は正本にしない（設計書 §3.3.2）。

## 系列と指標

軌跡は `T` 点（`token_index` が増える順、S1）の `x_0 … x_{T-1}`（`D` 次元）。**`series[name][t]` は点 `0..t` から計算した値**（`t = 0..T-1`）。
最後の要素（`t = T-1`）は `payload.metrics` の同じ値と同じ整数（S37）。`ε`・`δ`・`w_min` は payload の `thresholds`（封筒の整数。距離は 2⁻³⁰ の刻みを実数に戻したもの）。
**値なし**は Blob の要素が −2^63（Blob に `undefined: "int64_min"` を宣言する）。値なしは下の表の条件に当たる位置にだけ現れ、それ以外の位置が値なしなら誤り（S36）。逆に、条件に当たる位置は必ず値なし。

| 系列の名前（キー） | `metrics` の場所 | 種類・単位（`unit`） | 定義 | 値なしになる位置 |
|---|---|---|---|---|
| `displacement_mean` | `displacement.mean` | 実数・`distance` | 変位 `d_s = ‖x_s − x_{s−1}‖`（`s = 1..t`）の平均 | `t = 0`（変位が無い） |
| `displacement_median` | `displacement.median` | 実数・`distance` | 同じ `d_s` の中央値 | `t = 0` |
| `displacement_longest_sub_delta_run` | `displacement.longest_sub_delta_run` | 個数・`steps` | `d_s < δ` が続く最長の長さ（`s = 1..t`） | `δ` が null なら全位置。そうでなければ `t = 0` |
| `recurrence_pair_fraction` | `recurrence.pair_fraction` | 実数・`fraction` | 対 `i < j ≤ t`（`j − i > w_min`）のうち `‖x_i − x_j‖ < ε` の割合 | `ε` か `w_min` が null なら全位置。そうでなければ対が 1 つも無い位置、すなわち `t ≤ w_min` |
| `recurrence_return_count` | `recurrence.return_count` | 個数・`returns` | 「戻り」の数。`j ≤ t` で、ある過去の点 `i`（`j − i > w_min`）が `‖x_i − x_j‖ < ε` かつ `‖x_i − x_{j−1}‖ ≥ ε`（ε の球を出て戻った）となる `j` の数 | `ε` か `w_min` が null なら全位置。そうでなければ無し（0 から始まる） |
| `recurrence_return_rate` | `recurrence.return_rate` | 実数・`returns_per_point` | 戻りの数 ÷ `(t + 1)`（0〜1。1 点につき戻りは高々 1 回） | `recurrence_return_count` と同じ |
| `diversity_cell_entropy` | `diversity.cell_entropy` | 実数・`nats` | 点 `0..t` の符号帳のセル（最も近い中心）の占有のシャノン・エントロピー（自然対数） | 符号帳を使っていない（封筒の `parents` に役割 `codebook` が無い）なら全位置。そうでなければ無し |
| `diversity_participation_ratio` | `diversity.participation_ratio` | 実数・`dimensionless` | 点 `0..t` の共分散 `C`（点の数で割る）について `tr(C)² / tr(C²)`（`tr(C²)` が 0、つまり全点が一致なら 0） | `t = 0` |
| `persistence_longest_stay` | `persistence.longest_stay` | 個数・`points` | 点のセル番号が同じまま続く最長の長さ。セル番号は、符号帳があれば最も近い中心、なければ `ε` を半径とする先頭からの逐次のクラスタリング（点が既存のどの代表点とも `ε` 以上離れれば新しい代表点） | 符号帳が無く、かつ `ε` が null なら全位置。そうでなければ無し |
| `persistence_dominant_cell_share` | `persistence.dominant_cell_share` | 実数・`fraction` | 最も多くの点を持つセルの点数 ÷ `(t + 1)` | `persistence_longest_stay` と同じ |

- 実数の系列は Blob の `quantization` が `{quantum: {base: 2, exponent: −30}, rounding: ties-to-even, tolerance_quanta: 1}`。**個数の系列は `quantization: null`** で、値は 0 以上の整数そのもの（許容は 0）。
- 実数の系列の要素は `round_ties_to_even(値 × 2³⁰)`。範囲外・NaN・無限大は丸めずに計測を拒否する（§7.1）。
- 各 Blob は `shape: [T]`、`T = input_ref.length`（S35）。`uri` は `blob:sha256:<sha256>`（S34）、バイト列は int64・リトルエンディアン・C の順・ヘッダなし（S24）。
- 系列の名前の集合は 10 個で固定（スキーマの `additionalProperties: false`・すべて必須）。名前を足す・変えるのはスキーマの版を上げること。

## ペアの判定の式（設計書 §3.3.3、S38）

入力はすべて整数で、同じ単位（軌跡の空間の L2 距離）・同じ刻み（2⁻³⁰）：`d` ＝ `payload.distance.value`、`lo`・`hi` ＝ calibration の `values.noise_floor_lower`・`noise_floor_upper`。

| 条件 | `verdict.state` |
|---|---|
| `payload.calibration` が null | `UNCALIBRATED` |
| `d < lo` | `STAGNANT` |
| `lo ≤ d ≤ hi` | `NOISE_FLOOR_INTERNAL` |
| `d > hi` | `STRUCTURAL_SHIFT_DETECTED` |

- 量子化した後の整数で比べる（浮動小数で比べてから量子化しない）。境界（`d = lo`、`d = hi`）は `NOISE_FLOOR_INTERNAL`。
- calibration は `lo`・`hi` を必ず持つ（`lo ≤ hi` は S19）。noise floor を作れないときは calibration を作らず、比較は `UNCALIBRATED`。
- `x-common_prefix` の比較も同じ式（検査器は「拡張による判定」として区別して記録する）。
- 1 本の計測（`single`）の `verdict.state` は null（S3）。

## 検査器（`tools/contracts`）での扱い

`blob.py`（S34・S24/S25・S36 の宣言・個数の負の値）、`ic02.py`（S35・S36 の位置・S37・S38）、`gate.py`（S39 と payload のスキーマ）。
Blob の見本は `../common/vectors/blob/blob-refs.json`。
