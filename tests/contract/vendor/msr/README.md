# msr/ — IC-02（MSR の計測）と calibration の契約

**IC-02 は 0.9.0**（D-129：先頭 128 行の固定窓を契約に入れた）。IC-02 の版は下の表の組で、各 payload の版は `$id` の版（IC-00 の `contract.version` とは別）。

| ファイル | 版 | 中身 |
|---|---|---|
| `geometry-measurement.schema.json` | **0.9.0** | 計測の payload（`mode: single` / `pair`）。`series` は系列の名前をキーにした整数の Blob（§3.3.2）。0.7.1 からの変更：`single` に `input_scope`（必須。`full` か `window`）を足した。`pair` は変えていない。下の「窓」の節 |
| `trajectory-window.schema.json` | **0.9.0**（新） | 窓の成果物の payload（`msr.trajectory-window`）。元の軌跡の先頭 128 行を別の成果物として持つ（D-129）。下の「窓」の節 |
| `calibration.schema.json` | **0.8.0**（変更なし） | calibration の payload。`exponent`（整数、必須）。下の「calibration の量子化の指数」の節 |
| `archive/0.6.1/`・`archive/0.6.2/`・`archive/0.7.0/`・`archive/0.7.1/`・`archive/0.8.0/` | — | 旧スキーマと記録の写し。**そのまま残す**（書き換えない。スキーマは `contract.lock` の `archived` に sha256）。`archive/0.8.0/` は IC-02 0.8.0 の計測のスキーマ（`$id` は 0.7.1、`input_scope` が無い）と、0.8.0 の README。0.7.1 の計測は旧版の規則のまま Gate が読む（`input_scope` を要求せず、足すと拒否。0.9.0 としては読まない）。0.7.0 の calibration も旧版の規則のまま（指数の欄が無く −30 固定。0.8.0 としては読まない） |

Blob の参照の型（`intBlob`・`floatBlob`）は `../common/blob.schema.json`（0.7.0）。`series` の各系列は `intBlob` への `$ref`。
`payload_schema` の `{id, version}` は `msr.geometry-measurement` `0.9.0`（旧 `0.7.1` も旧規則で受理）、`msr.trajectory-window` `0.9.0`、`msr.calibration` `0.8.0`（`$id` の版。IC-00 の `contract.version` とは別。設計書 §2.6、S39）。

定義の出典：meaning-space-runtime の `feature/msr-ic02`（`13d0a51`）の `src/msr/geometry/metrics.py` の冒頭。この README が**規範**で、コードの冒頭は正本にしない（設計書 §3.3.2）。

## 系列と指標

軌跡は `T` 点（`token_index` が増える順、S1）の `x_0 … x_{T-1}`（`D` 次元）。**`series[name][t]` は点 `0..t` から計算した値**（`t = 0..T-1`）。
最後の要素（`t = T-1`）は `payload.metrics` の同じ値と同じ整数（S37）。`ε`・`δ`・`w_min` は payload の `thresholds`（封筒の整数。距離は 2⁻³⁰ の刻みを実数に戻したもの）。
**値なし**は Blob の要素が −2^63。**下の表の 10 系列はすべて値なしを持ちうるので、`undefined: "int64_min"` を常に宣言する**（実際に値なしが無い計測でも宣言する。表に無い系列に宣言があれば拒否する。S36）。値なしは表の条件に当たる位置にだけ現れ、それ以外の位置が値なしなら誤り（S36）。逆に、条件に当たる位置は必ず値なし。

**変位の区間が無いとき（`T = 1`）**：`displacement.count` は 0、`displacement.mean`・`median`・`longest_sub_delta_run` は null（値なし）、変位の 3 系列の `t = 0`（= 最後の要素）は値なし。S37 は「最後の要素が値なし ⇔ `metrics` の値が null」でそのまま成り立つ。

| 系列の名前（キー） | `metrics` の場所 | 種類・単位（`unit`） | 定義 | 値なしになる位置 | `undefined` の宣言 |
|---|---|---|---|---|---|
| `displacement_mean` | `displacement.mean` | 実数・`distance` | 変位 `d_s = ‖x_s − x_{s−1}‖`（`s = 1..t`）の平均 | `t = 0`（変位が無い） | 常に |
| `displacement_median` | `displacement.median` | 実数・`distance` | 同じ `d_s` の中央値 | `t = 0` | 常に |
| `displacement_longest_sub_delta_run` | `displacement.longest_sub_delta_run` | 個数・`steps` | `d_s < δ` が続く最長の長さ（`s = 1..t`） | `δ` が null なら全位置。そうでなければ `t = 0` | 常に |
| `recurrence_pair_fraction` | `recurrence.pair_fraction` | 実数・`fraction` | 対 `i < j ≤ t`（`j − i > w_min`）のうち `‖x_i − x_j‖ < ε` の割合 | `ε` か `w_min` が null なら全位置。そうでなければ対が 1 つも無い位置、すなわち `t ≤ w_min` | 常に |
| `recurrence_return_count` | `recurrence.return_count` | 個数・`returns` | 「戻り」の数。`j ≤ t` で、ある過去の点 `i`（`j − i > w_min`）が `‖x_i − x_j‖ < ε` かつ `‖x_i − x_{j−1}‖ ≥ ε`（ε の球を出て戻った）となる `j` の数 | `ε` か `w_min` が null なら全位置。そうでなければ無し（0 から始まる） | 常に |
| `recurrence_return_rate` | `recurrence.return_rate` | 実数・`returns_per_point` | 戻りの数 ÷ `(t + 1)`（0〜1。1 点につき戻りは高々 1 回） | `recurrence_return_count` と同じ | 常に |
| `diversity_cell_entropy` | `diversity.cell_entropy` | 実数・`nats` | 点 `0..t` の符号帳のセル（最も近い中心）の占有のシャノン・エントロピー（自然対数） | 符号帳を使っていない（封筒の `parents` に役割 `codebook` が無い）なら全位置。そうでなければ無し | 常に |
| `diversity_participation_ratio` | `diversity.participation_ratio` | 実数・`dimensionless` | 点 `0..t` の共分散 `C`（点の数で割る）について `tr(C)² / tr(C²)`（`tr(C²)` が 0、つまり全点が一致なら 0） | `t = 0` | 常に |
| `persistence_longest_stay` | `persistence.longest_stay` | 個数・`points` | 点のセル番号が同じまま続く最長の長さ。セル番号は、符号帳があれば最も近い中心、なければ `ε` を半径とする先頭からの逐次のクラスタリング（点が既存のどの代表点とも `ε` 以上離れれば新しい代表点） | 符号帳が無く、かつ `ε` が null なら全位置。そうでなければ無し | 常に |
| `persistence_dominant_cell_share` | `persistence.dominant_cell_share` | 実数・`fraction` | 最も多くの点を持つセルの点数 ÷ `(t + 1)` | `persistence_longest_stay` と同じ | 常に |

- 実数の系列は Blob の `quantization` が `{quantum: {base: 2, exponent: −30}, rounding: ties-to-even, tolerance_quanta: 1}`。**個数の系列は `quantization: null`** で、値は 0 以上の整数そのもの（許容は 0）。
- 実数の系列の要素は `round_ties_to_even(値 × 2³⁰)`。範囲外・NaN・無限大は丸めずに計測を拒否する（§7.1）。
- 各 Blob は `shape: [T]`、`T = input_ref.length`（S35）。`uri` は `blob:sha256:<sha256>`（S34）、バイト列は int64・リトルエンディアン・C の順・ヘッダなし（S24）。
- 系列の `unit` と種類（実数・個数）は**スキーマが正本**で、上の表はその写し（`tests/contracts` が一致を確かめる）。
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

## calibration の検証（0.7.1 の決定、Tomo 11-06 05:10）

- calibration の `content_id`・`artifact_id` を計算し直すには、親（`null_set` の各成果物、`codebook`）の `content_id` が要る。**親の `content_id` は HEKB から取得する**（取得の主体は nvs-runtime の Gate）。見つからない、または保存されている `content_id` と食い違うなら、検証は失敗する。
- 呼び出し側が任意に渡した親の情報（ファイルの写しに添えたもの、引数など）を、**無条件に信頼しない**。取得した親で `content_id`・`artifact_id` を計算し直し、封筒の値と一致することを確かめる。
- **ファイルだけで自己完結して検証できることは契約しない**。ファイルは写しで、正本は HEKB（設計書 §3.3.1）。
- 量子化の指数は calibration の payload の `exponent` に宣言する（次の節）。計測のスキーマの指数 −30 は定数のまま。

## calibration の量子化の指数（0.8.0 の仮決定、Tomo 11-07 15:29・15:30）

- payload に `exponent`（整数）を足した。**必須**。`values` の距離の欄（`noise_floor_lower`・`noise_floor_upper`・`background_mean`・`epsilon`・`delta`）の刻みは `2^exponent`。実数の値は `q × 2^exponent`（`math.ldexp`）。
- **−30 に固定しない**。許容範囲はここでは決めない（契約設計で後日）。スキーマにも Gate にも範囲の検査は無い。
- **Gate が検証するのは指数の有無と形式だけ**：欠落、文字列・小数（`-20.0` のような整数値の小数も）・真偽値・null は拒否（符号 `schema`）。
- **読む側は宣言された `exponent` で解釈する**（`epsilon` の復元など）。−30 を前提にしない。
- 分位点の水準（`method.quantiles`、確率）は 2⁻³⁰ の格子のままで、`exponent` の対象外。
- 計測（`geometry-measurement` 0.7.1 と 0.9.0）の `thresholds`・`distance` は 2⁻³⁰ のまま。刻みの違う calibration を黙って換算しない（MSR は拒否する）。Gate の `S38` は、整数どうしの比較が同じ刻みであることを前提にしており、刻みの照合は 0.8.0 では Gate に入れていない（指数の有無と形式だけ、という仮決定のため）。
- **0.7.x（指数の欄が無い）を 0.8.0 として暗黙に解釈しない**。`payload_schema` が 0.8.0 で `exponent` が無ければ拒否。0.7.0 は旧版の規則のまま（欄を足すと拒否）。旧データを 0.8.0 へ移す方法は別の決定で、ここでは実装しない。
- D-97 (5) の原文は変えていない。置き換える決定は Tomo が別に記録する。
- 見本は `vectors/calibration-exponent.json`（受理 3・拒否 8）。

## 窓（IC-02 0.9.0、D-129）

決定（D-129）：**先頭 128 行の固定窓を、IC-02 の窓入力として正式に入れる**。窓は**別の成果物**で、元の軌跡との対応は `parents` の親参照。`common_prefix` を承認して全長で計測する案は採らない（D-120 の承認の対象にしない）。

### 今回の実装（固定）

| 項目 | 固定の値 | 置き場所 |
|---|---|---|
| 規則 `window.rule` | `prefix_rows`（先頭から連続した行） | スキーマの定数 |
| 開始位置 `window.start` | `0` | スキーマの定数 |
| 長さ `window.length` | `128`（行） | スキーマの定数 |
| 窓の作り方 `policy.name` | `first-128-rows/1`（`policy.sha256` は名前の UTF-8 の sha256） | スキーマの定数 ＋ Gate |

窓が 128 行に満たない軌跡（元の長さ < 128）は窓にできない（`origin.length` の最小が 128。窓を作らず、全長の入力として扱う）。

### 窓の成果物（`msr.trajectory-window` 0.9.0）

| 項目 | 内容 |
|---|---|
| `kind` | `observation`（行を選んで切り出す機械的な操作で、解釈を作らない。D-89） |
| `producer.name` | `nvs-runtime`（処理の順序と受け渡しの主体、D-124。MSR は計測だけで窓を作らない）。IC-00 の登録は 1.0.0・1.1.0 とも `nvs-runtime`×`observation` を許す。窓の契約は kind と producer をさらに絞る（`window_kind`・`window_producer`） |
| 封筒の `space_id` | null でない。窓の空間（元の軌跡の空間とは別の識別。窓の長さが違えば別の空間）。計測の `space_id` は窓のものと同じ |
| `parents` | **ちょうど 1 つ**：役割 `source_trajectory`（`ordered: false`、`artifact_ids` は 1 件）＝元の軌跡の成果物。それ以外の親は持たない（`window_parents`）。元の成果物は HEKB にあること（`parent_unknown`） |
| `payload.origin` | `trajectory_id`（元の軌跡の行のバイト列の sha256＝RAW の識別子）、`position`（取得位置）、`length`（元の全長、128 以上） |
| `payload.window` | `rule`・`start`・`length`（上の固定値）、`content_sha256`（窓の中身の sha256） |
| `payload.points` | 窓の 128 行の Blob（binary32・リトルエンディアン・行優先・詰め物なし・量子化なし、`shape: [128, dimension]`）。NaN・±Inf は拒否（`window_nonfinite`）。**`content_sha256` ＝ `points.sha256` ＝ Blob のバイト列の sha256**（`window_sha256`） |
| `payload.token_index` | 窓の 128 行のトークン番号（int64 の個数の Blob、`shape: [128]`、厳密に増える。元の先頭 128 件で、0 から始まるとは限らない） |
| `payload.source`・`layer`・`dimension`・`model_build_id`・`policy` | 計測の `input_ref`・`layer`・`source` と同じ意味・同じ形（`TrajectoryInput` の欄） |

- 「窓の中身の sha256」は**窓の点の Blob のバイト列だけ**の sha256。`token_index` は自分の Blob の sha256 を持つ。
- **Gate が確かめるもの**：スキーマ（固定値）、kind・producer、親が 1 件であること、`content_sha256`・Blob の sha256 とバイト列、行数、有限性、親（元の成果物）が保存されていてその `raw_ref.trajectory_id`・`length`・`position` が `origin` と一致すること（`window_origin`。読む欄は草案 `nvs.hidden-trajectory` 0.1.0 の名前で、他の欄は読まない。読めなければ `window_origin_unreadable`）。
- **Gate が確かめられないもの**：窓が元の RAW の**先頭 128 行そのもの**であること（RAW は参照で持ち、成果物に行を入れないため）。RAW の行を持つ読み手は `prefix_matches`（`tools/contracts/ic02_window.py`）で確かめる。この確認を Gate の責任にするかは未決。
- 符号の一覧は `tools/contracts/ic02_window.py` の冒頭。見本は `vectors/window.json`。

### 計測（`msr.geometry-measurement` 0.9.0、`mode: single`）

`input_scope`（必須）が、計測した入力の範囲を宣言する。

| `input_scope` | 意味 | 親の役割 `trajectory` |
|---|---|---|
| `{"kind": "window", "rule": "prefix_rows", "start": 0, "length": 128, "source_trajectory_id", "content_sha256"}` | **窓の成果物を入力にした計測**。`input_ref.length` は 128（スキーマ） | 窓の成果物（`msr.trajectory-window`）1 件 |
| `{"kind": "full"}` | **窓なしの全長の入力**。軌跡をそのまま計測する。契約は長さの上限を決めない（実用の上限は MSR の資源の問題で、契約の外） | 窓でない軌跡の成果物 1 件 |

Gate の検査（`check_scope`）：
- `window`：`input_ref.length == scope.length`（スキーマ。窓の長さが 128 でなければ拒否）。親が `trajectory` に 1 件あり（`window_parent`）、保存されていて（`parent_unknown`）、窓の成果物であること（`window_not_window`）。親の窓が、`scope` の `source_trajectory_id`・`content_sha256`・`length`・`start` と、計測の `input_ref`（`dimension`・`model_build_id`・`policy`）・`layer`・`source`・`space_id` と一致すること（`window_mismatch`）。
- `full`：親が窓の成果物なら拒否（`window_undeclared`。窓を全長のふりで使わせない）。親の `length` が読めて整数なら、`input_ref.length` と一致すること（`full_length_mismatch`。宣言なしに先頭だけを計測させない）。読めない親（長さの欄が無い）は、この点を確かめない。
- **版の扱い**：`0.7.1`（IC-02 0.8.0）の計測は旧規則のまま。`input_scope` を要求せず、足すと `schema` で拒否。0.9.0 として暗黙に読まない。
- **`pair`（ペアの比較）は 0.9.0 で変えていない**。窓の成果物 2 つを `left`・`right` の親にして比べることはできる（`equal_only`）が、`pair` には `input_scope` を足していない。ペアが窓の入力であることの宣言は将来の拡張。

### 将来の拡張（今回の実装ではない）

次は**契約の文章としてだけ**書く。スキーマ・Gate・MSR は固定のまま。
- 窓の位置 `start` と長さ `length` をパラメータにする（末尾窓、中間の窓、128 以外の長さ）。欄は既に `window` と `input_scope` にあるので、定数を範囲に緩めるのは**追加の変更**で、版を上げる（0.9.0 のスキーマは固定のまま読み続ける）。そのとき `window.rule` を足す（例：`suffix_rows`）か、`prefix_rows` を一般化するかは、そのときに決める。
- 窓の位置が生成全体のループを反映するかは未確認（D-129）。末尾窓などは追加の実験として試す。窓の位置を変えるのは、その結果を見てから。
- `pair` への `input_scope` の追加。
- 窓の点の Blob の表現を `common/float32-blob`（草案）に寄せる（昇格したとき）。


## 検査の責任

- **完全なスキーマ検査は Gate が保存前に行う**（設計書 §5）。
- 成果物を**読む側**は、(1) `content_id`・`artifact_id` の再計算、(2) 使う欄の型と範囲、(3) 契約の版（知らない版は拒否、S39）、(4) 親の参照の整合、を検証する。Gate を経ていない成果物（ファイルの写しなど）を無条件に信頼しない。

## 親の参照整合（calibration の検証の中身、D-97 (2)）

calibration を**読む側**が、取得した親（`parent_contents`：親の `artifact_id` → 保存されている `content_id`）に対して確かめる 5 点。**1 つでも違えば検証失敗**で、その calibration は使わない。

| # | 規範 | 見本の符号 |
|---|---|---|
| 1 | 封筒の `parents` の各親（`artifact_ids` のすべて）が、取得・確認済みの親（`parent_contents`）の中にある。さらに、`parent_contents` の各 `content_id` は `cnt:` ＋ 64 文字の小文字の 16 進の形式である（形式違反はハッシュの不一致としてではなく、専用の符号で拒否する） | `parent_unknown`、`parent_content_id_form` |
| 2 | codebook の `content_id`（取得した親の）が、payload の `values.codebook_ref.content_id` と一致する。`codebook_ref` が null なら対象外 | `codebook_content_mismatch` |
| 3 | payload が参照する役割の集合（`null_set` は常に、`codebook` は `values.codebook_ref` が null でないときだけ）と、封筒の `parents` の役割の集合が一致する。役割が足りなくても余っても不一致。**`parents` の中に同じ役割が 2 つあるものは拒否する** | `role_set_mismatch`、`role_duplicate` |
| 4 | 役割 `null_set` と役割 `codebook` の `ordered` が、どちらも `false` である | `null_set_ordered`、`codebook_ordered` |
| 5 | payload の `null_set`（整列済みの `artifact_id` の列）が、役割 `null_set` の `artifact_ids` と一致する（順序も含めて同じ列） | `null_set_mismatch` |

- 違反は検証失敗である。補って読み進めない（推測で親を足さない、役割を直さない）。
- **取得と確認の主体は nvs-runtime（Gate）**：`parent_contents` は、Gate が HEKB から取得し、保存されている成果物と照合して確かめたものだけを渡す。**MSR は HEKB を呼ばない**。呼び出し側が任意に渡した親の情報は、上の 5 点を満たしても無条件には信頼しない（`content_id`・`artifact_id` の計算し直しが通ることが、さらに要る）。
- 見本は `vectors/calibration-parents.json`（0.8.0 の payload。受理 2・拒否 9。5 点それぞれの拒否と、`cnt:` 形式・役割の重複・codebook の `ordered` を含む）。検査器は `tools/contracts/ic02.py` の `check_calibration_parents`。

## 検査器（`tools/contracts`）での扱い

`blob.py`（S34・S24/S25・S36 の宣言・個数の負の値）、`ic02.py`（S35・S36 の位置・S37・S38、calibration の親の参照整合）、`ic02_window.py`（窓の成果物と `input_scope`、0.9.0）、`gate.py`（S39 と payload のスキーマ）。
Blob の見本は `../common/vectors/blob/blob-refs.json`、系列（T=1・T=2、δ なし・あり）の見本は `vectors/series.json`。
