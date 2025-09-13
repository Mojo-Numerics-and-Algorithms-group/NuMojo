# NuMojo

![logo](../assets/numojo_logo_360x360.png)

NuMojoは、Python の NumPy、SciPy と同様の数値計算機能を Mojo 🔥 で提供するライブラリです。

**[ドキュメントを見る»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md)**  |  **[変更履歴»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/changelog.md)**  |  **[Discordに参加»](https://discord.gg/NcnSH5n26F)**

**[中文·简»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_zhs.md)**  |  **[中文·繁»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_zht.md)**  |  **[English»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.md)** |  **[한국어»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_kr.md)**

**目次**

1. [プロジェクトについて](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#about-the-project)
2. [目標](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#goals)
3. [使用方法](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#usage)
4. [インストール方法](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#how-to-install)
5. [貢献について](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#contributing)
6. [注意事項](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#warnings)
7. [ライセンス](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#license)
8. [謝辞](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#acknowledgments)
9. [貢献者](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#Contributors)

## プロジェクトについて

NuMojoは、NumPy、SciPy、Scikit-learnなどのPythonパッケージにある幅広い数値計算機能の実現を目指しています。

***NuMojoとは***

私たちは、ベクトル化、並列化、GPU加速（利用可能になった場合）を含む、Mojoの潜在能力を最大限に活用することを目指しています。現在、NuMojoは、標準ライブラリの数学関数の（ほぼ）すべてを配列入力に対応するように拡張しています。

NuMojoのビジョンは、機械学習の逆伝播システムの追加的な負荷なしに、高速な数学演算を必要とする他のMojoパッケージにとって不可欠な構成要素として機能することです。

***NuMojoでないもの***

NuMojoは機械学習ライブラリではなく、ベースライブラリの一部として逆伝播を含むことはありません。

## 機能と目標

私たちの主な目的は、Mojoで高速で包括的な数値計算ライブラリを開発することです。以下に、いくつかの機能と長期的な目標を示します。一部はすでに（完全または部分的に）実装されています。

コアデータ型：

- ネイティブn次元配列（`numojo.NDArray`）
- ネイティブ2次元配列、つまり行列（`numojo.Matrix`）
- ネイティブn次元複素数配列（`numojo.ComplexNDArray`）
- ネイティブ固定次元配列（トレイトパラメータ化が利用可能になったときに実装予定）

ルーチンとオブジェクト：

- 配列作成ルーチン（`numojo.creation`）
- 配列操作ルーチン（`numojo.manipulation`）
- 入力と出力（`numojo.io`）
- 線形代数（`numojo.linalg`）
- 論理関数（`numojo.logic`）
- 数学関数（`numojo.math`）
- 指数と対数（`numojo.exponents`）
- 極値の発見（`numojo.extrema`）
- 丸め（`numojo.rounding`）
- 三角関数（`numojo.trig`）
- ランダムサンプリング（`numojo.random`）
- ソートと検索（`numojo.sorting`、`numojo.searching`）
- 統計（`numojo.statistics`）
- その他...

利用可能なすべての関数とオブジェクトは[こちら](docs/features.md)でご確認ください。最新のロードマップは[docs/roadmap.md](docs/roadmap.md)で管理されています。

詳細なロードマップについては、[docs/roadmap.md](docs/roadmap.md)ファイルを参照してください。

## 使用方法

n次元配列（`NDArray`型）の例は以下の通りです。

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # ランダムなfloat64値で2つの1000x1000行列を生成
    var A = nm.random.randn(Shape(1000, 1000))
    var B = nm.random.randn(Shape(1000, 1000))

    # 文字列表現から3x2行列を生成
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # 配列を出力
    print(A)

    # 配列の乗算
    var C = A @ B

    # 配列の逆行列
    var I = nm.inv(A)

    # 配列のスライス
    var A_slice = A[1:3, 4:19]

    # 配列からスカラーを取得
    var A_item = A[item(291, 141)]
    var A_item_2 = A.item(291, 141)
```

行列（`Matrix`型）の例は以下の通りです。

```mojo
from numojo import Matrix
from numojo.prelude import *


fn main() raises:
    # ランダムなfloat64値で2つの1000x1000行列を生成
    var A = Matrix.rand(shape=(1000, 1000))
    var B = Matrix.rand(shape=(1000, 1000))

    # ランダムなfloat64値で1000x1行列（列ベクトル）を生成
    var C = Matrix.rand(shape=(1000, 1))

    # 文字列表現から4x3行列を生成
    var F = Matrix.fromstring[i8](
        "[[12,11,10],[9,8,7],[6,5,4],[3,2,1]]", shape=(4, 3)
    )

    # 行列のスライス
    var A_slice = A[1:3, 4:19]
    var B_slice = B[255, 103:241:2]

    # 行列からスカラーを取得
    var A_item = A[291, 141]

    # 列ベクトルを反転
    print(C[::-1, :])

    # 軸に沿ってソートとargsort
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # 行列の合計
    print(nm.sum(B))
    print(nm.sum(B, axis=1))

    # 行列の乗算
    print(A @ B)

    # 行列の逆行列
    print(A.inv())

    # 線形代数の求解
    print(nm.solve(A, B))

    # 最小二乗法
    print(nm.lstsq(A, C))
```

`ComplexNDArray`の例は以下の通りです：

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # 複素数スカラー 5 + 5j を作成
    var complexscalar = ComplexSIMD[f32](re=5, im=5)
    # 複素数配列を作成
    var A = nm.full[f32](Shape(1000, 1000), fill_value=complexscalar)  # (5+5j)
    var B = nm.ones[f32](Shape(1000, 1000))                            # (1+1j)

    # 配列を出力
    print(A)

    # 配列のスライス
    var A_slice = A[1:3, 4:19]

    # 配列の乗算
    var C = A * B

    # 配列からスカラーを取得
    var A_item = A[item(291, 141)]
    # 配列の要素を設定
    A[item(291, 141)] = complexscalar
```## インストール方法

NuMojoパッケージをインストールして使用するには、3つのアプローチがあります。

### `pixi.toml`に`numojo`を追加

`pixi.toml`ファイルの依存関係セクションに、パッケージ`numojo`（再現性のため正確なバージョンに固定）を追加できます。

```toml
[dependencies]
numojo = "=0.7.0"
```

その後、`pixi install`を実行してパッケージをインストールします。

以下の表は、`numojo`のバージョンと必要な対応する`mojo`のバージョンを示しています。

| `numojo` | `mojo` |
| -------- | ------ |
| v0.7.0   | ==25.3 |
| v0.6.1   | ==25.2 |
| v0.6.0   | ==25.2 |

### パッケージをビルド

このアプローチでは、スタンドアロンパッケージファイル`numojo.mojopkg`をビルドし、他のプロジェクトにコピーできます（オフラインまたはhermetic buildに有用で、最新のNuMojoブランチを使用する場合に便利です）。

1. リポジトリをクローンします。
2. `pixi run package`を使用してパッケージをビルドします。
3. `numojo.mojopkg`をコードを含むディレクトリに移動します（またはその親ディレクトリをインクルードパスに追加します）。

### コンパイラとLSPにNuMojoのパスを含める

このアプローチでは、パッケージファイルをビルドする必要がありません。コンパイル時に、NuMojoソースパスを直接インクルードします：

```console
mojo run -I "../NuMojo" example.mojo
```

これは、コードをテストする際にNuMojoソースファイルを編集できるため、より柔軟です。

VSCodeのMojo LSPがインポートされた`numojo`パッケージを解決できるようにするには：

1. VSCodeの設定ページに移動します。
2. `Mojo › Lsp: Include Dirs`に移動します。
3. `add item`をクリックして、Numojoリポジトリが配置されているパスを書き込みます。例：`/Users/Name/Programs/NuMojo`
4. Mojo LSPサーバーを再起動します。

これで、VSCodeがNumojoパッケージの関数ヒントを表示できるようになりました！

## 貢献について

どのような貢献でも**大変感謝いたします**。ガイドライン（コーディングスタイル、テスト、ドキュメント、リリースサイクル）については、[CONTRIBUTING.md](CONTRIBUTING.md)をご覧ください。

## 注意事項

このライブラリはまだ初期段階にあり、マイナーバージョン間で破壊的変更が導入される可能性があります。本番環境や研究コードではバージョンを固定してください。

## ライセンス

LLVM例外付きApache 2.0ライセンスの下で配布されています。詳細については、[LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE)およびLLVM [License](https://llvm.org/LICENSE.txt)をご覧ください。

このプロジェクトには、Apache License v2.0 with LLVM Exceptions（LLVM [License](https://llvm.org/LICENSE.txt)を参照）でライセンスされた[Mojo Standard Library](https://github.com/modularml/mojo)からのコードが含まれています。MAXとMojoの使用と配布は、[MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license)の下でライセンスされています。

## 謝辞

[Modular](https://github.com/modularml)によって作成されたネイティブ[Mojo](https://github.com/modularml/mojo)で構築されています。

## 貢献者

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
mojo run -I "../NuMojo" example.mojo
```

これは、コードをテストするときにNuMojoソースファイルを編集できるので、より柔軟です。

VSCode LSPがインポートされた `numojo` パッケージを解決できるようにするには、次のようにします：

1. VSCodeの環境設定ページを開きます。
2. Mojo ' Lsp: Include Dirs` に移動します。
3. add item` をクリックし、Numojo リポジトリがあるパスを追加します。例えば `/Users/Name/Programs/NuMojo` です。
。
4. Mojo LSPサーバーを再起動します。

これでVSCodeがNumojoパッケージの関数ヒントを表示できるようになります！

## 貢献

どのような貢献でも大歓迎です**。コントリビュートに関する詳細やガイドラインは、[こちら](CONTRIBUTING.md)を参照してください。

## 警告

このライブラリはまだ非常に未完成であり、いつでも変更される可能性があります。

## ライセンス

LLVM例外を含むApache 2.0ライセンスの下で配布されています。詳細は[LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE)とLLVM [License](https://llvm.org/LICENSE.txt)を参照してください。

## 謝辞

* [Modular](https://github.com/modularml)によって作成されたネイティブの[Mojo](https://github.com/modularml/mojo)で構築されています。