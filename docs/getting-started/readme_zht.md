<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="../../assets/numojo_logo.png" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">NuMojo</h1>

  <p align="center">
    NuMojo 是爲 Mojo 🔥 設計的多維數組運算庫，類似 Python 中的 NumPy, SciPy。
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="https://numojo.readthedocs.io"><strong>閲讀文檔» </strong></a> &nbsp; &nbsp;
        <a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md"><strong>查看範例» </strong></a> &nbsp; &nbsp;
        <a href="../user-guide/changelog.md"><strong>更新日誌» </strong></a> &nbsp; &nbsp;
        <a href="https://discord.gg/NcnSH5n26F" ><strong>加入 Discord 討論頻道» </strong></a>
    </div>
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="./readme_zhs.md"><strong>中文·简» </strong></a>
        <a href="./readme_jp.md"><strong>日本語» </strong></a>
        <a href="./readme_kr.md"><strong>한국어» </strong></a>
        <a href="../../README.MD"><strong>English» </strong></a> &nbsp;
    </div>
  </p>
</div>

## 關於本項目

NuMojo，旨在為 Mojo 語言生態系統提供數值計算和多維數組運算功能，類似於 NumPy、SciPy 和 Scikit 等數學庫在 Python 語言生態系統中所扮演的角色。

***NuMojo 是什麼***

我們致力於充分發揮 Mojo 的潛力，包括向量化、並行化和 GPU 加速。目前，NuMojo 已擴展了大部分（甚至全部）Mojo 標準庫中的數學函數，使其支援數組輸入。

我們對 NuMojo 的願景，是讓它成為其他需要高速數值運算的 Mojo 第三方庫的基礎構件，同時不帶有機器學習反向傳播系統這類額外負擔。

***NuMojo 不是什麼***

NuMojo 不是一個機器學習庫，它永遠不會在核心庫中包含反向傳播（back-propagation）功能。

## 特性與目標

我們的核心目標，是使用 Mojo 打造一個快速、全面的數值計算庫。以下列出部分特性與長期目標，其中一些已經完整或部分實現。

核心資料型別：

- 原生 N 維數組（`numojo.NDArray`）。
- 原生 N 維複數數組（`numojo.ComplexNDArray`）
- 原生固定維度數組（待 Mojo 支援 trait 參數化後實現）。

程式庫與物件：

- 數組建立相關函數（`numojo.creation`）
- 數組操作相關函數（`numojo.manipulation`）
- 輸入輸出（`numojo.io`）
- 線性代數（`numojo.linalg`）
- 邏輯函數（`numojo.logic`）
- 數學函數（`numojo.math`）
- 指數與對數（`numojo.exponents`）
- 極值查找（`numojo.extrema`）
- 四捨五入（`numojo.rounding`）
- 三角函數（`numojo.trig`）
- 隨機取樣（`numojo.random`）
- 排序與搜尋（`numojo.sorting`、`numojo.searching`）
- 統計（`numojo.statistics`）
- 等等……

所有可用的函數與物件，請參閱[此文檔](../user-guide/features.md)。我們也維護了一份持續更新的路線圖，詳見 [docs/user-guide/roadmap.md](../user-guide/roadmap.md)。

詳細的路線圖，請參閱 [docs/user-guide/roadmap.md](../user-guide/roadmap.md) 文件。

## 使用方法

`examples/` 目錄下提供了可直接執行的範例（例如 `examples/quickstart.mojo`）。

n 維數組（`NDArray` 類型）的範例如下：

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # Generate two 1000x1000 matrices with random float64 values
    var A = nm.random.randn(Shape(1000, 1000)) # Shape is used for all shape related operations in numojo. 
    var B = nm.random.randn(Shape(1000, 1000))

    # Generate a 3x2 matrix from string representation
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # Print array
    print(A)

    # Array multiplication
    var C = A @ B

    # Array inversion
    var I = nm.inv(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Get scalar from array
    var A_item = A[Item(291, 141)] # Item() is used to define coordinates of an ndarray in numojo. 
    var A_item_2 = A.item(291, 141)

    # Sort and argsort along axis
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # Sum along axis
    print(nm.sum(A))
    print(nm.sum(A, axis=1))

    # Solve a linear system
    print(nm.solve(A, B))
```

`ComplexNDArray` 的範例如下：

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # Create a complex scalar 5 + 5j
    # cf32 is the complex version of f32 (DType.float32) used to identify complex types in numojo.
    var complexscalar = CScalar[cf32](5) # Equivalently ComplexSIMD[cf32](5, 5)
    # Also can be define as simple as  5 + 5*`1j`!
  
    # Create complex arrays
    var A = nm.full[cf32](Shape(1000, 1000), fill_value=complexscalar)  # filled with (5+5j)
    var B = nm.ones[cf32](Shape(1000, 1000))                            # filled with (1+1j)

    # Print array
    print(A)

    # Array slicing
    var A_slice = A[1:3, 4:19]

    # Array multiplication
    var C = A * B

    # Get scalar from array
    var A_item = A[Item(291, 141)]
    # Set an element of the array
    A[item(291, 141)] = complexscalar
```

## 安裝方法

NuMojo 提供多種安裝方式，以配合不同的開發需求。請選擇最符合你工作流程的方法：

### 方法一：透過 pixi-build-mojo 進行 Git 安裝（推薦）

直接從 GitHub 倉庫安裝 NuMojo，同時取得穩定版本與最新功能。此方法最適合想使用最新功能，或需要最新穩定版本的開發者。

在現有的 `pixi.toml` 中加入以下內容：

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project_name"
version = "0.1.0"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[package.build.config.pkg]
name = "your_package_name"

[package.host-dependencies]
modular = "0.26.2.*"

[package.build-dependencies]
modular = "0.26.2.*"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[package.run-dependencies]
modular = "0.26.2.*"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[dependencies]
modular = "26.2.*"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}
```

接著執行：

```bash
pixi install
```

**分支選擇：**

- **`main` 分支**：提供穩定版本，目前支援 NuMojo v0.8.0，相容於 Mojo 25.6.0。若需使用更早期的 NuMojo 版本，請改用方法二。
- **`pre-x.y` 分支**：積極開發中的分支，支援最新的 Mojo 版本（目前為 mojo nightly 26.2.0.dev2026022717）。請注意，此分支更新頻繁，功能與語法可能出現不相容的變動。

安裝完成後，該套件會自動出現在你的 Pixi 環境中，VSCode LSP 也會提供智能程式碼提示。

### 方法二：透過 Pixi（prefix.dev）安裝穩定版本

對大多數使用者而言，我們建議透過 Pixi 安裝穩定版本，以確保相容性與可重現性。

在 `pixi.toml` 檔案中加入以下內容：

```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]

[dependencies]
numojo = "=0.9.0"
```

接著執行：

```bash
pixi install
```

**版本相容表：**

| NuMojo 版本 | 所需 Mojo 版本 |
| ----------- | --------------- |
| v0.9.0      | ==26.2          |
| v0.8.0      | ==25.7          |
| v0.7.0      | ==25.3          |
| v0.6.1      | ==25.2          |
| v0.6.0      | ==25.2          |

### 方法三：構建獨立文件包

此方法會構建一個可攜式的 `numojo.mojopkg` 文件包，適合跨多個項目使用，或用於離線開發、封閉式構建。

1. 克隆本倉庫：

   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   cd NuMojo
   ```

2. 構建文件包：

   ```bash
   pixi run package
   ```

3. 將 `numojo.mojopkg` 複製到你的項目目錄，或將其所在目錄加入編譯器的引用路徑。

### 方法四：直接引用原始碼

若希望獲得最大彈性，並能在開發過程中直接修改 NuMojo 原始碼：

1. 將本倉庫克隆至你所需的位置：

   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   ```

2. 編譯代碼時，指明 NuMojo 原始碼所在路徑：

   ```bash
   mojo run -I "/path/to/NuMojo" your_program.mojo
   ```

3. **設置 VSCode LSP**（用於程式碼提示與自動補全）：
   - 打開 VSCode 偏好設置
   - 進入 `Mojo › Lsp: Include Dirs`
   - 點擊 `Add Item`，輸入 NuMojo 目錄的完整路徑（例如 `/Users/YourName/Projects/NuMojo`）
   - 重啓 Mojo LSP server

設置完成後，VSCode 便會為 NuMojo 函數提供智能程式碼補全與提示！

## 貢獻方式

我們非常感謝任何形式的貢獻。貢獻指南（程式碼風格、測試、文檔撰寫、發佈週期）請參閱 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## 警告

本庫目前仍處於早期階段，各次要版本之間可能存在不相容的變動。若用於生產環境或研究專案，請固定版本號。

## 授權條款

本項目依 Apache 2.0 License（附 LLVM Exceptions）發佈。詳情請參閱 [LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE) 與 LLVM [License](https://llvm.org/LICENSE.txt)。

本項目包含來自 [Mojo 標準庫](https://github.com/modularml/mojo) 的代碼，依 Apache License v2.0（附 LLVM Exceptions）授權（詳見 LLVM [License](https://llvm.org/LICENSE.txt)）。MAX 與 Mojo 的使用及散佈，依 [MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license) 授權。

## 致謝

本項目使用原生 [Mojo](https://github.com/modularml/mojo) 開發，Mojo 由 [Modular](https://github.com/modularml) 創建。

## 貢獻者

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
