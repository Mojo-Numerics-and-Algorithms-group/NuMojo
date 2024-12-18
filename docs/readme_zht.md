<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="../assets/numojo_logo.png" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">NuMojo</h1>

  <p align="center">
    NuMojo 是爲 Mojo 🔥 設計的多維數組運算庫，類似 Python 中的 NumPy, SciPy。
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md"><strong>閲讀文檔» </strong></a> &nbsp; &nbsp; 
        <a href="./changelog.md"><strong>更新日誌» </strong></a> &nbsp; &nbsp;
        <a href="https://discord.com/channels/1149778565756366939/1149778566603620455" ><strong>加入 Discord 討論頻道» </strong></a>
    </div>
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="./readme_zhs.md"><strong>中文·简» </strong></a>
        <a href="./readme_jp.md"><strong>日本語» </strong></a>
        <a href="../readme.md"><strong>English» </strong></a> &nbsp;
    </div>
  </p>
</div>

## 關於本項目

NuMojo，旨在為 Mojo 語言生態系統提供數值計算和多維數組運算功能，類似於 NumPy、SciPy 和 Scikit 等數學庫在 Python 語言生態系統中所扮演的角色。

NuMojo 充分利用 Mojo 的潛力，包括向量化、並行化和 GPU 加速。目前，NuMojo 擴展了大部分 Mojo 標準庫中的數學函數，用以處理多維數組。

NuMojo 也可為其他需要高速數值計算、多維數組運算等功能的 Mojo 第三方庫提供基礎類型和函數。

注意：NuMojo 不是一個機器學習庫，它永遠不會在核心庫中包含機器學習算法。

## 目标及路线图

有關本項目詳細的路線圖，請參閱 [roadmap.md](../roadmap.md) 文件（英文）。

我們的核心目標，是使用 Mojo 實現一個快速、全面的數值計算庫。以下是部分長期目標：

- 原生 N 維數組類型
- 向量化、並行化的數值運算
- 線性代數
- 數組操作：疊加、切片、拼合等
- 微積分
- 優化
- 函數逼近和估值
- 排序

## 使用方法

以下爲部分代碼實例：

```mojo
import numojo as nm

fn main() raises:
    # 生成兩個 1000x1000 矩陣，數值隨機且爲 64 位浮點數
    var A = nm.random.randn[f64](shape=List[Int](1000, 1000))
    var B = nm.random.randn[f64](shape=List[Int](1000, 1000))

    # 根據字符串生成 3x2 矩陣，数據類型爲 32 位浮點數
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # 打印矩陣
    print(A)

    # 矩陣相乘
    var C = A @ B

    # 矩陣求逆
    var I = nm.inv(A)

    # 矩陣切片
    var A_slice = A[1:3, 4:19]

    # 提取矩陣元素
    var A_item = A.at(291, 141)
```

請在 [此文檔](../features.md) 中查詢所有可用的函數。

## 安裝方法

Numojo 库可通过两種方法安裝並使用。

### 構建文件包

这種方法會構建一個獨立文件包 `numojo.mojopkg`。步骤爲：

1. 克隆本倉庫。
1. 在控制臺使用 `mojo package numojo` 命令構建文件包。
1. 將 `numojo.mojopkg` 移動到包含代碼的目録中，即可使用。

### 將 NuMojo 路徑添加至編譯器和 LSP

這種方法不需要生成文件包，僅需在編譯時，通這以下命令指明 `Numojo` 的文件路徑：

```console
mojo run -I "../NuMojo" example.mojo
```

這種方法自由度更高，允許你在調試你代碼的過程中修改 `NuMojo` 源碼。它適合想要爲本庫贡獻代碼的用户。

爲了使 VSCode 的 LSP （語言服務引擎）解析 `numojo` 庫，你可以：

1. 進入 VSCode 的 Preference 頁面（偏好設置）。
1. 選擇 `Mojo › Lsp: Include Dirs`
1. 點擊 `add item`，並添加 NuMojo 源碼所在的路徑，例如 `/Users/Name/Programs/NuMojo`
1. 重啓 Mojo LSP server

如此，VSCode 便可以提供 NuMojo 包的函數提示。
