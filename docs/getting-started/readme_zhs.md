<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="../../assets/numojo_logo.png" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">NuMojo</h1>

  <p align="center">
    NuMojo 是为 Mojo 🔥 设计的多维数组运算库，类似 Python 中的 NumPy, SciPy。
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md"><strong>阅读文档» </strong></a> &nbsp; &nbsp; 
        <a href="../user-guide/changelog.md"><strong>更新日志» </strong></a> &nbsp; &nbsp;
        <a href="https://discord.gg/NcnSH5n26F" ><strong>加入 Discord 讨论频道» </strong></a>
    </div>
    <br />
    <div style="font-family: 'Arial'; border: 1px solid black; padding: 5px;">
        <a href="./readme_zht.md"><strong>中文·繁» </strong></a> &nbsp;
        <a href="./readme_jp.md"><strong>日本語» </strong></a>
        <a href="../../README.MD"><strong>English» </strong></a> &nbsp;
    </div>
  </p>
</div>

## 关于本项目

NuMojo，旨在为 Mojo 语言生态系统提供数值计算和多维数组运算功能，类似于 NumPy、SciPy 和 Scikit 等数学库在 Python 语言生态系统中所扮演的角色。

NuMojo 充分利用 Mojo 的潜力，包括向量化、并行化和 GPU 加速。目前，NuMojo 扩展了大部分 Mojo 标准库中的数学函数，用以处理多维数组。

NuMojo 也可为其他需要高速数值计算、多维数组运算等功能的 Mojo 第三方库提供基础类型和函数。

注意：NuMojo 不是一个机器学习库，它永远不会在核心库中包含机器学习算法。

## 目标及路线图

有关本项目详细的路线图，请参阅 [roadmap.md](../user-guide/roadmap.md) 文件（英文）。

我们的核心目标，是使用 Mojo 实现一个快速、全面的数值计算库。以下是部分长期目标：

- 原生 N 维数组类型
- 向量化、并行化的数值运算
- 线性代数
- 数组操作：叠加、切片、拼合等
- 微积分
- 优化
- 函数逼近和估值
- 排序

## 使用方法

n维数组（`NDArray` 类型）的示例如下：

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # 生成两个 1000x1000 矩阵，使用随机 float64 值
    var A = nm.random.randn(Shape(1000, 1000))
    var B = nm.random.randn(Shape(1000, 1000))

    # 从字符串表示生成 3x2 矩阵
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # 打印数组
    print(A)

    # 数组乘法
    var C = A @ B

    # 数组求逆
    var I = nm.inv(A)

    # 数组切片
    var A_slice = A[1:3, 4:19]
    var B_slice = B[255, 103:241:2]

    # 提取矩阵元素
    var A_item = A.item(291, 141)
    var A_item_2 = A.item(291, 141)
```

矩阵（`Matrix` 类型）的示例如下：

```mojo
from numojo import Matrix
from numojo.prelude import *


fn main() raises:
    # 生成两个 1000x1000 矩阵，使用随机 float64 值
    var A = Matrix.rand(shape=(1000, 1000))
    var B = Matrix.rand(shape=(1000, 1000))

    # 生成 1000x1 矩阵（列向量），使用随机 float64 值
    var C = Matrix.rand(shape=(1000, 1))

    # 从字符串表示生成 4x3 矩阵
    var F = Matrix.fromstring[i8](
        "[[12,11,10],[9,8,7],[6,5,4],[3,2,1]]", shape=(4, 3)
    )

    # 矩阵切片
    var A_slice = A[1:3, 4:19]
    var B_slice = B[255, 103:241:2]

    # 从矩阵获取标量
    var A_item = A[291, 141]

    # 翻转列向量
    print(C[::-1, :])

    # 沿轴排序和 argsort
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # 矩阵求和
    print(nm.sum(B))
    print(nm.sum(B, axis=1))

    # 矩阵乘法
    print(A @ B)

    # 矩阵求逆
    print(A.inv())

    # 求解线性代数方程
    print(nm.solve(A, B))

    # 最小二乘法
    print(nm.lstsq(A, C))
```

`ComplexNDArray` 的示例如下：

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # 创建复数标量 5 + 5j
    var complexscalar = ComplexSIMD[f32](re=5, im=5)
    # 创建复数数组
    var A = nm.full[f32](Shape(1000, 1000), fill_value=complexscalar)  # (5+5j)
    var B = nm.ones[f32](Shape(1000, 1000))                            # (1+1j)

    # 打印数组
    print(A)

    # 数组切片
    var A_slice = A[1:3, 4:19]

    # 数组乘法
    var C = A * B

    # 从数组获取标量
    var A_item = A[item(291, 141)]
    # 设置数组元素
    A[item(291, 141)] = complexscalar
```

请在 [此文档](../user-guide/features.md) 中查询所有可用的函数。

## 安装方法

Numojo 库可通过两种方法安装并使用。

### 构建文件包

这种方法会构建一个独立文件包 `numojo.mojopkg`。步骤为：

1. 克隆本仓库。
1. 在控制台使用 `mojo package numojo` 命令构建文件包。
1. 将 `numojo.mojopkg` 移动到包含代码的目录中，即可使用。

### 将 NuMojo 路径添加至编译器和 LSP

这种方法不需要生成文件包，仅需在编译时，通过以下命令指明 `Numojo` 的文件路径：

```console
mojo run -I "../NuMojo" example.mojo
```

这种方法自由度更高，允许你在调试你代码的过程中修改 `NuMojo` 源码。它适合想要为本库贡献代码的用户。

为了使 VSCode 的 LSP （语言服务引擎）解析 `numojo` 库，你可以：

1. 进入 VSCode 的 Preference 页面（偏好设置）。
1. 选择 `Mojo › Lsp: Include Dirs`
1. 点击 `add item`，并添加 NuMojo 源码所在的路径，例如 `/Users/Name/Programs/NuMojo`
1. 重启 Mojo LSP server

如此，VSCode 便可以提供 NuMojo 包的函数提示。
