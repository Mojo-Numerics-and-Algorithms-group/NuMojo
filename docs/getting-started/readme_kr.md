# NuMojo

![logo](../../assets/numojo_logo_360x360.png)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Mojo](https://img.shields.io/badge/mojo-%3E%3D1.0.0-orange.svg)](https://www.modular.com/mojo)

NuMojo는 Python의 NumPy와 유사한 Mojo 🔥 수치 계산 라이브러리입니다.

**[예제 살펴보기»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md)**  |  **[변경 로그»](../user-guide/changelog.md)**  |  **[Discord 참여하기»](https://discord.gg/NcnSH5n26F)**

**[中文·简»](readme_zhs.md)**  |  **[中文·繁»](readme_zht.md)**  |  **[日本語»](readme_jp.md)** | **[English»](../../README.MD)**

**목차**

1. [프로젝트 소개](#프로젝트-소개)
2. [왜 NuMojo인가](#왜-numojo인가)
3. [기능과 목표](#기능과-목표)
4. [사용법](#사용법)
5. [설치 방법](#설치-방법)
6. [기여하기](#기여하기)
7. [주의사항](#주의사항)
8. [라이센스](#라이센스)
9. [감사의 글](#감사의-글)
10. [기여자](#기여자)

## 프로젝트 소개

NuMojo는 NumPy에서 볼 수 있는 광범위한 수치 계산 기능을 포괄하는 것을 목표로 합니다.

우리는 벡터화, 병렬화, GPU 가속을 포함하여 Mojo의 모든 잠재력을 활용하고자 합니다. 현재 NuMojo는 표준 라이브러리 수학 함수의 (거의) 모든 기능을 배열 입력을 지원하도록 확장했습니다.

NuMojo의 비전은 기계 학습 역전파 시스템의 추가적인 부담 없이 빠른 수학 연산이 필요한 다른 Mojo 패키지들의 필수적인 구성 요소로 역할하는 것입니다.

## 왜 NuMojo인가

- **Mojo 네이티브.** NuMojo의 `NDArray`는 NumPy나 MAX의 텐서 타입을 감싼 바인딩이 아니라 Mojo 네이티브의 SIMD 기반 타입이므로, Python 상호운용에 따른 오버헤드 없이 그대로 프로그램에 컴파일됩니다.
- **NumPy에 익숙한 API.** 슬라이싱, 브로드캐스팅, 행렬 곱셈을 위한 `@` 연산자, 그리고 함수 이름까지 합리적인 범위 내에서 NumPy를 따르므로 기존에 익힌 직관을 그대로 활용할 수 있습니다.
- **Mojo의 강점을 살린 설계.** 모든 루틴 전반에 걸쳐 벡터화와 병렬화를 활용하며, Mojo 자체의 디바이스 지원이 성숙해짐에 따라 GPU와 기타 가속기 지원(`AcceleratorNDArray`)도 순차적으로 추가될 예정입니다.

## 기능과 목표

우리의 주요 목적은 Mojo에서 빠르고 포괄적인 수치 계산 라이브러리를 개발하는 것입니다. 아래는 일부 기능과 장기적인 목표입니다. 일부는 이미 (완전히 또는 부분적으로) 구현되었습니다.

핵심 데이터 타입:

- 네이티브 n차원 배열 (`numojo.NDArray`).
- 네이티브 n차원 복소수 배열 (`numojo.ComplexNDArray`)
- 네이티브 고정 차원 배열 (트레이트 매개변수화가 가능해지면 구현 예정).

루틴과 객체:

- 배열 생성 루틴 (`numojo.creation`)
- 배열 조작 루틴 (`numojo.manipulation`)
- 입력과 출력 (`numojo.io`)
- 선형 대수 (`numojo.linalg`)
- 논리 함수 (`numojo.logic`)
- 수학 함수 (`numojo.math`)
- 지수와 로그 (`numojo.exponents`)
- 극값 찾기 (`numojo.extrema`)
- 반올림 (`numojo.rounding`)
- 삼각 함수 (`numojo.trig`)
- 랜덤 샘플링 (`numojo.random`)
- 정렬과 검색 (`numojo.sorting`, `numojo.searching`)
- 통계 (`numojo.statistics`)
- 기타...

사용 가능한 모든 함수와 객체는 [여기](../user-guide/features.md)에서 확인하세요. 최신 로드맵은 [docs/user-guide/roadmap.md](../user-guide/roadmap.md)에서 관리됩니다.

## 사용법

`examples/` 디렉터리(예: `examples/quickstart.mojo`)에서 바로 실행 가능한 예제를 확인할 수 있습니다.

n차원 배열(`NDArray` 타입)의 예시는 다음과 같습니다.

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # 랜덤한 float64 값으로 두 개의 1000x1000 행렬 생성
    var A = nm.random.randn(Shape(1000, 1000)) # Shape은 numojo에서 형태(shape)와 관련된 모든 연산에 사용됩니다. 
    var B = nm.random.randn(Shape(1000, 1000))

    # 문자열 표현으로부터 3x2 행렬 생성
    var X = nm.fromstring[f32]("[[1.1, -0.32, 1], [0.1, -3, 2.124]]")

    # 배열 출력
    print(A)

    # 배열 곱셈
    var C = A @ B

    # 배열 역행렬
    var I = nm.inv(A)

    # 배열 슬라이싱
    var A_slice = A[1:3, 4:19]

    # 배열에서 스칼라 가져오기
    var A_item = A[Item(291, 141)] # Item()은 numojo에서 ndarray의 좌표를 정의하는 데 사용됩니다. 
    var A_item_2 = A.item(291, 141)

    # 축을 따른 정렬과 argsort
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # 축을 따른 합계
    print(nm.sum(A))
    print(nm.sum(A, axis=1))

    # 선형 시스템 풀이
    print(nm.solve(A, B))
```

`ComplexNDArray`의 예시는 다음과 같습니다:

```mojo
import numojo as nm
from numojo.prelude import *


def main() raises:
    # 복소수 스칼라 5 + 5j 생성
    # cf32는 numojo에서 복소수 타입을 식별하는 데 사용되는 f32(DType.float32)의 복소수 버전입니다.
    var complexscalar = CScalar[cf32](5) # ComplexSIMD[cf32](5, 5)와 동일합니다
    # 5 + 5*`1j`처럼 간단하게 정의할 수도 있습니다!
  
    # 복소수 배열 생성
    var A = nm.full[cf32](Shape(1000, 1000), fill_value=complexscalar)  # (5+5j)로 채움
    var B = nm.ones[cf32](Shape(1000, 1000))                            # (1+1j)로 채움

    # 배열 출력
    print(A)

    # 배열 슬라이싱
    var A_slice = A[1:3, 4:19]

    # 배열 곱셈
    var C = A * B

    # 배열에서 스칼라 가져오기
    var A_item = A[Item(291, 141)]
    # 배열의 요소 설정
    A[item(291, 141)] = complexscalar
```

## 설치 방법

NuMojo는 다양한 개발 요구에 맞춰 여러 가지 설치 방법을 제공합니다. 자신의 작업 방식에 가장 적합한 방법을 선택하세요.

### 방법 1: pixi-build-mojo를 이용한 Git 설치 (권장)

GitHub 저장소에서 직접 NuMojo를 설치하여 안정적인 릴리스와 최신 기능을 모두 사용할 수 있습니다. 이 방법은 최신 기능을 원하거나 가장 최근의 안정 버전으로 작업해야 하는 개발자에게 적합합니다.

기존 `pixi.toml`에 다음 내용을 추가하세요.

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
mojo = "==1.0.0"
max-core = "==26.5.0"

[package.build-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[package.run-dependencies]
mojo = "==1.0.0"
max-core = "==26.5.0"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}

[dependencies]
mojo = ">=1.0.0, <1.1.0"
max-core = ">=26.5.0,<27"
numojo = { git = "https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo", branch = "main"}
```

그런 다음 다음을 실행합니다.
```bash
pixi install
```

**브랜치 선택:**
- **`main` 브랜치**: 최신 안정 릴리스를 제공합니다. 현재 NuMojo v0.10.0이며 Mojo >=1.0.0, <1.1.0과 호환됩니다. 이전 버전의 NuMojo가 필요하다면 방법 2를 사용하세요.
- **`pre-x.y` 브랜치**: 다음 릴리스를 위한 활발한 개발 브랜치입니다. 이 브랜치는 자주 업데이트되며 기능과 문법에 호환성을 깨는 변경사항이 있을 수 있습니다.

패키지는 Pixi 환경에 자동으로 반영되며, VSCode LSP가 지능형 코드 힌트를 제공합니다.

### 방법 2: Pixi(prefix.dev)를 통한 안정 릴리스 설치

대부분의 사용자에게는 안정적인 호환성과 재현성을 보장하는 Pixi를 통한 안정 릴리스 설치를 권장합니다.

`pixi.toml` 파일에 다음 내용을 추가하세요.

```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]

[dependencies]
numojo = "=0.10.0"
```

그런 다음 다음을 실행합니다.
```bash
pixi install
```

**버전 호환성:**

| NuMojo 버전 | 필요한 Mojo 버전 |
| ----------- | ----------------- |
| v0.10.0     | ==1.0.0            |
| v0.9.0      | ==26.2             |
| v0.8.0      | ==25.7             |
| v0.7.0      | ==25.3             |
| v0.6.1      | ==25.2             |
| v0.6.0      | ==25.2             |

### 방법 3: 독립형 패키지 빌드

이 방법은 여러 프로젝트에서 사용할 수 있는 이식 가능한 `numojo.mojopkg` 파일을 생성합니다. 오프라인 개발이나 밀폐된(hermetic) 빌드에 적합합니다.

1. 저장소를 클론합니다.
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   cd NuMojo
   ```

2. 패키지를 빌드합니다.
   ```bash
   pixi run package
   ```

3. `numojo.mojopkg`를 프로젝트 디렉터리로 복사하거나, 그 상위 디렉터리를 include 경로에 추가합니다.

### 방법 4: 소스 코드 직접 통합

개발 중 NuMojo 소스 코드를 직접 수정할 수 있는 최대한의 유연성을 원한다면 다음과 같이 진행하세요.

1. 원하는 위치에 저장소를 클론합니다.
   ```bash
   git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
   ```

2. 코드를 컴파일할 때 NuMojo 소스 경로를 포함합니다.
   ```bash
   mojo run -I "/path/to/NuMojo" your_program.mojo
   ```

3. **VSCode LSP 설정** (코드 힌트와 자동 완성을 위해):
   - VSCode 환경설정을 엽니다
   - `Mojo › Lsp: Include Dirs`로 이동합니다
   - `Add Item`을 클릭하고 NuMojo 디렉터리의 전체 경로를 입력합니다 (예: `/Users/YourName/Projects/NuMojo`)
   - Mojo LSP 서버를 재시작합니다

설정을 마치면 VSCode가 NuMojo 함수에 대한 지능형 코드 완성과 힌트를 제공합니다!

## 기여하기

여러분의 모든 기여를 **진심으로 감사드립니다**. NuMojo는 아직 초기 단계이며 루틴 구현, 테스트 작성, 문서 개선, 또는 어색한 부분에 대한 이슈 제기 등 도울 수 있는 부분이 많습니다.

기여자를 위한 빠른 시작 방법은 다음과 같습니다.

```bash
git clone https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo.git
cd NuMojo
pixi install
pixi run final   # 포맷팅 + 테스트
```

코딩 스타일, 디렉터리 구조, PR 절차 등 전체 가이드라인은 [docs/developer-guide/contributing.md](../developer-guide/contributing.md)를, 독스트링/포맷팅 규칙은 [docs/developer-guide/style-guide.md](../developer-guide/style-guide.md)를, PR을 올리기 전에 확인해야 할 사항은 [docs/developer-guide/pre-pr-checks.md](../developer-guide/pre-pr-checks.md)를 참조하세요.

## 주의사항

이 라이브러리는 아직 초기 단계이며 마이너 버전 간에 호환성을 깨는 변경사항이 도입될 수 있습니다. 프로덕션이나 연구 코드에서는 버전을 고정하세요.

## 라이센스

LLVM 예외가 포함된 Apache 2.0 라이센스 하에 배포됩니다. 자세한 정보는 [LICENSE](../../LICENSE)와 LLVM [License](https://llvm.org/LICENSE.txt)를 참조하세요.

이 프로젝트는 Apache License v2.0 with LLVM Exceptions로 라이센스된 [Mojo Standard Library](https://github.com/modularml/mojo)의 코드를 포함합니다 (LLVM [License](https://llvm.org/LICENSE.txt) 참조). MAX와 Mojo 사용 및 배포는 [MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license) 하에 라이센스됩니다.

## 감사의 글

[Modular](https://github.com/modularml)에서 만든 네이티브 [Mojo](https://github.com/modularml/mojo)로 구축되었습니다.

## 기여자

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>
