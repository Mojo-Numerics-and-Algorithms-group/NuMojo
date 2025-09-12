# NuMojo

![logo](../assets/numojo_logo_360x360.png)

NuMojo는 Python의 NumPy, SciPy와 유사한 Mojo 🔥 수치 계산 라이브러리입니다.

**[문서 살펴보기»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo-Examples-and-Benchmarks/blob/main/docs/README.md)**  |  **[변경 로그»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/changelog.md)**  |  **[Discord 참여하기»](https://discord.gg/NcnSH5n26F)**

**[中文·简»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_zhs.md)**  |  **[中文·繁»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_zht.md)**  |  **[日本語»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/docs/readme_jp.md)** | **[English»](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.md)**

**목차**

1. [프로젝트 소개](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#about-the-project)
2. [목표](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#goals)
3. [사용법](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#usage)
4. [설치 방법](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#how-to-install)
5. [기여하기](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#contributing)
6. [주의사항](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#warnings)
7. [라이센스](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#license)
8. [감사의 글](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#acknowledgments)
9. [기여자](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/README.MD#Contributors)

## 프로젝트 소개

NuMojo는 NumPy, SciPy, Scikit-learn과 같은 Python 패키지에서 볼 수 있는 광범위한 수치 계산 기능을 포괄하는 것을 목표로 합니다.

***NuMojo란 무엇인가***

우리는 벡터화, 병렬화, GPU 가속(사용 가능할 때)을 포함하여 Mojo의 모든 잠재력을 활용하고자 합니다. 현재 NuMojo는 표준 라이브러리 수학 함수의 (거의) 모든 기능을 배열 입력을 지원하도록 확장했습니다.

NuMojo의 비전은 기계 학습 역전파 시스템의 추가적인 부담 없이 빠른 수학 연산이 필요한 다른 Mojo 패키지들의 필수적인 구성 요소로 역할하는 것입니다.

***NuMojo가 아닌 것***

NuMojo는 기계 학습 라이브러리가 아니며 기본 라이브러리의 일부로 역전파를 포함하지 않을 것입니다.

## 기능과 목표

우리의 주요 목적은 Mojo에서 빠르고 포괄적인 수치 계산 라이브러리를 개발하는 것입니다. 아래는 일부 기능과 장기적인 목표입니다. 일부는 이미 (완전히 또는 부분적으로) 구현되었습니다.

핵심 데이터 타입:

- 네이티브 n차원 배열 (`numojo.NDArray`)
- 네이티브 2차원 배열, 즉 행렬 (`numojo.Matrix`)
- 네이티브 n차원 복소수 배열 (`numojo.ComplexNDArray`)
- 네이티브 고정 차원 배열 (트레이트 매개변수화가 가능해지면 구현 예정)

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

사용 가능한 모든 함수와 객체는 [여기](docs/features.md)에서 확인하세요. 최신 로드맵은 [docs/roadmap.md](docs/roadmap.md)에서 관리됩니다.

자세한 로드맵은 [docs/roadmap.md](docs/roadmap.md) 파일을 참조하세요.

## 사용법

n차원 배열(`NDArray` 타입)의 예시는 다음과 같습니다.

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # 랜덤한 float64 값으로 두 개의 1000x1000 행렬 생성
    var A = nm.random.randn(Shape(1000, 1000))
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
    var A_item = A[item(291, 141)]
    var A_item_2 = A.item(291, 141)
```

행렬(`Matrix` 타입)의 예시는 다음과 같습니다.

```mojo
from numojo import Matrix
from numojo.prelude import *


fn main() raises:
    # 랜덤한 float64 값으로 두 개의 1000x1000 행렬 생성
    var A = Matrix.rand(shape=(1000, 1000))
    var B = Matrix.rand(shape=(1000, 1000))

    # 랜덤한 float64 값으로 1000x1 행렬(열 벡터) 생성
    var C = Matrix.rand(shape=(1000, 1))

    # 문자열 표현으로부터 4x3 행렬 생성
    var F = Matrix.fromstring[i8](
        "[[12,11,10],[9,8,7],[6,5,4],[3,2,1]]", shape=(4, 3)
    )

    # 행렬 슬라이싱
    var A_slice = A[1:3, 4:19]
    var B_slice = B[255, 103:241:2]

    # 행렬에서 스칼라 가져오기
    var A_item = A[291, 141]

    # 열 벡터 뒤집기
    print(C[::-1, :])

    # 축을 따른 정렬과 argsort
    print(nm.sort(A, axis=1))
    print(nm.argsort(A, axis=0))

    # 행렬 합계
    print(nm.sum(B))
    print(nm.sum(B, axis=1))

    # 행렬 곱셈
    print(A @ B)

    # 행렬 역행렬
    print(A.inv())

    # 선형 대수 풀이
    print(nm.solve(A, B))

    # 최소 제곱법
    print(nm.lstsq(A, C))
```

`ComplexNDArray`의 예시는 다음과 같습니다:

```mojo
import numojo as nm
from numojo.prelude import *


fn main() raises:
    # 복소수 스칼라 5 + 5j 생성
    var complexscalar = ComplexSIMD[f32](re=5, im=5)
    # 복소수 배열 생성
    var A = nm.full[f32](Shape(1000, 1000), fill_value=complexscalar)  # (5+5j)
    var B = nm.ones[f32](Shape(1000, 1000))                            # (1+1j)

    # 배열 출력
    print(A)

    # 배열 슬라이싱
    var A_slice = A[1:3, 4:19]

    # 배열 곱셈
    var C = A * B

    # 배열에서 스칼라 가져오기
    var A_item = A[item(291, 141)]
    # 배열의 요소 설정
    A[item(291, 141)] = complexscalar
```

## 설치 방법

NuMojo 패키지를 설치하고 사용하는 세 가지 방법이 있습니다.

### `pixi.toml`에 `numojo` 추가

`pixi.toml` 파일의 의존성 섹션에 패키지 `numojo`를 추가할 수 있습니다 (재현성을 위해 정확한 버전으로 고정).

```toml
[dependencies]
numojo = "=0.7.0"
```

그런 다음 `pixi install`을 실행하여 패키지를 설치합니다.

다음 표는 `numojo` 버전과 필요한 해당 `mojo` 버전을 보여줍니다.

| `numojo` | `mojo` |
| -------- | ------ |
| v0.7.0   | ==25.3 |
| v0.6.1   | ==25.2 |
| v0.6.0   | ==25.2 |

### 패키지 빌드

이 방법은 다른 프로젝트에 복사할 수 있는 독립형 패키지 파일 `numojo.mojopkg`를 빌드합니다 (오프라인 또는 밀폐된 빌드에 유용하며 최신 NuMojo 브랜치를 사용하는 데 편리합니다).

1. 저장소를 클론합니다.
2. `pixi run package`를 사용하여 패키지를 빌드합니다.
3. `numojo.mojopkg`를 코드가 포함된 디렉터리로 이동합니다 (또는 부모 디렉터리를 포함 경로에 추가합니다).

### 컴파일러와 LSP에 NuMojo 경로 포함

이 방법은 패키지 파일을 빌드할 필요가 없습니다. 컴파일할 때 NuMojo 소스 경로를 직접 포함합니다:

```console
mojo run -I "../NuMojo" example.mojo
```

이는 코드를 테스트할 때 NuMojo 소스 파일을 편집할 수 있어 더 유연합니다.

VSCode의 Mojo LSP가 가져온 `numojo` 패키지를 해결할 수 있도록 하려면:

1. VSCode의 설정 페이지로 이동합니다.
2. `Mojo › Lsp: Include Dirs`로 이동합니다.
3. `add item`을 클릭하고 Numojo 저장소가 위치한 경로를 작성합니다. 예: `/Users/Name/Programs/NuMojo`
4. Mojo LSP 서버를 재시작합니다.

이제 VSCode가 Numojo 패키지의 함수 힌트를 표시할 수 있습니다!

## 기여하기

여러분의 모든 기여를 **진심으로 감사드립니다**. 가이드라인(코딩 스타일, 테스트, 문서화, 릴리스 주기)은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

## 주의사항

이 라이브러리는 아직 초기 단계이며 마이너 버전 간에 호환성을 깨는 변경사항이 도입될 수 있습니다. 프로덕션이나 연구 코드에서는 버전을 고정하세요.

## 라이센스

LLVM 예외가 포함된 Apache 2.0 라이센스 하에 배포됩니다. 자세한 정보는 [LICENSE](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE)와 LLVM [License](https://llvm.org/LICENSE.txt)를 참조하세요.

이 프로젝트는 Apache License v2.0 with LLVM Exceptions로 라이센스된 [Mojo Standard Library](https://github.com/modularml/mojo)의 코드를 포함합니다 (LLVM [License](https://llvm.org/LICENSE.txt) 참조). MAX와 Mojo 사용 및 배포는 [MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license) 하에 라이센스됩니다.

## 감사의 글

[Modular](https://github.com/modularml)에서 만든 네이티브 [Mojo](https://github.com/modularml/mojo)로 구축되었습니다.

## 기여자

<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Mojo-Numerics-and-Algorithms-group/NuMojo" />
</a>