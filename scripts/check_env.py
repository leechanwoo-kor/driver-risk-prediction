"""
환경 검증 스크립트
평가 서버 환경과 로컬 환경이 호환되는지 확인
"""
import sys
import platform
from typing import Dict, List, Tuple

# Windows 콘솔 인코딩 문제 해결
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_python_version() -> Tuple[bool, str]:
    """Python 버전 확인"""
    expected = (3, 10, 12)
    actual = sys.version_info[:3]

    if actual == expected:
        return True, f"✅ Python {'.'.join(map(str, actual))}"
    else:
        return False, f"❌ Python {'.'.join(map(str, actual))} (필요: {'.'.join(map(str, expected))})"


def check_package_versions() -> List[Tuple[bool, str]]:
    """필수 패키지 버전 확인"""
    results = []

    # 필수 패키지와 예상 버전
    required_packages = {
        'numpy': '1.26.4',
        'pandas': '2.2.2',
        'sklearn': '1.5.2',
        'yaml': '6.0.2',
    }

    # 추가 설치 패키지 (버전 확인만)
    additional_packages = {
        'xgboost': None,   # 2.1.3 권장
    }

    for package, expected_version in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                actual_version = sklearn.__version__
            elif package == 'yaml':
                import yaml
                actual_version = yaml.__version__
            else:
                module = __import__(package)
                actual_version = module.__version__

            if expected_version and actual_version == expected_version:
                results.append((True, f"✅ {package}: {actual_version}"))
            elif expected_version:
                results.append((False, f"❌ {package}: {actual_version} (필요: {expected_version})"))
            else:
                results.append((True, f"✅ {package}: {actual_version}"))

        except ImportError:
            results.append((False, f"❌ {package}: 설치되지 않음"))

    # 추가 패키지 확인
    for package, _ in additional_packages.items():
        try:
            module = __import__(package)
            actual_version = module.__version__
            results.append((True, f"✅ {package}: {actual_version} (추가 설치)"))
        except ImportError:
            results.append((False, f"❌ {package}: 설치되지 않음 (필수)"))

    return results


def check_optional_packages() -> List[Tuple[bool, str]]:
    """선택적 패키지 확인 (개발용)"""
    results = []

    optional_packages = ['matplotlib', 'seaborn', 'jupyter', 'notebook', 'tqdm', 'loguru']

    for package in optional_packages:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                version = module.__version__
            else:
                version = "unknown"
            results.append((True, f"ℹ️  {package}: {version} (개발용)"))
        except ImportError:
            results.append((False, f"ℹ️  {package}: 미설치 (개발용, 선택사항)"))

    return results


def check_system_info() -> List[str]:
    """시스템 정보 확인"""
    info = [
        f"OS: {platform.system()} {platform.release()}",
        f"Architecture: {platform.machine()}",
        f"Processor: {platform.processor()}",
    ]
    return info


def main():
    """메인 검증 함수"""
    print("=" * 70)
    print("환경 검증 스크립트")
    print("=" * 70)
    print()

    # 시스템 정보
    print("📋 시스템 정보")
    print("-" * 70)
    for info in check_system_info():
        print(f"  {info}")
    print()

    # Python 버전 확인
    print("🐍 Python 버전")
    print("-" * 70)
    py_ok, py_msg = check_python_version()
    print(f"  {py_msg}")
    print()

    # 필수 패키지 확인
    print("📦 필수 패키지 (평가 서버 호환)")
    print("-" * 70)
    package_results = check_package_versions()
    all_packages_ok = all(ok for ok, _ in package_results)

    for ok, msg in package_results:
        print(f"  {msg}")
    print()

    # 개발용 패키지 확인
    print("🔧 개발용 패키지 (선택사항)")
    print("-" * 70)
    optional_results = check_optional_packages()
    for ok, msg in optional_results:
        print(f"  {msg}")
    print()

    # 최종 결과
    print("=" * 70)
    if py_ok and all_packages_ok:
        print("✅ 모든 필수 환경이 평가 서버와 호환됩니다!")
        print("제출 전에는 scripts/validate_submission.py로 제출 파일을 검증하세요.")
        return 0
    else:
        print("❌ 일부 환경 설정에 문제가 있습니다.")
        print()
        print("해결 방법:")
        if not py_ok:
            print("  1. Python 3.10.12 설치")
            print("     conda install python=3.10.12")
        if not all_packages_ok:
            print("  2. 필수 패키지 재설치")
            print("     conda env update -f environment.yml --prune")
            print("     또는")
            print("     conda env remove -n driver-risk")
            print("     conda env create -f environment.yml")
        print()
        print("자세한 내용은 SETUP.md를 참조하세요.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
