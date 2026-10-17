"""
req.py  -  Install the project's dependencies.

Run:  python req.py
"""
import subprocess
import sys

REQUIREMENTS = [
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "joblib>=1.3",
    "streamlit>=1.30",
]


def main():
    for pkg in REQUIREMENTS:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print("\nAll dependencies installed.")


if __name__ == "__main__":
    main()
