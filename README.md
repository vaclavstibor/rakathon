# Breast Tumor Recurrence Prediction

Streamlit app for breast cancer recurrence risk estimation with SHAP explanation.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
./run_app.sh
```

The app starts at [http://127.0.0.1:8501](http://127.0.0.1:8501).

## Requirements

- Python 3.12+ recommended
- Dependencies from `requirements.txt` (installed by `run_app.sh`)

## macOS Note (LightGBM)

If you see `Library not loaded: @rpath/libomp.dylib`, install OpenMP:

```bash
brew install libomp
```

Then rerun:

```bash
./run_app.sh
```
