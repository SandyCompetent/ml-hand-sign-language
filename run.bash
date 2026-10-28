#!/usr/bin/env bash
# ==============================================================================
# Real-Time Sign Language Translator: Launcher & Environment Runner
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# ------------------------------------------------------------------------------
# 1. Ensure Virtual Environment Exists
# ------------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "[run.bash] Virtual environment not found at $VENV_DIR. Creating..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"
    elif [ -x "$HOME/Library/Python/3.9/bin/uv" ]; then
        "$HOME/Library/Python/3.9/bin/uv" venv --python 3.12 "$VENV_DIR" || "$HOME/Library/Python/3.9/bin/uv" venv "$VENV_DIR"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m venv "$VENV_DIR"
    else
        echo "[run.bash] Error: Neither uv nor python3 found. Please install Python 3.9+." >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 2. Activate Virtual Environment
# ------------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------------------
# 3. Ensure Package Dependencies Are Installed
# ------------------------------------------------------------------------------
if ! python -c "import sign_language" >/dev/null 2>&1; then
    echo "[run.bash] Installing package in editable mode..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install -e .
    elif [ -x "$HOME/Library/Python/3.9/bin/uv" ]; then
        "$HOME/Library/Python/3.9/bin/uv" pip install -e .
    else
        pip install -e .
    fi
fi

# ------------------------------------------------------------------------------
# 4. Command Dispatcher
# ------------------------------------------------------------------------------
COMMAND="${1:-app}"

case "$COMMAND" in
    app|"")
        echo "=========================================================="
        echo " 🤟 Launching Real-Time Sign Language Translator..."
        echo "=========================================================="
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "💡 macOS Note: If prompted, please allow Camera access in the system dialog."
            echo "   (System Settings -> Privacy & Security -> Camera -> Enable Terminal)"
        fi
        exec python -m sign_language
        ;;
    test)
        echo "=========================================================="
        echo " 🧪 Running Automated Unit & Integration Tests..."
        echo "=========================================================="
        exec pytest tests/ -v "${@:2}"
        ;;
    two-way|dictionary)
        echo "=========================================================="
        echo " ⇄ Launching Hearing Communicator (Text to Sign)..."
        echo "=========================================================="
        exec python -c "from sign_language.ui.two_way import TwoWayDictionaryWindow; import tkinter as tk; root=tk.Tk(); root.withdraw(); TwoWayDictionaryWindow(None); tk.mainloop()"
        ;;
    legacy-v3)
        echo "=========================================================="
        echo " 📜 Launching Legacy CodeV3 ASL App..."
        echo "=========================================================="
        if [ -f "legacy/phase3_asl_threshold/app.py" ]; then
            exec python legacy/phase3_asl_threshold/app.py
        else
            exec python CodeV3/app.py
        fi
        ;;
    legacy-dashboard)
        echo "=========================================================="
        echo " 📜 Launching Legacy Code Dashboard..."
        echo "=========================================================="
        if [ -d "legacy/phase1_basic_gestures" ]; then
            cd legacy/phase1_basic_gestures && exec python 03_dashboard.py
        else
            cd Code && exec python 03_dashboard.py
        fi
        ;;
    help|--help|-h)
        echo "Usage: ./run.bash [command]"
        echo ""
        echo "Commands:"
        echo "  app               Launch the modern MediaPipe Sign Language Translator (default)"
        echo "  test              Run automated test suite with pytest"
        echo "  two-way           Launch the Hearing-to-Deaf visual dictionary directly"
        echo "  legacy-v3         Launch the legacy CodeV3 thresholding app"
        echo "  legacy-dashboard  Launch the legacy Phase-1 Tkinter dashboard"
        echo "  help              Show this help message"
        exit 0
        ;;
    *)
        echo "[run.bash] Unknown command: '$COMMAND'"
        echo "Run './run.bash help' for usage instructions."
        exit 1
        ;;
esac
