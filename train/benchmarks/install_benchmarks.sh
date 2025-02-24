# Parse arguments
OVERRIDE=false
STOP=false

while [[ $STOP = false ]]
do
    case "$1" in
        -o|--override)
            OVERRIDE=true
            shift
            ;;
        *)
            STOP=true
            ;;
    esac
done

# Remove existing benchmarks
if [ "$OVERRIDE" = true ]; then
    rm -rf benchmarks/*/
fi

# Set up a Python venv
echo "Setting up Python environment ..."
python3 -m venv benchmarks_venv
source benchmarks_venv/bin/activate
pip --disable-pip-version-check install -q -r benchmarks/requirements.txt

# OWASP
if [ ! -d "benchmarks/owasp" ]; then
    echo "Installing OWASP Benchmark ..."
    git clone -q https://github.com/OWASP-Benchmark/BenchmarkJava benchmarks/owasp_original
    python3 benchmarks/preprocess_owasp.py
    rm -rf benchmarks/owasp_original
fi

# CryptoAPIBench
if [ ! -d "benchmarks/cryptoapibench" ]; then
    echo "Installing CryptoAPI Benchmark ..."
    git clone -q https://github.com/CryptoAPI-Bench/CryptoAPI-Bench benchmarks/cryptoapibench_original
    python3 benchmarks/preprocess_cryptoapibench.py
    rm -rf benchmarks/cryptoapibench_original
fi

# Clean up
rm -r benchmarks_venv
echo "Done."
