#!/bin/bash
#
# Benchmark runner for tldr-code
#
# Usage:
#   ./benchmarks/run_benchmarks.sh              # Run all benchmarks
#   ./benchmarks/run_benchmarks.sh baseline     # Save as baseline
#   ./benchmarks/run_benchmarks.sh compare      # Compare against baseline
#   ./benchmarks/run_benchmarks.sh quick        # Quick run (fewer iterations)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_ROOT/.benchmarks"

# Ensure benchmark storage directory exists
mkdir -p "$BENCHMARK_DIR"

cd "$PROJECT_ROOT"

case "${1:-run}" in
    baseline)
        echo "Running benchmarks and saving as baseline..."
        uv run pytest benchmarks/bench_performance.py \
            -v \
            --benchmark-only \
            --benchmark-save=baseline \
            --benchmark-storage="$BENCHMARK_DIR" \
            --benchmark-sort=mean \
            --benchmark-columns=min,max,mean,stddev,rounds
        echo ""
        echo "Baseline saved to $BENCHMARK_DIR"
        ;;

    compare)
        if [ ! -d "$BENCHMARK_DIR" ] || [ -z "$(ls -A "$BENCHMARK_DIR" 2>/dev/null)" ]; then
            echo "No baseline found. Run './benchmarks/run_benchmarks.sh baseline' first."
            exit 1
        fi
        echo "Running benchmarks and comparing against baseline..."
        uv run pytest benchmarks/bench_performance.py \
            -v \
            --benchmark-only \
            --benchmark-compare=baseline \
            --benchmark-storage="$BENCHMARK_DIR" \
            --benchmark-sort=mean \
            --benchmark-columns=min,max,mean,stddev,rounds \
            --benchmark-compare-fail=mean:10%
        ;;

    quick)
        echo "Running quick benchmarks (fewer iterations)..."
        uv run pytest benchmarks/bench_performance.py \
            -v \
            --benchmark-only \
            --benchmark-disable-gc \
            --benchmark-warmup=on \
            --benchmark-min-rounds=3 \
            --benchmark-sort=mean
        ;;

    group)
        if [ -z "$2" ]; then
            echo "Usage: ./run_benchmarks.sh group <group-name>"
            echo "Available groups: structure, callgraph, search, impact, cfg, dfg, synthetic, scale-*"
            exit 1
        fi
        echo "Running benchmarks for group: $2"
        uv run pytest benchmarks/bench_performance.py \
            -v \
            --benchmark-only \
            --benchmark-group-by=group \
            -k "$2" \
            --benchmark-sort=mean
        ;;

    json)
        echo "Running benchmarks with JSON output..."
        OUTPUT_FILE="$BENCHMARK_DIR/results_$(date +%Y%m%d_%H%M%S).json"
        uv run pytest benchmarks/bench_performance.py \
            -v \
            --benchmark-only \
            --benchmark-json="$OUTPUT_FILE" \
            --benchmark-sort=mean
        echo ""
        echo "Results saved to: $OUTPUT_FILE"
        ;;

    run|*)
        echo "Running all benchmarks..."
        uv run pytest benchmarks/bench_performance.py \
            -v \
            --benchmark-only \
            --benchmark-sort=mean \
            --benchmark-columns=min,max,mean,stddev,rounds \
            --benchmark-group-by=group
        ;;
esac

echo ""
echo "Done."
