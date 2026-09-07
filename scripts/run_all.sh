set -e
python3 -m qshield.cli benchmark --input cases/reference_case.json --trials 4000 --output results/benchmark_reference.json --quiet
python3 -m qshield.cli benchmark --input cases/chokepoint_case.json --trials 4000 --output results/benchmark_chokepoint.json --quiet
python3 -m qshield.cli sensitivity --input cases/reference_case.json --output results/sensitivity_reference.json --quiet
python3 -m qshield.cli sensitivity --input cases/chokepoint_case.json --output results/sensitivity_chokepoint.json --quiet
python3 -m qshield.cli ablation --instances 400 --trials 200 --output results/ablation.json --quiet
python3 -m qshield.cli generalization --instances 300 --output results/generalization.json --quiet
python3 scripts/greedy_gap.py > /dev/null
echo ALL_DONE
