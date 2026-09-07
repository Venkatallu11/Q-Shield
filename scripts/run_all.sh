set -e
python3 -m qshield.cli benchmark --input cases/reference_case.json --trials 4000 --output results/benchmark_reference.json --quiet
python3 -m qshield.cli benchmark --input cases/chokepoint_case.json --trials 4000 --output results/benchmark_chokepoint.json --quiet
python3 -m qshield.cli sensitivity --input cases/reference_case.json --output results/sensitivity_reference.json --quiet
python3 -m qshield.cli sensitivity --input cases/chokepoint_case.json --output results/sensitivity_chokepoint.json --quiet
python3 -m qshield.cli ablation --instances 400 --trials 200 --output results/ablation.json --quiet
python3 -m qshield.cli generalization --instances 300 --output results/generalization.json --quiet
python3 scripts/greedy_gap.py > /dev/null

# --- 0.5: tail risk, identity infrastructure, personal identity -------------
python3 -m qshield.cli benchmark --input cases/pki_case.json --trials 4000 --output results/benchmark_pki.json --quiet
python3 -m qshield.cli benchmark --input cases/personal_identity_case.json --trials 4000 --output results/benchmark_personal.json --quiet
python3 -m qshield.cli tail-risk --instances 400 --output results/tail_risk.json --quiet
python3 -m qshield.cli hierarchy --instances 300 --output results/hierarchy.json --quiet
python3 -m qshield.cli hierarchy --instances 200 --sweep-root-cost --output results/hierarchy_root_cost.json --quiet
python3 -m qshield.cli threat-class --instances 300 --output results/threat_class.json --quiet
python3 -m qshield.cli personal --instances 300 --output results/personal_identity.json --quiet
echo ALL_DONE
