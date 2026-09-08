# Benchmark report workflow

The report builder is read-only with respect to model checkpoints and
environments. It consumes only canonical `result.json` files produced by the
unified evaluator.

```bash
python scripts/build_benchmark_report.py \
  --result \
  results/benchmark/benchmark_act_libero_dev5_v1/result.json \
  results/benchmark/benchmark_openvla_libero_dev5_v1/result.json \
  --output-dir results/benchmark/report_v1
```

The output contains:

- `report.md`: human-readable comparison;
- `summary.csv`: spreadsheet-friendly table;
- `summary.json`: machine-readable report with provenance.

The report explicitly labels these as five-state development results. It does
not overwrite or reinterpret the preregistered 45-state held-out final files.
