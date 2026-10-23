# P27 healthcheck gold reports

Generated on CPU with:

```bash
OMP_NUM_THREADS=1 python code/28_minigpt_healthcheck.py --json > docs/gold/p27_healthcheck.json
OMP_NUM_THREADS=1 python code/28_minigpt_healthcheck.py --mtp --json > docs/gold/p27_healthcheck_mtp.json
python scripts/plot_p27_healthcheck.py docs/gold/p27_healthcheck_mtp.json --out-dir assets/diagrams/p27
```

Do not hand-edit the JSON or SVGs. Regenerate after changing `code/28_minigpt_healthcheck.py` or the models it imports. Timing medians are machine-local; error magnitudes must stay within `atol=3e-6`.
