# LLM Power-Flow Benchmark (Per Case)

| model | task | case | success_rate | voltage_mae | flow_mae | loading_rmse | voltage_f1 | thermal_f1 | conv_match | prompt_tokens | completion_tokens | total_tokens | cost_usd_mean | cost_usd_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini:gemini-2.5-flash-lite | baseline_pf | case118 | 0.0 | None | None | None | None | None | None | None | None | None | None | None |
| gemini:gemini-2.5-flash-lite | baseline_pf | case14 | 0.5 | 0.04358706006076492 | 31.633811238183696 | 4.282554438229779 | 0.30769230769230765 | 1.0 | 1.0 | 2578.0 | 1317.0 | 3895.0 | 0.0007846000000000001 | 0.0007846000000000001 |
| gemini:gemini-2.5-flash-lite | baseline_pf | case30 | 0.5 | 0.011057299270549444 | 7.785598049264872 | 38.126686211307074 | 1.0 | 0.0 | 1.0 | 3999.0 | 3061.0 | 7060.0 | 0.0016243 | 0.0016243 |
| gemini:gemini-2.5-flash-lite | baseline_pf | case300 | 0.0 | None | None | None | None | None | None | None | None | None | None | None |
| gemini:gemini-2.5-flash-lite | baseline_pf | case57 | 0.5 | 0.11390041891227162 | 102.48930151930017 | 9.714717808771196 | 0.0 | 1.0 | 1.0 | 7732.0 | 6275.0 | 14007.0 | 0.0032832 | 0.0032832 |
| gemini:gemini-2.5-flash-lite | blueprint_pf | case118 | 0.0 | None | None | None | None | None | None | None | None | None | None | None |
| gemini:gemini-2.5-flash-lite | blueprint_pf | case14 | 1.0 | 0.008018421845643022 | 22.43194995900013 | 7.973425102330718 | 0.5714285714285714 | 1.0 | 1.0 | 2620.0 | 2282.0 | 4902.0 | 0.0011748000000000001 | 0.0023496000000000003 |
| gemini:gemini-2.5-flash-lite | blueprint_pf | case30 | 1.0 | 0.037623359460070914 | 7.591522774196829 | 33.85846882390924 | 1.0 | 0.0 | 1.0 | 3952.0 | 3731.0 | 7683.0 | 0.0018876000000000001 | 0.0037752000000000003 |
| gemini:gemini-2.5-flash-lite | blueprint_pf | case300 | 0.0 | None | None | None | None | None | None | None | None | None | None | None |
| gemini:gemini-2.5-flash-lite | blueprint_pf | case57 | 1.0 | 0.10234472868898557 | 85.36126642607405 | 5.107475755083976 | 0.18181818181818182 | 1.0 | 1.0 | 7923.0 | 7372.0 | 15295.0 | 0.0037411000000000002 | 0.0074822000000000005 |

Prompt:
```
python3 benchmarks/evaluate_llms.py \
  --model gemini:gemini-2.5-flash-lite \
  --task baseline_pf \
  --task blueprint_pf \
  --case case14 \
  --case case30 \
  --case case57 \
  --case case118 \
  --case case300 \
  --runs 2 \
  --pricing-file pricing.json \
  --out-dir benchmarks/results
```