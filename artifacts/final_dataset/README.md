# Final scored analysis dataset

This is the final table used as input to the thesis analyses. Each row corresponds to one generated image/trajectory and should include:

- prompt/run identifiers;
- prompt text;
- seed information;
- automatic quality scores for CLIP, HPSv2, and ImageReward;
- trajectory-derived prefix features;
- any derived columns used by the analysis scripts.

This file is the main documentation artifact for the experiment. It replaces the need to include duplicated intermediate result folders such as separate prompt-controlled analysis outputs, rich-analysis outputs, or per-metric result directories.

Example inspection command:

```bash
python - <<'PY'
import pandas as pd
path = 'analysis_input_all_quality.csv'
df = pd.read_csv(path)
print(df.shape)
print(df.columns.tolist())
print(df.head())
PY
```
