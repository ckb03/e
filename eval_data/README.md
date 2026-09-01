# Evaluation data

Generate the frozen manifest with:

```bash
role-confusion-eval prepare
```

Manifests are intentionally ignored because they contain redistributed Wikipedia
HTML and are roughly 15 MB. For comparisons, copy the same manifest through durable
storage and verify the SHA-256 recorded in every run's `run.json`. Never regenerate
or overwrite a manifest between baseline and intervention runs.
