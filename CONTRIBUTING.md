# Contributing

1. Create a feature branch.
2. Keep experiments reproducible.
3. Add tests for new deterministic components.
4. Document new research assumptions.
5. Do not describe behavioral metrics as proof of consciousness.
6. Keep generated logs/checkpoints out of Git unless intentionally selected as research artifacts.

Suggested workflow:

```bash
git checkout -b feature/your-feature
pytest
git add .
git commit -m "feat: describe the change"
git push -u origin feature/your-feature
```
