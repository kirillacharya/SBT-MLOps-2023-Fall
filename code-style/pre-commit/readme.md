# Pre-commit


Для начала вам понадобится либа:
```bash
pip install pre-commit
```

После этого перейдите в директорию с `.pre-commit-config.yaml` и выполните команду:

```bash
pre-commit install
```

Это команда установит хуки git в репозитории так, чтобы pre-commit мог их вызывать перед каждым коммитом.
То есть все будет запускаться автоматически.

---

Если хочется запустить отдельно, то можно так:
```bash
pre-commit run --all-files
```

Либо же для конкретного хука:
```bash
pre-commit run <hook_id> --all-files
```


Если хочется пропустить хуки, то можно так:
```bash
git commit -m "..." --no-verify
```