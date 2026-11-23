# Git Flow del proyecto (grupo 10)

Ramas:

| Rama | Para qué sirve |
|------|----------------|
| `main` | Solo versiones entregadas (PC3, PC4, TB2). Cada entrega lleva un tag: `v0.3-pc3`, `v0.4-pc4`, `v1.0-tb2`. |
| `develop` | Rama de integración. Todo el trabajo terminado se junta aquí. |
| `feature/<tema>` | Una rama por tarea. Sale de `develop` y vuelve a `develop` por Pull Request. |
| `release/<version>` | (opcional) Se corta de `develop` para preparar la entrega y se mergea a `main` y `develop`. |
| `hotfix/<tema>` | Correcciones urgentes sobre `main`. |

Reglas: commits en inglés con el formato `type: description` (`feat`, `fix`, `test`,
`docs`, `chore`, `refactor`, `perf`).

## 1. Crear el repositorio

```bash
git init -b main
git add .gitignore go.mod
git commit -m "chore: initialize Go module and gitignore"
git remote add origin https://github.com/<usuario>/consumo-electrico-nn-go.git
git push -u origin main
git switch -c develop
git push -u origin develop
```

En GitHub: Settings → Branches → proteger `main` y `develop` (exigir Pull Request).
Crear el tablero Kanban (Projects) con columnas *Todo / In progress / Review / Done*.

## 2. Ramas feature para la PC3

Cada integrante clona el repositorio, configura su identidad y trabaja su rama:

```bash
git config user.name "Nombre Apellido"
git config user.email "correo-de-su-cuenta-github"
git switch develop && git pull
git switch -c feature/<tema>
# ... copiar/editar los archivos de su parte ...
git add <archivos>
git commit -m "feat: ..."
git push -u origin feature/<tema>
# abrir Pull Request feature/<tema> -> develop en GitHub; otro integrante lo revisa y aprueba
```

| Rama | Archivos | Commits sugeridos |
|------|----------|-------------------|
| `feature/data-cleaning` | `internal/preprocessing/`, `cmd/pc3/` | `feat: add concurrent CSV loader with producer/worker/reducer pipeline` · `fix: keep rows with negative billing as adjustments` · `test: cover negative billing and worker-count determinism` |
| `feature/mlp-model` | `internal/mlp/` (model, features, metrics) | `feat: add MLP regressor with sparse one-hot input layer` · `feat: add train/test split and train-only normalization` · `feat: add RMSE, MAE and R2 metrics` |
| `feature/parallel-training` | `internal/mlp/train.go`, `cmd/train/`, tests | `feat: add data-parallel training with goroutine workers and gradient reduction` · `test: check sequential and concurrent training give the same weights` · `feat: add benchmark mode for speedup measurements` |
| `feature/docker-docs` | `Dockerfile`, `.dockerignore`, `README.md`, `docs/` | `chore: add multi-stage Dockerfile` · `docs: add README with usage and results` |

## 3. Cerrar la entrega PC3

```bash
git switch develop && git pull
git switch main && git pull
git merge --no-ff develop -m "release: PC3 deliverable"
git tag -a v0.3-pc3 -m "PC3: data cleaning + sequential/concurrent MLP"
git push origin main --tags
```

## 4. Ramas feature para la PC4

Orden de integración, porque la API usa el clúster y las bases de datos: **1 → 2 → 3 → 4**.

| # | Rama | Responsable | Archivos | Commits sugeridos |
|---|------|-------------|----------|-------------------|
| 1 | `feature/distributed-cluster` | Rodrigo | `internal/mlp/train.go`, `internal/cluster/`, `cmd/node/`, `cmd/clusterbench/`, `results/pc4/cluster_benchmark.*` | `refactor: extract reusable GradPool from concurrent training` · `feat: add TCP protocol and ML node server` · `feat: add synchronous parameter-server coordinator` · `test: check distributed training matches sequential` · `feat: add cluster benchmark command` |
| 2 | `feature/databases` | Colleen | `internal/store/`, `docker-compose.yml` | `feat: add BSON encoder and decoder` · `feat: add minimal MongoDB OP_MSG client with connection pool` · `feat: add minimal Redis RESP client` · `feat: add repository for users, jobs, models and predictions` · `chore: add docker compose with MongoDB, Redis, ML nodes and API` |
| 3 | `feature/api` | Eduardo | `internal/api/`, `cmd/api/`, `cmd/loadtest/`, `Dockerfile`, `docker-entrypoint.sh`, `docs/postman_collection.json`, `results/pc4/loadtest*` | `feat: add JWT auth and password hashing` · `feat: add WebSocket hub for real-time events` · `feat: add REST endpoints for prediction, training and cluster` · `docs: add OpenAPI spec and Swagger UI` · `feat: add load test command` · `chore: build PC4 binaries in Docker image` |
| 4 | `feature/docs-pc4` | Rodrigo | `README.md`, `docs/GITFLOW.md` | `docs: document PC4 architecture, endpoints and results` |

## 5. Cerrar la entrega PC4

```bash
git switch develop && git pull
git switch main && git pull
git merge --no-ff develop -m "release: PC4 deliverable"
git tag -a v0.4-pc4 -m "PC4: distributed ML cluster, REST API, MongoDB and Redis"
git push origin main --tags
```
