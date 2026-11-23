# Monitoreo y predicción de consumo eléctrico con redes neuronales en Go

Trabajo final de **CC65 Programación Concurrente y Distribuida** (UPC, 2026-20), grupo 10:
Rodrigo Meza, Eduardo Bravo y Loana Rodríguez.

**Caso 5:** monitoreo y predicción del consumo eléctrico doméstico. Usamos los datos abiertos de
facturación y consumo por cliente de **Electro Puno** (datosabiertos.gob.pe), que suman
2,369,272 registros en 7 meses de 2023. Algoritmo: **red neuronal (MLP) con data parallelism**.

**Pregunta de impacto social:** ¿cuánta energía consumirá cada zona (distrito, tarifa y mes) de la
región Puno? Con esa estimación, la distribuidora puede anticipar los picos de demanda, planificar
la capacidad de la red y priorizar el mantenimiento antes de que la red colapse.

Todo está escrito en Go, solo con la librería estándar (sin dependencias externas).

## Estructura

```
cmd/pc3/                 limpieza concurrente de los CSV crudos
cmd/train/               entrenamiento del MLP (secuencial, concurrente y benchmark)
internal/preprocessing/  productor -> workers -> reductor (goroutines y channels)
internal/mlp/            MLP, codificación de features, entrenamiento y métricas
internal/platform/       desactiva EcoQoS en Windows para medir tiempos de forma justa
internal/cluster/        (PC4) nodos ML y coordinador: entrenamiento distribuido por TCP
internal/store/          (PC4) clientes de MongoDB (BSON + OP_MSG) y Redis (RESP) con la librería estándar
internal/api/            (PC4) API REST, JWT, WebSocket y Swagger
cmd/node, cmd/api        (PC4) nodo ML y API coordinadora
cmd/loadtest             (PC4) prueba de carga de la API
cmd/clusterbench         (PC4) tiempos del entrenamiento con 1..K nodos
docker-compose.yml       (PC4) MongoDB + Redis + 3 nodos ML + API
results/                 evidencias: benchmark, métricas, salidas y modelo entrenado
docs/                    Git Flow y material del informe
```

## Cómo ejecutar

Requisitos: Go 1.22 o superior. Los CSV de Electro Puno van en `data/raw/puno/` (no se suben al repositorio).

```bash
go test ./...                                    # pruebas
go run ./cmd/pc3 -workers 8                      # 1) limpieza -> data/processed/dataset_limpio.csv
go run ./cmd/train -mode seq                     # 2a) entrenamiento secuencial
go run ./cmd/train -mode conc -workers 8         # 2b) entrenamiento concurrente
go run ./cmd/train -mode bench -bench-workers 1,2,4,8,16   # 3) tiempos y speedup
```

Opciones de `train`: `-epochs 10 -batch 4096 -lr 0.003 -hidden 64,32 -test 0.2 -seed 42 -limit N`.

### Docker

```bash
docker build -t consumo-nn .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" consumo-nn            # limpieza + entrenamiento
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" consumo-nn train -mode bench
```

Las pruebas se ejecutan durante el `docker build`: si alguna falla, la imagen no se construye.

## Parte A: limpieza concurrente

Patrón **productor → workers → reductor**:

1. El **productor** (una goroutine) lee los archivos línea por línea y los parte en bloques de 5,000 líneas, que envía por un channel.
2. Los **workers** (N goroutines) limpian cada bloque por separado, sin memoria compartida. Convierten Latin-1 a UTF-8, validan el número de columnas, los números, el ubigeo, el periodo y las fechas, y calculan la antigüedad del cliente.
3. El **reductor** (una goroutine) recibe los resultados, los reordena por número de bloque (la salida es idéntica con 1 u 8 workers), elimina los duplicados (cliente, año, mes) y es el único que escribe el CSV y el resumen. Por eso no se necesitan mutex.

Resultado: 2,369,272 filas leídas y 2,369,272 válidas. Las 18,159 filas con **facturación negativa se
conservan**: son ajustes o notas de crédito, su consumo es válido y la facturación no se usa como
feature. El resumen solo las cuenta (`facturacion_negativa_conservadas`).

## Parte B: MLP de regresión

- **Objetivo:** consumo mensual en kWh. Se entrena sobre `log(1+kWh)` estandarizado, porque la distribución tiene una cola enorme (mediana de 16 kWh y máximo de ~500,000 kWh).
- **Entradas (134):** one-hot de tarifa (10), one-hot de distrito por ubigeo (111), one-hot de mes (12) y antigüedad estandarizada (1).
- **Se excluye la FACTURACIÓN:** se calcula a partir del consumo, así que usarla sería fuga de información.
- **Arquitectura:** 134 → 64 (ReLU) → 32 (ReLU) → 1 (lineal), con 10,753 parámetros. Pérdida MSE, optimizador Adam y mini-lotes de 4,096 filas.
- **Sin fuga de datos:** split train/test 80/20 aleatorio con semilla. El `Encoder` (vocabularios, medias y desviaciones) se ajusta **solo con train**.
- **Primera capa dispersa:** solo hay 4 entradas activas por fila, así que la red suma esas 4 filas de pesos en lugar de multiplicar las 134.

### Versión secuencial vs. concurrente (data parallelism)

| | Secuencial | Concurrente |
|---|---|---|
| Por cada mini-lote | una goroutine recorre las 4,096 filas y acumula el gradiente | el lote se parte en trozos de 128 filas dentro de una cola (channel); P workers toman trozos y acumulan el gradiente en **su propio buffer** |
| Sincronización | — | barrera: el coordinador espera el aviso de todos los trozos |
| Actualización | Adam | el coordinador **suma los P gradientes** (reducción tipo AllReduce) y **solo él** aplica Adam |

No hay condiciones de carrera. Mientras los workers leen los pesos, nadie los escribe. Mientras el
coordinador los actualiza, todos los workers están bloqueados esperando el siguiente trozo. El
channel garantiza ese orden (happens-before). El test `TestSecuencialIgualConcurrente` verifica
que las dos versiones llegan a los mismos pesos (diferencia < 1e-8).

## Resultados (i7-12700KF: 8 P-cores + 4 E-cores, 20 hilos; Windows 11; Go 1.27)

Entrenamiento con 1,895,418 filas, 3 épocas (`results/benchmark.csv`):

| Modo | Workers | s/época | Speedup | Eficiencia |
|---|---:|---:|---:|---:|
| secuencial | 1 | 5.69 | 1.00x | 100% |
| concurrente | 2 | 2.87 | 1.98x | 99% |
| concurrente | 4 | 1.64 | 3.46x | 87% |
| concurrente | 8 | 0.99 | 5.75x | 72% |
| concurrente | 16 | 0.85 | 6.68x | 42% |

Métricas en test, con 10 épocas (`results/metrics_conc.json`, idénticas a `metrics_seq.json`):

| | RMSE (kWh) | MAE (kWh) | R² (kWh) | R² (log) |
|---|---:|---:|---:|---:|
| MLP | 1303.08 | 51.17 | 0.019 | 0.249 |
| Baseline (predice la mediana) | 1316.21 | 56.78 | -0.001 | 0.000 |

La red explica el 25% de la variación del consumo en escala logarítmica y baja el MAE un 10% respecto al
baseline. El R² en kWh es bajo porque unos pocos clientes con consumos gigantes dominan el RMSE.
Además, las 4 features no describen al cliente individual. Mejoras para la PC4: agregar el consumo
del mes anterior y el estado del cliente.

## PC4: clúster de nodos ML, API y bases de datos

```bash
docker compose up --build -d          # MongoDB, Redis, 3 nodos ML y la API
# Swagger:  http://localhost:8080/swagger/   (usuarios: admin/admin123, demo/demo123)
# Postman:  importar docs/postman_collection.json
go run ./cmd/loadtest -n 10000 -c 50                                  # prueba de carga
docker compose run --rm api clusterbench -epochs 2                     # 1, 2 y 3 nodos
```

### Arquitectura (topología en estrella)

```
            cliente (Postman / SPA de la TB2)
                     │ HTTP + WebSocket (JWT)
              ┌──────▼───────┐        ┌─────────┐
              │ API (Go)     │───────▶│ MongoDB │  users, training_jobs, models,
              │ coordinador  │        └─────────┘  predictions, node_metrics
              │ Adam + Σ g   │───────▶┌─────────┐
              └──┬────┬────┬─┘        │  Redis  │  caché de predicciones, estado del
         TCP/gob │    │    │          └─────────┘  clúster, progreso, eventos
          ┌──────▼┐ ┌─▼─────┐ ┌▼──────┐
          │ node1 │ │ node2 │ │ node3 │   cada nodo: su shard (fila i -> nodo i % K)
          └───────┘ └───────┘ └───────┘   y un GradPool de goroutines
```

En cada paso, el coordinador envía a cada nodo los pesos (10,753 float64) y las filas del lote que le
pertenecen. Cada nodo calcula el gradiente en paralelo y lo devuelve. El coordinador espera a todos
(barrera), suma los gradientes, promedia y es el **único** que actualiza los pesos. El resultado es
idéntico al secuencial (`TestDistribuidoIgualSecuencial`). Si un nodo falla, se marca como caído y el
entrenamiento sigue con los demás (`TestNodoCaidoSeOmite`).

### Endpoints

| Método | Ruta | Rol | Descripción |
|---|---|---|---|
| POST | `/api/auth/login` | público | JWT HS256 (8 h) |
| GET | `/api/health` | público | estado de MongoDB, Redis, nodos y modelo |
| POST | `/api/predict` | user | kWh estimados (caché Redis; registro asíncrono en MongoDB) |
| GET | `/api/demand?mes=7` | user | demanda estimada por distrito (pregunta de impacto) |
| GET | `/api/model` | user | modelo activo y métricas |
| GET | `/api/predictions`, `/api/predictions/stats` | user | historial y agregación en MongoDB |
| POST | `/api/train` | admin | entrenamiento distribuido en los nodos (202) |
| GET | `/api/train`, `/api/train/{id}` | user | historial y progreso |
| GET | `/api/cluster/nodes`, `/api/cluster/logs` | admin | estado de los nodos y eventos |
| GET | `/api/metrics` | admin | latencia p50/p95/p99 y uso del proceso |
| GET | `/ws?token=` | user | eventos en tiempo real |

### Resultados (i7-12700KF, Docker Desktop con WSL2)

Entrenamiento distribuido, 2 épocas y 1,895,418 filas, con 4 goroutines por nodo (`results/pc4/cluster_benchmark.txt`):

| Modo | Nodos | s/época | Speedup | Paso (ms) | Comunicación |
|---|---:|---:|---:|---:|---:|
| secuencial | – | 6.80 | 1.00x | – | – |
| distribuido | 1 | 2.32 | 2.94x | 4.95 | 12 % |
| distribuido | 2 | 1.56 | 4.35x | 3.32 | 20 % |
| distribuido | 3 | 1.47 | 4.62x | 3.12 | 21 % |

Prueba de carga de `POST /api/predict` (`results/pc4/loadtest.txt`):

| Concurrencia | req/s | p50 | p95 | p99 | < 100 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,547 | 0.50 ms | 1.14 ms | 1.23 ms | 100 % |
| 50 | 11,278 | 4.00 ms | 5.78 ms | 7.50 ms | 99.85 % |
| 200 | 13,968 | 11.61 ms | 14.75 ms | 161.57 ms | 98.43 % |

## Git Flow

Ver [docs/GITFLOW.md](docs/GITFLOW.md): ramas `main`, `develop` y `feature/*`, commits en inglés y Pull Requests a `develop`.
