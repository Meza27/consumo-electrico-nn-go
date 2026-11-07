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

## Git Flow

Ver [docs/GITFLOW.md](docs/GITFLOW.md): ramas `main`, `develop` y `feature/*`, commits en inglés y Pull Requests a `develop`.
