#!/bin/sh
# Punto de entrada del contenedor.
#   all      -> limpia los CSV (pc3) y entrena el MLP concurrente (PC3)
#   pc3      -> solo limpieza                 (args extra se pasan a pc3)
#   train    -> solo entrenamiento local      (args extra se pasan a train)
#   prep     -> limpia los CSV solo si falta data/processed/dataset_limpio.csv
#   node     -> nodo ML del clúster (PC4)
#   api      -> API coordinadora (PC4)
#   loadtest -> prueba de carga de la API (PC4)
#   clusterbench -> tiempos del entrenamiento con 1..K nodos (PC4)
set -e
cmd="${1:-all}"
[ $# -gt 0 ] && shift
case "$cmd" in
  all)
    pc3 "$@"
    train -mode conc
    ;;
  prep)
    if [ -f /app/data/processed/dataset_limpio.csv ]; then
      echo "dataset limpio ya existe: no se vuelve a procesar"
    else
      pc3 -in /app/data/raw/puno -out /app/data/processed/dataset_limpio.csv -summary /app/data/processed/resumen_limpieza.json
    fi
    ;;
  pc3)      exec pc3 "$@" ;;
  train)    exec train "$@" ;;
  node)     exec node "$@" ;;
  api)      exec api "$@" ;;
  loadtest) exec loadtest "$@" ;;
  clusterbench) exec clusterbench "$@" ;;
  *)        exec "$cmd" "$@" ;;
esac
