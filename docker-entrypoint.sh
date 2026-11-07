#!/bin/sh
# Punto de entrada del contenedor.
#   all   -> limpia los CSV (pc3) y entrena el MLP concurrente (por defecto)
#   pc3   -> solo limpieza          (args extra se pasan a pc3)
#   train -> solo entrenamiento     (args extra se pasan a train, ej: -mode bench)
set -e
cmd="${1:-all}"
[ $# -gt 0 ] && shift
case "$cmd" in
  all)
    pc3 "$@"
    train -mode conc
    ;;
  pc3)   exec pc3 "$@" ;;
  train) exec train "$@" ;;
  *)     exec "$cmd" "$@" ;;
esac
