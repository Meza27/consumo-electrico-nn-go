// Comando node: nodo ML del clúster. Escucha por TCP y calcula gradientes de
// su shard con un pool de goroutines cuando el coordinador (la API) se lo pide.
//
// Uso:
//
//	go run ./cmd/node -id node1 -addr :9000 -workers 4
//
// También lee NODE_ID, NODE_ADDR y NODE_WORKERS (útil en Docker Compose).
package main

import (
	"flag"
	"log"
	"os"
	"strconv"

	"consumo-electrico-nn-go/internal/cluster"
	"consumo-electrico-nn-go/internal/platform"
)

func main() {
	id := flag.String("id", env("NODE_ID", "node1"), "identificador del nodo")
	addr := flag.String("addr", env("NODE_ADDR", ":9000"), "dirección TCP donde escuchar")
	workers := flag.Int("workers", envInt("NODE_WORKERS", 0), "goroutines de cómputo (0 = núcleos de la CPU)")
	flag.Parse()
	platform.SinEcoQoS()
	log.SetPrefix("[" + *id + "] ")
	log.Fatal(cluster.NewNode(*id, *workers).ListenAndServe(*addr))
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return def
}
