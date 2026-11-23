// Comando api: API REST/WebSocket coordinadora del clúster de nodos ML.
//
// Variables de entorno (con sus valores por defecto):
//
//	API_ADDR=:8080  NODES=localhost:9001,localhost:9002,localhost:9003
//	MONGO_ADDR=localhost:27017  MONGO_DB=consumo  REDIS_ADDR=localhost:6379
//	DATA_PATH=data/processed/dataset_limpio.csv  MODEL_PATH=results/model.json
//	JWT_SECRET=...  ADMIN_USER=admin  ADMIN_PASS=admin123  DEMO_USER=demo  DEMO_PASS=demo123
package main

import (
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"consumo-electrico-nn-go/internal/api"
	"consumo-electrico-nn-go/internal/cluster"
	"consumo-electrico-nn-go/internal/platform"
	"consumo-electrico-nn-go/internal/store"
)

func main() {
	platform.SinEcoQoS()
	log.SetPrefix("[api] ")
	nodos := strings.Split(env("NODES", "localhost:9001,localhost:9002,localhost:9003"), ",")

	// Las bases de datos pueden tardar en arrancar dentro de Docker Compose:
	// se reintenta durante un minuto.
	var st *store.Store
	var err error
	for i := 0; i < 30; i++ {
		if st, err = store.Open(env("MONGO_ADDR", "localhost:27017"), env("MONGO_DB", "consumo"), env("REDIS_ADDR", "localhost:6379")); err == nil {
			break
		}
		log.Printf("esperando bases de datos: %v", err)
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		log.Fatal(err)
	}

	coord := cluster.NewCoordinator(nodos, log.Printf)
	srv, err := api.NewServer(api.Config{
		Addr:      env("API_ADDR", ":8080"),
		NodeAddrs: nodos,
		DataPath:  env("DATA_PATH", "data/processed/dataset_limpio.csv"),
		ModelPath: env("MODEL_PATH", "results/model.json"),
		JWTSecret: []byte(env("JWT_SECRET", "cambiar-este-secreto-en-produccion")),
		AdminUser: env("ADMIN_USER", "admin"),
		AdminPass: env("ADMIN_PASS", "admin123"),
		DemoUser:  env("DEMO_USER", "demo"),
		DemoPass:  env("DEMO_PASS", "demo123"),
	}, st, coord)
	if err != nil {
		log.Fatal(err)
	}
	srv.Start()
	addr := env("API_ADDR", ":8080")
	log.Printf("API escuchando en %s (Swagger en /swagger/), nodos: %v", addr, nodos)
	hs := &http.Server{Addr: addr, Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(hs.ListenAndServe())
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
