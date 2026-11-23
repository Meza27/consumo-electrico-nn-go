// Comando clusterbench: mide el entrenamiento distribuido con 1, 2, ..., K
// nodos ML (mismos datos, misma semilla) y lo compara con el entrenamiento
// secuencial local. Se ejecuta dentro de la red de Docker Compose:
//
//	docker compose run --rm api clusterbench -nodes node1:9000,node2:9000,node3:9000
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"consumo-electrico-nn-go/internal/cluster"
	"consumo-electrico-nn-go/internal/mlp"
)

type corrida struct {
	Modo         string  `json:"modo"`
	Nodos        int     `json:"nodos"`
	Segundos     float64 `json:"segundos"`
	SegPorEpoca  float64 `json:"segundos_por_epoca"`
	Speedup      float64 `json:"speedup"`
	PasoMs       float64 `json:"paso_ms"`
	ComunicMs    float64 `json:"comunicacion_ms"`
	PctComunic   float64 `json:"pct_comunicacion"`
	PerdidaFinal float64 `json:"perdida_final"`
	R2Log        float64 `json:"r2_log_test"`
}

func main() {
	data := flag.String("data", "/app/data/processed/dataset_limpio.csv", "CSV limpio")
	nodes := flag.String("nodes", "node1:9000,node2:9000,node3:9000", "nodos ML")
	epochs := flag.Int("epochs", 2, "épocas por corrida")
	batch := flag.Int("batch", 4096, "mini-lote")
	out := flag.String("out", "", "archivo JSON de salida (opcional)")
	flag.Parse()
	addrs := strings.Split(*nodes, ",")

	// Referencia: entrenamiento secuencial en este mismo contenedor.
	filas, err := mlp.LoadCSV(*data, 0)
	if err != nil {
		log.Fatal(err)
	}
	train, _ := mlp.Split(filas, 0.2, 42)
	enc := mlp.FitEncoder(train)
	dTr := enc.Encode(train)
	sizes := []int{enc.InputDim(), 64, 32, 1}
	m := mlp.NewMLP(sizes, 42)
	ini := time.Now()
	h := mlp.TrainSequential(m, dTr, mlp.TrainConfig{Epochs: *epochs, Batch: *batch, LR: 0.003, Seed: 42})
	tSeq := time.Since(ini).Seconds()
	res := []corrida{{Modo: "secuencial (1 proceso)", Nodos: 0, Segundos: tSeq, SegPorEpoca: tSeq / float64(*epochs), Speedup: 1, PerdidaFinal: h[len(h)-1].Loss}}
	filas, train, dTr = nil, nil, nil
	fmt.Printf("secuencial: %.2f s\n", tSeq)

	for k := 1; k <= len(addrs); k++ {
		coord := cluster.NewCoordinator(addrs[:k], log.New(io.Discard, "", 0).Printf)
		r, err := coord.Train(context.Background(), cluster.JobConfig{JobID: fmt.Sprintf("bench-%d", k), DataPath: *data,
			TestFrac: 0.2, Seed: 42, Epochs: *epochs, Batch: *batch, LR: 0.003, Hidden: []int{64, 32}}, nil)
		if err != nil {
			log.Fatal(err)
		}
		c := corrida{Modo: "distribuido", Nodos: k, Segundos: r.Seconds, SegPorEpoca: r.Seconds / float64(*epochs),
			Speedup: tSeq / r.Seconds, PasoMs: r.AvgStepMs, ComunicMs: r.AvgCommMs, PerdidaFinal: r.History[len(r.History)-1].Loss,
			R2Log: r.Test.Log.R2}
		if r.AvgStepMs > 0 {
			c.PctComunic = 100 * r.AvgCommMs / r.AvgStepMs
		}
		res = append(res, c)
		fmt.Printf("%d nodo(s): %.2f s\n", k, r.Seconds)
	}

	fmt.Println("\n  modo                    nodos  tiempo(s)  s/época  speedup  paso(ms)  comunic.(ms)  pct.comunic.  pérdida")
	for _, r := range res {
		fmt.Printf("  %-22s %6d %10.2f %8.2f %7.2fx %9.2f %13.2f %9.1f%% %8.4f\n",
			r.Modo, r.Nodos, r.Segundos, r.SegPorEpoca, r.Speedup, r.PasoMs, r.ComunicMs, r.PctComunic, r.PerdidaFinal)
	}
	if *out != "" {
		b, _ := json.MarshalIndent(map[string]any{"epochs": *epochs, "batch": *batch, "corridas": res}, "", "  ")
		os.WriteFile(*out, b, 0o644)
	}
}
