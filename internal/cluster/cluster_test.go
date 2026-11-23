package cluster

import (
	"context"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"consumo-electrico-nn-go/internal/mlp"
)

// csvSintetico escribe un CSV con el mismo formato que dataset_limpio.csv.
func csvSintetico(t *testing.T, n int) string {
	t.Helper()
	rng := rand.New(rand.NewSource(7))
	tarifas := []string{"BT5B", "BT5D", "MT3"}
	base := []float64{20, 60, 900}
	var sb strings.Builder
	sb.WriteString("codigo_cliente,ubigeo,departamento,provincia,distrito,tarifa,estado_cliente,anio,mes,antiguedad_anios,consumo_kwh,facturacion\n")
	for i := 0; i < n; i++ {
		tt := rng.Intn(3)
		mes := 1 + rng.Intn(12)
		ant := rng.Float64() * 40
		kwh := base[tt] * (1 + 0.3*math.Sin(float64(mes))) * (1 + ant/40) * math.Exp(0.1*rng.NormFloat64())
		fmt.Fprintf(&sb, "%d,%d,PUNO,PUNO,D,%s,NORMAL,2023,%d,%.3f,%.2f,1\n", i+1, 210101+rng.Intn(5), tarifas[tt], mes, ant, kwh)
	}
	ruta := filepath.Join(t.TempDir(), "limpio.csv")
	if err := os.WriteFile(ruta, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return ruta
}

// levantarNodos inicia k nodos reales escuchando en puertos TCP locales.
func levantarNodos(t *testing.T, k int) []string {
	t.Helper()
	var addrs []string
	for i := 0; i < k; i++ {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		n := NewNode(fmt.Sprintf("node%d", i+1), 2)
		n.Logf = func(string, ...any) {}
		go n.Serve(ln)
		t.Cleanup(func() { ln.Close() })
		addrs = append(addrs, ln.Addr().String())
	}
	return addrs
}

func silencio() func(string, ...any) { return log.New(io.Discard, "", 0).Printf }

// El entrenamiento distribuido en 3 nodos por TCP debe llegar a los MISMOS
// pesos que el entrenamiento secuencial (solo cambia el orden de las sumas).
func TestDistribuidoIgualSecuencial(t *testing.T) {
	ruta := csvSintetico(t, 4000)
	cfg := JobConfig{JobID: "t1", DataPath: ruta, TestFrac: 0.2, Seed: 11, Epochs: 3, Batch: 256, LR: 0.01, Hidden: []int{16, 8}}

	coord := NewCoordinator(levantarNodos(t, 3), silencio())
	res, err := coord.Train(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}

	filas, _ := mlp.LoadCSV(ruta, 0)
	train, _ := mlp.Split(filas, cfg.TestFrac, cfg.Seed)
	enc := mlp.FitEncoder(train)
	ref := mlp.NewMLP([]int{enc.InputDim(), 16, 8, 1}, cfg.Seed)
	mlp.TrainSequential(ref, enc.Encode(train), mlp.TrainConfig{Epochs: cfg.Epochs, Batch: cfg.Batch, LR: cfg.LR, Seed: cfg.Seed})

	var dif float64
	for k := range ref.Params {
		dif = math.Max(dif, math.Abs(ref.Params[k]-res.Model.Params[k]))
	}
	if dif > 1e-8 {
		t.Fatalf("distribuido y secuencial difieren: %g", dif)
	}
	if len(res.Nodes) != 3 || res.Test.Log.R2 < 0.5 {
		t.Fatalf("resultado inesperado: nodos=%d R2=%.3f", len(res.Nodes), res.Test.Log.R2)
	}
	total := 0
	for _, n := range res.Nodes {
		total += n.ShardRows
	}
	if total != res.TrainRow {
		t.Fatalf("los shards no cubren train: %d vs %d", total, res.TrainRow)
	}
}

// Si un nodo no está disponible al inicio, el entrenamiento sigue con los demás.
func TestNodoCaidoSeOmite(t *testing.T) {
	ruta := csvSintetico(t, 1500)
	addrs := append(levantarNodos(t, 2), "127.0.0.1:1") // puerto sin servidor
	coord := NewCoordinator(addrs, silencio())
	res, err := coord.Train(context.Background(), JobConfig{JobID: "t2", DataPath: ruta, Seed: 1, Epochs: 2, Batch: 128, LR: 0.01, Hidden: []int{8}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Nodes) != 2 {
		t.Fatalf("esperaba 2 nodos activos, hay %d", len(res.Nodes))
	}
	st := coord.Probe()
	vivos := 0
	for _, s := range st {
		if s.Alive {
			vivos++
		}
	}
	if vivos != 2 {
		t.Fatalf("Probe: esperaba 2 vivos, hay %d (%+v)", vivos, st)
	}
}
