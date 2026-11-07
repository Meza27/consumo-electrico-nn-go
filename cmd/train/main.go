// Comando train: entrena el MLP de regresión de consumo eléctrico.
//
// Uso:
//
//	go run ./cmd/train -mode conc -workers 8          # entrenamiento concurrente
//	go run ./cmd/train -mode seq                      # entrenamiento secuencial
//	go run ./cmd/train -mode bench -bench-workers 1,2,4,8
//
// Guarda en -out: model.json (pesos + encoder), metrics.json y, en modo bench,
// benchmark.csv / benchmark.json con tiempos y speedup.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"consumo-electrico-nn-go/internal/mlp"
	"consumo-electrico-nn-go/internal/platform"
)

func main() {
	data := flag.String("data", "data/processed/dataset_limpio.csv", "CSV limpio (salida de cmd/pc3)")
	mode := flag.String("mode", "conc", "seq | conc | bench")
	workers := flag.Int("workers", runtime.NumCPU(), "goroutines para el entrenamiento concurrente")
	epochs := flag.Int("epochs", 5, "épocas")
	batch := flag.Int("batch", 4096, "tamaño del mini-lote")
	lr := flag.Float64("lr", 0.003, "tasa de aprendizaje (Adam)")
	hidden := flag.String("hidden", "64,32", "neuronas por capa oculta, separadas por coma")
	testFrac := flag.Float64("test", 0.2, "fracción de filas para test")
	seed := flag.Int64("seed", 42, "semilla")
	limit := flag.Int("limit", 0, "leer solo N filas (0 = todas)")
	out := flag.String("out", "results", "carpeta de salida")
	benchWorkers := flag.String("bench-workers", "1,2,4,8", "workers a probar en modo bench")
	benchEpochs := flag.Int("bench-epochs", 2, "épocas por corrida en modo bench")
	flag.Parse()
	platform.SinEcoQoS()

	check(os.MkdirAll(*out, 0o755))

	t0 := time.Now()
	filas, err := mlp.LoadCSV(*data, *limit)
	check(err)
	train, test := mlp.Split(filas, *testFrac, *seed)
	enc := mlp.FitEncoder(train) // SOLO con train
	dTr, dTe := enc.Encode(train), enc.Encode(test)
	filas = nil
	fmt.Printf("Datos: %d filas (train %d / test %d) cargadas en %.2f s\n",
		dTr.Len()+dTe.Len(), dTr.Len(), dTe.Len(), time.Since(t0).Seconds())

	sizes := []int{enc.InputDim()}
	for _, h := range strings.Split(*hidden, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(h))
		check(err)
		sizes = append(sizes, n)
	}
	sizes = append(sizes, 1)
	fmt.Printf("Red: %v (%d parámetros) | entrada: %d tarifas + %d distritos + 12 meses + antigüedad\n",
		sizes, mlp.NewMLP(sizes, *seed).NumParams(), len(enc.Tarifas), len(enc.Distritos))
	fmt.Printf("CPU: %d núcleos lógicos | %s/%s | %s\n", runtime.NumCPU(), runtime.GOOS, runtime.GOARCH, runtime.Version())

	cfg := mlp.TrainConfig{Epochs: *epochs, Batch: *batch, LR: *lr, Workers: *workers, Seed: *seed}

	if *mode == "bench" {
		bench(sizes, dTr, cfg, *benchWorkers, *benchEpochs, *seed, *out)
		return
	}

	cfg.OnEpoch = func(s mlp.EpochStats) {
		fmt.Printf("  época %2d | pérdida %.4f | %.2f s\n", s.Epoch, s.Loss, s.Segundos)
	}
	m := mlp.NewMLP(sizes, *seed)
	inicio := time.Now()
	var hist []mlp.EpochStats
	switch *mode {
	case "seq":
		fmt.Println("Entrenamiento SECUENCIAL")
		hist = mlp.TrainSequential(m, dTr, cfg)
	case "conc":
		fmt.Printf("Entrenamiento CONCURRENTE con %d workers\n", *workers)
		hist = mlp.TrainConcurrent(m, dTr, cfg)
	default:
		check(fmt.Errorf("modo desconocido %q", *mode))
	}
	segundos := time.Since(inicio).Seconds()

	evTr := mlp.Evaluate(m, enc, dTr, *workers)
	evTe := mlp.Evaluate(m, enc, dTe, *workers)
	base := mlp.Baseline(enc, dTe)
	fmt.Printf("Tiempo de entrenamiento: %.2f s\n\n", segundos)
	fmt.Println("                      RMSE (kWh)  MAE (kWh)  R² (kWh) | RMSE log  MAE log  R² log")
	fila := func(nombre string, e mlp.Evaluation) {
		fmt.Printf("  %-18s %10.2f %10.2f %9.4f | %8.4f %8.4f %7.4f\n", nombre,
			e.KWh.RMSE, e.KWh.MAE, e.KWh.R2, e.Log.RMSE, e.Log.MAE, e.Log.R2)
	}
	fila("train", evTr)
	fila("test", evTe)
	fila("baseline (test)", base)

	metricas := map[string]any{
		"modo": *mode, "workers": *workers, "sizes": sizes, "epochs": *epochs, "batch": *batch, "lr": *lr,
		"filas_train": dTr.Len(), "filas_test": dTe.Len(), "segundos_entrenamiento": segundos,
		"historial": hist, "train": evTr, "test": evTe, "baseline_test": base,
		"cpu_logicos": runtime.NumCPU(), "go": runtime.Version(),
	}
	escribirJSON(filepath.Join(*out, "metrics_"+*mode+".json"), metricas)
	check(mlp.Save(filepath.Join(*out, "model.json"), mlp.SavedModel{Model: m, Encoder: enc, Metrics: map[string]any{"test": evTe}}))
	fmt.Printf("\nGuardado: %s, %s\n", filepath.Join(*out, "model.json"), filepath.Join(*out, "metrics_"+*mode+".json"))

	// Ejemplo de predicción (lo mismo que hará la API en la PC4).
	sm, err := mlp.Load(filepath.Join(*out, "model.json"))
	check(err)
	for _, q := range []struct {
		t       string
		ub, mes int
		ant     float64
	}{{"BT5B", 211101, 7, 10}, {"BT5B", 210101, 1, 25}, {"BT5D", 211101, 7, 10}} {
		fmt.Printf("  predicción tarifa=%s ubigeo=%d mes=%d antigüedad=%.0f años -> %.1f kWh\n",
			q.t, q.ub, q.mes, q.ant, sm.PredictKWh(q.t, q.ub, q.mes, q.ant))
	}
}

type corrida struct {
	Modo           string  `json:"modo"`
	Workers        int     `json:"workers"`
	Segundos       float64 `json:"segundos"`
	SegundosEpoca  float64 `json:"segundos_por_epoca"`
	Speedup        float64 `json:"speedup"`
	Eficiencia     float64 `json:"eficiencia"`
	PerdidaFinal   float64 `json:"perdida_final"`
	MaxDifPesosSeq float64 `json:"max_dif_pesos_vs_seq"`
}

// bench entrena el mismo modelo (misma inicialización y mismo orden de datos)
// en secuencial y con distintos números de workers, y compara tiempos.
func bench(sizes []int, d *mlp.Dataset, cfg mlp.TrainConfig, lista string, epocas int, seed int64, out string) {
	cfg.Epochs = epocas
	fmt.Printf("\nBENCHMARK: %d épocas, lote %d, %d filas de train\n", epocas, cfg.Batch, d.Len())
	medir := func(f func(*mlp.MLP, *mlp.Dataset, mlp.TrainConfig) []mlp.EpochStats) (*mlp.MLP, float64, float64) {
		m := mlp.NewMLP(sizes, seed)
		runtime.GC()
		ini := time.Now()
		h := f(m, d, cfg)
		return m, time.Since(ini).Seconds(), h[len(h)-1].Loss
	}
	ref, tSeq, lSeq := medir(mlp.TrainSequential)
	res := []corrida{{Modo: "secuencial", Workers: 1, Segundos: tSeq, SegundosEpoca: tSeq / float64(epocas), Speedup: 1, Eficiencia: 1, PerdidaFinal: lSeq}}
	for _, s := range strings.Split(lista, ",") {
		w, err := strconv.Atoi(strings.TrimSpace(s))
		check(err)
		cfg.Workers = w
		m, t, l := medir(mlp.TrainConcurrent)
		var dif float64
		for k := range m.Params {
			if v := m.Params[k] - ref.Params[k]; v > dif || -v > dif {
				dif = max(v, -v)
			}
		}
		res = append(res, corrida{Modo: "concurrente", Workers: w, Segundos: t, SegundosEpoca: t / float64(epocas),
			Speedup: tSeq / t, Eficiencia: tSeq / t / float64(w), PerdidaFinal: l, MaxDifPesosSeq: dif})
	}

	fmt.Println("\n  modo         workers  tiempo(s)  s/época  speedup  eficiencia  pérdida   máx|Δpesos| vs seq")
	var csv strings.Builder
	csv.WriteString("modo,workers,segundos,segundos_por_epoca,speedup,eficiencia,perdida_final,max_dif_pesos_vs_seq\n")
	for _, r := range res {
		fmt.Printf("  %-12s %7d %10.2f %8.2f %8.2fx %10.0f%% %8.4f   %.1e\n",
			r.Modo, r.Workers, r.Segundos, r.SegundosEpoca, r.Speedup, r.Eficiencia*100, r.PerdidaFinal, r.MaxDifPesosSeq)
		fmt.Fprintf(&csv, "%s,%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.3e\n",
			r.Modo, r.Workers, r.Segundos, r.SegundosEpoca, r.Speedup, r.Eficiencia, r.PerdidaFinal, r.MaxDifPesosSeq)
	}
	check(os.WriteFile(filepath.Join(out, "benchmark.csv"), []byte(csv.String()), 0o644))
	escribirJSON(filepath.Join(out, "benchmark.json"), map[string]any{
		"cpu_logicos": runtime.NumCPU(), "go": runtime.Version(), "os": runtime.GOOS,
		"batch": cfg.Batch, "epocas": epocas, "filas_train": d.Len(), "sizes": sizes, "corridas": res,
	})
	fmt.Printf("\nGuardado: %s\n", filepath.Join(out, "benchmark.csv"))
}

func escribirJSON(ruta string, v any) {
	datos, err := json.MarshalIndent(v, "", "  ")
	check(err)
	check(os.WriteFile(ruta, datos, 0o644))
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
