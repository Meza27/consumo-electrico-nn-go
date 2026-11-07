package mlp

import (
	"math"
	"math/rand"
	"testing"
)

// filas sintéticas: el consumo depende de la tarifa, el mes y la antigüedad.
func filasSinteticas(n int, semilla int64) []Row {
	rng := rand.New(rand.NewSource(semilla))
	tarifas := []string{"BT5B", "BT5D", "MT3"}
	base := []float64{20, 60, 900}
	filas := make([]Row, n)
	for i := range filas {
		t := rng.Intn(3)
		mes := 1 + rng.Intn(12)
		ant := rng.Float64() * 40
		kwh := base[t] * (1 + 0.3*math.Sin(float64(mes))) * (1 + ant/40) * math.Exp(0.1*rng.NormFloat64())
		filas[i] = Row{Tarifa: tarifas[t], Ubigeo: 210101 + rng.Intn(5), Mes: mes, Antiguedad: ant, ConsumoKWh: kwh}
	}
	return filas
}

// El gradiente de backpropagation debe coincidir con el gradiente numérico
// (diferencias finitas). Si esto falla, el backward está mal.
func TestGradienteNumerico(t *testing.T) {
	filas := filasSinteticas(5, 1)
	e := FitEncoder(filas)
	d := e.Encode(filas)
	m := NewMLP([]int{e.InputDim(), 6, 4, 1}, 3)
	s := m.newScratch()
	grad := make([]float64, m.NumParams())
	perdida := func() float64 {
		var tot float64
		for i := 0; i < d.Len(); i++ {
			p := m.forward(d.Cat[i*NumCat:(i+1)*NumCat], d.Num[i*NumNum:(i+1)*NumNum], s)
			tot += 0.5 * (p - d.Y[i]) * (p - d.Y[i])
		}
		return tot
	}
	for i := 0; i < d.Len(); i++ {
		m.accumulateGrad(d.Cat[i*NumCat:(i+1)*NumCat], d.Num[i*NumNum:(i+1)*NumNum], d.Y[i], s, grad)
	}
	const h = 1e-6
	for k := range m.Params {
		orig := m.Params[k]
		m.Params[k] = orig + h
		mas := perdida()
		m.Params[k] = orig - h
		menos := perdida()
		m.Params[k] = orig
		num := (mas - menos) / (2 * h)
		if math.Abs(num-grad[k]) > 1e-5*math.Max(1, math.Abs(num)) {
			t.Fatalf("parámetro %d: backprop=%g numérico=%g", k, grad[k], num)
		}
	}
}

// Secuencial y concurrente (con cualquier número de workers) deben llegar a
// los mismos pesos: solo cambia el orden de las sumas de punto flotante.
func TestSecuencialIgualConcurrente(t *testing.T) {
	filas := filasSinteticas(3000, 2)
	e := FitEncoder(filas)
	d := e.Encode(filas)
	cfg := TrainConfig{Epochs: 3, Batch: 256, LR: 0.01, Seed: 9}
	ref := NewMLP([]int{e.InputDim(), 16, 8, 1}, 5)
	TrainSequential(ref, d, cfg)
	for _, w := range []int{1, 2, 3, 8} {
		m := NewMLP([]int{e.InputDim(), 16, 8, 1}, 5)
		cfg.Workers = w
		TrainConcurrent(m, d, cfg)
		for k := range m.Params {
			if math.Abs(m.Params[k]-ref.Params[k]) > 1e-8 {
				t.Fatalf("workers=%d: parámetro %d difiere: %g vs %g", w, k, m.Params[k], ref.Params[k])
			}
		}
	}
}

// La red debe aprender: el R² en test tiene que superar claramente al baseline.
func TestAprende(t *testing.T) {
	train, test := Split(filasSinteticas(6000, 3), 0.2, 1)
	e := FitEncoder(train)
	dTr, dTe := e.Encode(train), e.Encode(test)
	m := NewMLP([]int{e.InputDim(), 16, 8, 1}, 1)
	hist := TrainConcurrent(m, dTr, TrainConfig{Epochs: 15, Batch: 128, LR: 0.01, Workers: 4, Seed: 1})
	if hist[len(hist)-1].Loss >= hist[0].Loss {
		t.Fatalf("la pérdida no bajó: %+v", hist)
	}
	ev := Evaluate(m, e, dTe, 4)
	if ev.Log.R2 < 0.8 {
		t.Fatalf("R² (log) muy bajo: %+v", ev)
	}
}

// El encoder se ajusta solo con train: una categoría que aparece solo en test
// queda como "desconocida" (-1) y no cambia el vocabulario.
func TestEncoderSoloTrain(t *testing.T) {
	train := []Row{{Tarifa: "BT5B", Ubigeo: 210101, Mes: 1, Antiguedad: 10, ConsumoKWh: 50}, {Tarifa: "BT5B", Ubigeo: 210101, Mes: 2, Antiguedad: 20, ConsumoKWh: 70}}
	e := FitEncoder(train)
	if e.AntMean != 15 || len(e.Tarifas) != 1 || len(e.Distritos) != 1 {
		t.Fatalf("encoder mal ajustado: %+v", e)
	}
	d := e.Encode([]Row{{Tarifa: "MT2", Ubigeo: 999999, Mes: 3, Antiguedad: 15, ConsumoKWh: 1}})
	if d.Cat[0] != -1 || d.Cat[1] != -1 || d.Num[0] != 0 {
		t.Fatalf("categorías nuevas mal codificadas: %+v", d)
	}
}

func TestMetricas(t *testing.T) {
	m := Regression([]float64{1, 2, 3}, []float64{1, 2, 3})
	if m.RMSE != 0 || m.MAE != 0 || m.R2 != 1 {
		t.Fatalf("%+v", m)
	}
	m = Regression([]float64{1, 2, 3}, []float64{2, 2, 2})
	if math.Abs(m.R2) > 1e-12 || math.Abs(m.MAE-2.0/3) > 1e-12 {
		t.Fatalf("%+v", m)
	}
}
