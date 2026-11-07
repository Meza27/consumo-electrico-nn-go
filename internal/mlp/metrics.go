package mlp

import "math"

// Metrics son las métricas de regresión.
type Metrics struct {
	RMSE float64 `json:"rmse"`
	MAE  float64 `json:"mae"`
	R2   float64 `json:"r2"`
}

// Regression calcula RMSE, MAE y R² entre valores reales y predichos.
//
//	RMSE = sqrt(promedio((y-ŷ)²))   castiga mucho los errores grandes
//	MAE  = promedio(|y-ŷ|)           error típico, en las mismas unidades
//	R²   = 1 - SSE/SST               1 = perfecto, 0 = igual que predecir el promedio
func Regression(real, pred []float64) Metrics {
	n := float64(len(real))
	var media float64
	for _, y := range real {
		media += y
	}
	media /= n
	var sse, sae, sst float64
	for i, y := range real {
		e := y - pred[i]
		sse += e * e
		sae += math.Abs(e)
		sst += (y - media) * (y - media)
	}
	r2 := 0.0
	if sst > 0 {
		r2 = 1 - sse/sst
	}
	return Metrics{RMSE: math.Sqrt(sse / n), MAE: sae / n, R2: r2}
}

// Evaluation agrupa las métricas en kWh (lo que importa al usuario) y en
// escala logarítmica (la escala en la que se entrenó).
type Evaluation struct {
	KWh Metrics `json:"kwh"`
	Log Metrics `json:"log1p"`
}

// Evaluate predice todo el dataset (en paralelo) y calcula las métricas.
func Evaluate(m *MLP, e *Encoder, d *Dataset, workers int) Evaluation {
	predNorm := Predict(m, d, workers)
	realK, predK := d.KWh, make([]float64, d.Len())
	realL, predL := make([]float64, d.Len()), make([]float64, d.Len())
	for i, p := range predNorm {
		predK[i] = e.ToKWh(p)
		realL[i] = math.Log1p(realK[i])
		predL[i] = p*e.YStd + e.YMean
	}
	return Evaluation{KWh: Regression(realK, predK), Log: Regression(realL, predL)}
}

// Baseline evalúa el modelo trivial que siempre predice la mediana de train
// (en log, la media de log1p): sirve para saber si la red aprende algo.
func Baseline(e *Encoder, d *Dataset) Evaluation {
	pk := e.ToKWh(0)
	predK := make([]float64, d.Len())
	realL, predL := make([]float64, d.Len()), make([]float64, d.Len())
	for i := range predK {
		predK[i] = pk
		realL[i] = math.Log1p(d.KWh[i])
		predL[i] = e.YMean
	}
	return Evaluation{KWh: Regression(d.KWh, predK), Log: Regression(realL, predL)}
}
