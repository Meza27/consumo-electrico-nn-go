package mlp

import (
	"math"
	"math/rand"
	"time"
)

// TrainConfig son los hiperparámetros del entrenamiento.
type TrainConfig struct {
	Epochs  int
	Batch   int     // filas por mini-lote
	LR      float64 // tasa de aprendizaje de Adam
	Workers int     // goroutines (solo para la versión concurrente)
	Seed    int64
	// OnEpoch se llama al terminar cada época (puede ser nil).
	OnEpoch func(EpochStats)
}

// EpochStats resume una época.
type EpochStats struct {
	Epoch    int     `json:"epoch"`
	Loss     float64 `json:"loss"` // MSE en la escala normalizada
	Segundos float64 `json:"segundos"`
}

// adam implementa el optimizador Adam. Lo usa SOLO el coordinador, así que no
// necesita sincronización.
type adam struct {
	lr, b1, b2, eps float64
	m, v            []float64
	t               int
}

func newAdam(n int, lr float64) *adam {
	return &adam{lr: lr, b1: 0.9, b2: 0.999, eps: 1e-8, m: make([]float64, n), v: make([]float64, n)}
}

// step actualiza params con el gradiente PROMEDIO del lote.
func (o *adam) step(params, grad []float64) {
	o.t++
	c1 := 1 - math.Pow(o.b1, float64(o.t))
	c2 := 1 - math.Pow(o.b2, float64(o.t))
	for i, g := range grad {
		o.m[i] = o.b1*o.m[i] + (1-o.b1)*g
		o.v[i] = o.b2*o.v[i] + (1-o.b2)*g*g
		params[i] -= o.lr * (o.m[i] / c1) / (math.Sqrt(o.v[i]/c2) + o.eps)
	}
}

// orden devuelve el orden (barajado) de las filas para una época. Las dos
// versiones usan exactamente el mismo orden, así se pueden comparar.
func orden(n int, seed int64, epoch int) []int {
	return rand.New(rand.NewSource(seed + int64(epoch)*7919)).Perm(n)
}

func zero(v []float64) {
	for i := range v {
		v[i] = 0
	}
}

func (cfg *TrainConfig) defaults() {
	if cfg.Epochs < 1 {
		cfg.Epochs = 1
	}
	if cfg.Batch < 1 {
		cfg.Batch = 1024
	}
	if cfg.LR <= 0 {
		cfg.LR = 1e-3
	}
	if cfg.Workers < 1 {
		cfg.Workers = 1
	}
}

// TrainSequential entrena con mini-batch Adam en UNA sola goroutine: para cada
// lote recorre sus filas una por una, acumula el gradiente y actualiza.
func TrainSequential(m *MLP, d *Dataset, cfg TrainConfig) []EpochStats {
	cfg.defaults()
	opt := newAdam(m.NumParams(), cfg.LR)
	grad := make([]float64, m.NumParams())
	s := m.newScratch()
	var hist []EpochStats

	for ep := 1; ep <= cfg.Epochs; ep++ {
		inicio := time.Now()
		perm := orden(d.Len(), cfg.Seed, ep)
		var perdida float64
		for ini := 0; ini < len(perm); ini += cfg.Batch {
			fin := min(ini+cfg.Batch, len(perm))
			zero(grad)
			for _, i := range perm[ini:fin] {
				perdida += m.accumulateGrad(d.Cat[i*NumCat:(i+1)*NumCat], d.Num[i*NumNum:(i+1)*NumNum], d.Y[i], s, grad)
			}
			escala := 1 / float64(fin-ini)
			for k := range grad {
				grad[k] *= escala
			}
			opt.step(m.Params, grad)
		}
		st := EpochStats{Epoch: ep, Loss: perdida / float64(d.Len()), Segundos: time.Since(inicio).Seconds()}
		hist = append(hist, st)
		if cfg.OnEpoch != nil {
			cfg.OnEpoch(st)
		}
	}
	return hist
}

// trozo es una tarea: "calcula el gradiente de estas filas del lote".
type trozo struct {
	filas []int
}

// hecho es el aviso de un worker al terminar un trozo. El gradiente se queda
// acumulado en el buffer propio del worker (grads[w]); el coordinador lo lee
// recién cuando recibió el aviso de TODOS los trozos del lote, momento en que
// todos los workers están bloqueados esperando el siguiente trozo. El channel
// garantiza ese orden (relación happens-before): no hay carreras sin mutex.
type hecho struct {
	worker  int
	perdida float64
}

// TamTrozo es cuántas filas tiene cada tarea de la cola. Trozos pequeños
// permiten balanceo dinámico: un worker en un núcleo rápido (P-core) toma más
// trozos que uno en un núcleo lento (E-core), y nadie se queda esperando.
const TamTrozo = 128

// TrainConcurrent entrena con DATA PARALLELISM usando goroutines y channels
// (patrón worker pool + cola de tareas + reducción):
//
//  1. Se lanzan P workers que viven toda la ejecución. Cada uno tiene su propio
//     buffer de gradiente y sus propias activaciones (nada compartido escribible).
//  2. Por cada mini-lote, el coordinador lo parte en trozos de TamTrozo filas y
//     los pone en una cola (channel compartido). Cada worker libre toma el
//     siguiente trozo (balanceo dinámico de carga).
//  3. Cada worker hace forward + backward de sus filas leyendo los pesos
//     actuales (solo lectura), suma el gradiente en SU buffer y avisa.
//  4. Cuando llegaron los avisos de todos los trozos (barrera), el coordinador
//     SUMA los P gradientes parciales (reducción tipo AllReduce), los promedia
//     y SOLO ÉL actualiza los pesos con Adam. Luego limpia los buffers.
//
// Mientras los workers leen los pesos, nadie los escribe; mientras el
// coordinador los escribe, todos los workers están esperando trabajo. Por eso
// no hay condiciones de carrera en los parámetros del modelo.
func TrainConcurrent(m *MLP, d *Dataset, cfg TrainConfig) []EpochStats {
	cfg.defaults()
	P := cfg.Workers
	n := m.NumParams()
	opt := newAdam(n, cfg.LR)
	grads := make([][]float64, P)
	cola := make(chan trozo, cfg.Batch/TamTrozo+1)
	avisos := make(chan hecho, cfg.Batch/TamTrozo+1)

	for w := 0; w < P; w++ {
		grads[w] = make([]float64, n)
		go func(w int) {
			s := m.newScratch()
			g := grads[w]
			for t := range cola {
				var perdida float64
				for _, i := range t.filas {
					perdida += m.accumulateGrad(d.Cat[i*NumCat:(i+1)*NumCat], d.Num[i*NumNum:(i+1)*NumNum], d.Y[i], s, g)
				}
				avisos <- hecho{worker: w, perdida: perdida}
			}
		}(w)
	}
	defer close(cola) // al terminar, los workers salen de su bucle

	total := make([]float64, n)
	var hist []EpochStats
	for ep := 1; ep <= cfg.Epochs; ep++ {
		inicio := time.Now()
		perm := orden(d.Len(), cfg.Seed, ep)
		var perdida float64
		for ini := 0; ini < len(perm); ini += cfg.Batch {
			fin := min(ini+cfg.Batch, len(perm))
			lote := perm[ini:fin]
			// Scatter: el lote se reparte en trozos dentro de la cola.
			pendientes := 0
			for a := 0; a < len(lote); a += TamTrozo {
				cola <- trozo{filas: lote[a:min(a+TamTrozo, len(lote))]}
				pendientes++
			}
			// Barrera: esperar el aviso de cada trozo.
			for ; pendientes > 0; pendientes-- {
				perdida += (<-avisos).perdida
			}
			// Reducción: sumar los gradientes parciales de los P workers.
			zero(total)
			for w := 0; w < P; w++ {
				for k, v := range grads[w] {
					total[k] += v
				}
				zero(grads[w])
			}
			escala := 1 / float64(len(lote))
			for k := range total {
				total[k] *= escala
			}
			opt.step(m.Params, total) // única escritura de los pesos
		}
		st := EpochStats{Epoch: ep, Loss: perdida / float64(d.Len()), Segundos: time.Since(inicio).Seconds()}
		hist = append(hist, st)
		if cfg.OnEpoch != nil {
			cfg.OnEpoch(st)
		}
	}
	return hist
}

// Predict devuelve las predicciones (escala normalizada) usando `workers`
// goroutines: cada una predice un trozo contiguo y escribe solo en su parte
// del slice de salida (posiciones disjuntas => sin carreras).
func Predict(m *MLP, d *Dataset, workers int) []float64 {
	if workers < 1 {
		workers = 1
	}
	out := make([]float64, d.Len())
	listo := make(chan struct{}, workers)
	for w := 0; w < workers; w++ {
		a, b := w*d.Len()/workers, (w+1)*d.Len()/workers
		go func(a, b int) {
			s := m.newScratch()
			for i := a; i < b; i++ {
				out[i] = m.forward(d.Cat[i*NumCat:(i+1)*NumCat], d.Num[i*NumNum:(i+1)*NumNum], s)
			}
			listo <- struct{}{}
		}(a, b)
	}
	for w := 0; w < workers; w++ {
		<-listo
	}
	return out
}
