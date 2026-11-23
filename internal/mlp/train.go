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

// GradPool es un pool persistente de P goroutines que calcula el gradiente
// SUMADO de un conjunto de filas (patrón worker pool + cola de tareas +
// reducción). Lo usan TrainConcurrent (PC3) y cada nodo ML del clúster (PC4).
//
//  1. Los P workers viven mientras viva el pool. Cada uno tiene su propio
//     buffer de gradiente y sus propias activaciones (nada compartido escribible).
//  2. Compute parte las filas en trozos de TamTrozo y los pone en una cola
//     (channel compartido). Cada worker libre toma el siguiente trozo
//     (balanceo dinámico de carga).
//  3. Cada worker hace forward + backward leyendo los pesos (solo lectura),
//     suma el gradiente en SU buffer y avisa por el channel avisos.
//  4. Cuando llegan los avisos de todos los trozos (barrera), Compute suma los
//     P buffers (reducción) y limpia los buffers.
//
// Quien llama puede modificar m.Params ENTRE dos llamadas a Compute (los
// workers están bloqueados esperando trabajo), nunca durante una.
type GradPool struct {
	m      *MLP
	d      *Dataset
	grads  [][]float64
	total  []float64
	cola   chan trozo
	avisos chan hecho
}

// NewGradPool lanza los workers del pool.
func NewGradPool(m *MLP, d *Dataset, workers int) *GradPool {
	if workers < 1 {
		workers = 1
	}
	n := m.NumParams()
	p := &GradPool{m: m, d: d, grads: make([][]float64, workers), total: make([]float64, n),
		cola: make(chan trozo, 64), avisos: make(chan hecho, 64)}
	for w := 0; w < workers; w++ {
		p.grads[w] = make([]float64, n)
		go func(w int) {
			s := m.newScratch()
			g := p.grads[w]
			for t := range p.cola {
				var perdida float64
				for _, i := range t.filas {
					perdida += m.accumulateGrad(d.Cat[i*NumCat:(i+1)*NumCat], d.Num[i*NumNum:(i+1)*NumNum], d.Y[i], s, g)
				}
				p.avisos <- hecho{worker: w, perdida: perdida}
			}
		}(w)
	}
	return p
}

// Workers devuelve el número de goroutines del pool.
func (p *GradPool) Workers() int { return len(p.grads) }

// Compute devuelve el gradiente SUMADO (sin promediar) y la suma de errores
// cuadráticos de las filas indicadas. El slice devuelto pertenece al pool y
// se sobrescribe en la siguiente llamada.
func (p *GradPool) Compute(filas []int) ([]float64, float64) {
	// La cola tiene capacidad limitada: un goroutine aparte la llena mientras
	// esta goroutine recoge los avisos, así nunca se bloquean mutuamente.
	pendientes := (len(filas) + TamTrozo - 1) / TamTrozo
	go func() {
		for a := 0; a < len(filas); a += TamTrozo {
			p.cola <- trozo{filas: filas[a:min(a+TamTrozo, len(filas))]}
		}
	}()
	var perdida float64
	for ; pendientes > 0; pendientes-- {
		perdida += (<-p.avisos).perdida
	}
	zero(p.total)
	for _, g := range p.grads {
		for k, v := range g {
			p.total[k] += v
		}
		zero(g)
	}
	return p.total, perdida
}

// Close termina los workers.
func (p *GradPool) Close() { close(p.cola) }

// Adam es el optimizador exportado para el coordinador del clúster.
type Adam struct{ a *adam }

// NewAdam crea el optimizador para n parámetros.
func NewAdam(n int, lr float64) *Adam { return &Adam{newAdam(n, lr)} }

// Step aplica un paso con el gradiente PROMEDIO.
func (o *Adam) Step(params, grad []float64) { o.a.step(params, grad) }

// EpochOrder es el orden barajado de las filas de una época (el mismo que usan
// TrainSequential y TrainConcurrent, para poder comparar resultados).
func EpochOrder(n int, seed int64, epoch int) []int { return orden(n, seed, epoch) }

// TrainConcurrent entrena con DATA PARALLELISM dentro de un proceso: por cada
// mini-lote, el GradPool calcula el gradiente en paralelo y SOLO esta
// goroutine (el coordinador) promedia y actualiza los pesos con Adam.
//
// Mientras los workers leen los pesos, nadie los escribe; mientras el
// coordinador los escribe, todos los workers están esperando trabajo. Por eso
// no hay condiciones de carrera en los parámetros del modelo.
func TrainConcurrent(m *MLP, d *Dataset, cfg TrainConfig) []EpochStats {
	cfg.defaults()
	opt := newAdam(m.NumParams(), cfg.LR)
	pool := NewGradPool(m, d, cfg.Workers)
	defer pool.Close()

	var hist []EpochStats
	for ep := 1; ep <= cfg.Epochs; ep++ {
		inicio := time.Now()
		perm := orden(d.Len(), cfg.Seed, ep)
		var perdida float64
		for ini := 0; ini < len(perm); ini += cfg.Batch {
			fin := min(ini+cfg.Batch, len(perm))
			total, l := pool.Compute(perm[ini:fin])
			perdida += l
			escala := 1 / float64(fin-ini)
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
