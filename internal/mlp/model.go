package mlp

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
)

// MLP es una red feed-forward totalmente conectada:
//
//	entrada (Sizes[0]) -> oculta 1 (ReLU) -> ... -> oculta k (ReLU) -> salida (1, lineal)
//
// Todos los parámetros (pesos y sesgos de todas las capas) viven en un único
// arreglo Params. Así el gradiente tiene exactamente la misma forma, y sumar
// gradientes de varios workers o aplicar Adam es un simple bucle.
//
// La capa l tiene pesos W_l de tamaño in×out guardados por filas
// (W[i*out+o] conecta la neurona de entrada i con la de salida o) y sesgos b_l.
type MLP struct {
	Sizes  []int     `json:"sizes"`
	Params []float64 `json:"params"`

	wOff, bOff []int // dónde empieza W_l y b_l dentro de Params
}

// NewMLP crea la red con inicialización He (adecuada para ReLU).
func NewMLP(sizes []int, semilla int64) *MLP {
	m := &MLP{Sizes: append([]int(nil), sizes...)}
	m.layout()
	rng := rand.New(rand.NewSource(semilla))
	for l := 0; l < m.NumLayers(); l++ {
		in, out := m.Sizes[l], m.Sizes[l+1]
		fanIn := float64(in)
		if l == 0 {
			fanIn = NumCat + NumNum // con one-hot solo ~4 entradas están activas
		}
		std := math.Sqrt(2 / fanIn)
		w := m.Params[m.wOff[l] : m.wOff[l]+in*out]
		for i := range w {
			w[i] = rng.NormFloat64() * std
		}
	}
	return m
}

func (m *MLP) layout() {
	total := 0
	m.wOff = make([]int, m.NumLayers())
	m.bOff = make([]int, m.NumLayers())
	for l := 0; l < m.NumLayers(); l++ {
		m.wOff[l] = total
		total += m.Sizes[l] * m.Sizes[l+1]
		m.bOff[l] = total
		total += m.Sizes[l+1]
	}
	if len(m.Params) != total {
		m.Params = make([]float64, total)
	}
}

// NumLayers es la cantidad de capas con pesos.
func (m *MLP) NumLayers() int { return len(m.Sizes) - 1 }

// NumParams es la cantidad total de parámetros entrenables.
func (m *MLP) NumParams() int { return len(m.Params) }

// Clone hace una copia independiente.
func (m *MLP) Clone() *MLP {
	c := &MLP{Sizes: append([]int(nil), m.Sizes...), Params: append([]float64(nil), m.Params...)}
	c.layout()
	return c
}

// scratch son los buffers de UNA goroutine (activaciones y deltas). Cada
// worker tiene el suyo, así nunca comparten memoria escribible.
type scratch struct {
	z, a, delta [][]float64
}

func (m *MLP) newScratch() *scratch {
	s := &scratch{}
	for l := 1; l < len(m.Sizes); l++ {
		s.z = append(s.z, make([]float64, m.Sizes[l]))
		s.a = append(s.a, make([]float64, m.Sizes[l]))
		s.delta = append(s.delta, make([]float64, m.Sizes[l]))
	}
	return s
}

// forward calcula la salida para una fila. La primera capa aprovecha que la
// entrada es casi toda ceros (one-hot): en vez de multiplicar ~130 entradas,
// suma solo las filas de W_0 de las posiciones activas.
func (m *MLP) forward(cat []int32, num []float64, s *scratch) float64 {
	p := m.Params
	h := m.Sizes[1]
	numBase := m.Sizes[0] - NumNum
	z := s.z[0]
	copy(z, p[m.bOff[0]:m.bOff[0]+h])
	for _, c := range cat {
		if c >= 0 {
			w := p[m.wOff[0]+int(c)*h : m.wOff[0]+int(c)*h+h]
			for o := range z {
				z[o] += w[o]
			}
		}
	}
	for j, x := range num {
		w := p[m.wOff[0]+(numBase+j)*h : m.wOff[0]+(numBase+j)*h+h]
		for o := range z {
			z[o] += w[o] * x
		}
	}
	relu(z, s.a[0], m.NumLayers() > 1)

	for l := 1; l < m.NumLayers(); l++ {
		in, out := m.Sizes[l], m.Sizes[l+1]
		prev, z := s.a[l-1], s.z[l]
		copy(z, p[m.bOff[l]:m.bOff[l]+out])
		W := p[m.wOff[l] : m.wOff[l]+in*out]
		for i, ai := range prev {
			if ai == 0 {
				continue
			}
			wi := W[i*out : i*out+out]
			for o := range z {
				z[o] += ai * wi[o]
			}
		}
		relu(z, s.a[l], l < m.NumLayers()-1)
	}
	return s.a[len(s.a)-1][0]
}

// relu copia z en a aplicando ReLU (o identidad en la capa de salida).
func relu(z, a []float64, activar bool) {
	for i, v := range z {
		if activar && v < 0 {
			v = 0
		}
		a[i] = v
	}
}

// accumulateGrad hace forward + backward de una fila y SUMA su gradiente en
// grad (mismo layout que Params). Devuelve el error cuadrático de la fila.
// Pérdida por fila: 0.5 * (ŷ - y)².
func (m *MLP) accumulateGrad(cat []int32, num []float64, y float64, s *scratch, grad []float64) float64 {
	pred := m.forward(cat, num, s)
	diff := pred - y
	L := m.NumLayers()
	s.delta[L-1][0] = diff

	for l := L - 1; l >= 1; l-- {
		in, out := m.Sizes[l], m.Sizes[l+1]
		d, prev := s.delta[l], s.a[l-1]
		gW := grad[m.wOff[l] : m.wOff[l]+in*out]
		gb := grad[m.bOff[l] : m.bOff[l]+out]
		for o, v := range d {
			gb[o] += v
		}
		W := m.Params[m.wOff[l] : m.wOff[l]+in*out]
		dPrev, zPrev := s.delta[l-1], s.z[l-1]
		for i, ai := range prev {
			wi := W[i*out : i*out+out]
			gi := gW[i*out : i*out+out]
			var sum float64
			for o, v := range d {
				gi[o] += ai * v
				sum += wi[o] * v
			}
			if zPrev[i] <= 0 { // derivada de ReLU
				sum = 0
			}
			dPrev[i] = sum
		}
	}

	// Primera capa (entrada dispersa).
	h := m.Sizes[1]
	numBase := m.Sizes[0] - NumNum
	d := s.delta[0]
	gb := grad[m.bOff[0] : m.bOff[0]+h]
	for o, v := range d {
		gb[o] += v
	}
	for _, c := range cat {
		if c >= 0 {
			g := grad[m.wOff[0]+int(c)*h : m.wOff[0]+int(c)*h+h]
			for o, v := range d {
				g[o] += v
			}
		}
	}
	for j, x := range num {
		g := grad[m.wOff[0]+(numBase+j)*h : m.wOff[0]+(numBase+j)*h+h]
		for o, v := range d {
			g[o] += v * x
		}
	}
	return diff * diff
}

// SavedModel es lo que se guarda en disco: la red y el encoder (para que la
// API de la PC4 pueda cargar el modelo y predecir sin reentrenar).
type SavedModel struct {
	Model   *MLP           `json:"model"`
	Encoder *Encoder       `json:"encoder"`
	Metrics map[string]any `json:"metrics,omitempty"`
}

// Save escribe el modelo en JSON.
func Save(ruta string, sm SavedModel) error {
	datos, err := json.Marshal(sm)
	if err != nil {
		return err
	}
	return os.WriteFile(ruta, datos, 0o644)
}

// Load lee un modelo guardado con Save.
func Load(ruta string) (*SavedModel, error) {
	datos, err := os.ReadFile(ruta)
	if err != nil {
		return nil, err
	}
	var sm SavedModel
	if err := json.Unmarshal(datos, &sm); err != nil {
		return nil, err
	}
	sm.Model.layout()
	sm.Encoder.indexar()
	return &sm, nil
}

// PredictKWh predice el consumo (kWh) de una sola consulta.
func (sm *SavedModel) PredictKWh(tarifa string, ubigeo, mes int, antiguedad float64) float64 {
	cat, num := sm.Encoder.EncodeFeatures(tarifa, ubigeo, mes, antiguedad)
	return sm.Encoder.ToKWh(sm.Model.forward(cat[:], num[:], sm.Model.newScratch()))
}
