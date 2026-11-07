// Package mlp implementa un perceptrón multicapa (MLP) de regresión para
// predecir el consumo eléctrico mensual (kWh) de un cliente de Electro Puno,
// con entrenamiento secuencial y entrenamiento concurrente por data parallelism.
//
// Features (entrada): tarifa, distrito (ubigeo), mes y antigüedad del cliente.
// La FACTURACIÓN se excluye a propósito: se calcula a partir del consumo, así
// que usarla sería fuga de información (el modelo "vería" la respuesta).
package mlp

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Row es una fila del CSV limpio con solo las columnas que usa el modelo.
type Row struct {
	Tarifa     string
	Ubigeo     int
	Mes        int
	Antiguedad float64
	ConsumoKWh float64
}

// LoadCSV lee data/processed/dataset_limpio.csv. Si limite > 0 lee solo las
// primeras `limite` filas (útil para pruebas rápidas).
func LoadCSV(ruta string, limite int) ([]Row, error) {
	f, err := os.Open(ruta)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	if !sc.Scan() {
		return nil, fmt.Errorf("%s está vacío", ruta)
	}
	col := map[string]int{}
	for i, c := range strings.Split(sc.Text(), ",") {
		col[c] = i
	}
	for _, c := range []string{"ubigeo", "tarifa", "mes", "antiguedad_anios", "consumo_kwh"} {
		if _, ok := col[c]; !ok {
			return nil, fmt.Errorf("falta la columna %q", c)
		}
	}
	tarifas := map[string]string{} // para no guardar millones de copias del mismo texto
	var filas []Row
	for sc.Scan() {
		if limite > 0 && len(filas) >= limite {
			break
		}
		c := strings.Split(sc.Text(), ",")
		ub, e1 := strconv.Atoi(c[col["ubigeo"]])
		mes, e2 := strconv.Atoi(c[col["mes"]])
		ant, e3 := strconv.ParseFloat(c[col["antiguedad_anios"]], 64)
		kwh, e4 := strconv.ParseFloat(c[col["consumo_kwh"]], 64)
		if e1 != nil || e2 != nil || e3 != nil || e4 != nil {
			return nil, fmt.Errorf("fila %d mal formada", len(filas)+2)
		}
		t := c[col["tarifa"]]
		if s, ok := tarifas[t]; ok {
			t = s
		} else {
			tarifas[t] = t
		}
		filas = append(filas, Row{Tarifa: t, Ubigeo: ub, Mes: mes, Antiguedad: ant, ConsumoKWh: kwh})
	}
	return filas, sc.Err()
}

// Split separa las filas al azar en train y test (fracTest en test), siempre
// igual para la misma semilla.
func Split(filas []Row, fracTest float64, semilla int64) (train, test []Row) {
	idx := rand.New(rand.NewSource(semilla)).Perm(len(filas))
	nTest := int(float64(len(filas)) * fracTest)
	test = make([]Row, 0, nTest)
	train = make([]Row, 0, len(filas)-nTest)
	for i, j := range idx {
		if i < nTest {
			test = append(test, filas[j])
		} else {
			train = append(train, filas[j])
		}
	}
	return train, test
}

// Encoder convierte filas en vectores de entrada. Se AJUSTA SOLO CON TRAIN:
// los vocabularios y las medias/desviaciones salen del conjunto de
// entrenamiento, para que el test no influya en nada (evita fuga de datos).
//
// Vector de entrada (dimensión InputDim):
//
//	[ one-hot tarifa | one-hot distrito | one-hot mes (12) | antigüedad estandarizada ]
//
// Objetivo: y = (log(1+kWh) - YMean) / YStd. Se usa logaritmo porque el
// consumo tiene una cola enorme (mediana 16 kWh, máximo ~500 000 kWh).
type Encoder struct {
	Tarifas   []string `json:"tarifas"`
	Distritos []int    `json:"distritos"` // ubigeos
	AntMean   float64  `json:"antiguedad_media"`
	AntStd    float64  `json:"antiguedad_std"`
	YMean     float64  `json:"y_media_log"`
	YStd      float64  `json:"y_std_log"`

	idxTarifa   map[string]int
	idxDistrito map[int]int
}

// NumCat es cuántas variables categóricas tiene cada fila (tarifa, distrito, mes).
const NumCat = 3

// NumNum es cuántas variables numéricas tiene cada fila (antigüedad).
const NumNum = 1

// FitEncoder calcula vocabularios y estadísticas usando solo `train`.
func FitEncoder(train []Row) *Encoder {
	ts, ds := map[string]bool{}, map[int]bool{}
	var sa, saa, sy, syy float64
	for _, r := range train {
		ts[r.Tarifa] = true
		ds[r.Ubigeo] = true
		sa += r.Antiguedad
		saa += r.Antiguedad * r.Antiguedad
		y := math.Log1p(r.ConsumoKWh)
		sy += y
		syy += y * y
	}
	e := &Encoder{}
	for t := range ts {
		e.Tarifas = append(e.Tarifas, t)
	}
	for d := range ds {
		e.Distritos = append(e.Distritos, d)
	}
	sort.Strings(e.Tarifas)
	sort.Ints(e.Distritos)
	n := float64(len(train))
	e.AntMean, e.AntStd = sa/n, stdDe(sa, saa, n)
	e.YMean, e.YStd = sy/n, stdDe(sy, syy, n)
	e.indexar()
	return e
}

func stdDe(s, ss, n float64) float64 {
	v := ss/n - (s/n)*(s/n)
	if v < 1e-12 {
		return 1
	}
	return math.Sqrt(v)
}

func (e *Encoder) indexar() {
	e.idxTarifa = map[string]int{}
	for i, t := range e.Tarifas {
		e.idxTarifa[t] = i
	}
	e.idxDistrito = map[int]int{}
	for i, d := range e.Distritos {
		e.idxDistrito[d] = i
	}
}

// InputDim es el tamaño del vector de entrada.
func (e *Encoder) InputDim() int { return len(e.Tarifas) + len(e.Distritos) + 12 + NumNum }

// Dataset guarda las entradas codificadas en arreglos planos (rápido y sin
// punteros). Para la fila i:
//   - Cat[i*NumCat : (i+1)*NumCat] = posiciones del vector de entrada que valen 1
//     (-1 si la categoría no se vio en train: esa parte queda en ceros).
//   - Num[i*NumNum : (i+1)*NumNum] = valores numéricos ya estandarizados.
//   - Y[i] = objetivo normalizado; KWh[i] = consumo real (para las métricas).
type Dataset struct {
	Cat []int32
	Num []float64
	Y   []float64
	KWh []float64
}

// Len devuelve la cantidad de filas.
func (d *Dataset) Len() int { return len(d.Y) }

// Encode transforma filas en un Dataset usando lo aprendido de train.
func (e *Encoder) Encode(filas []Row) *Dataset {
	n := len(filas)
	d := &Dataset{
		Cat: make([]int32, n*NumCat),
		Num: make([]float64, n*NumNum),
		Y:   make([]float64, n),
		KWh: make([]float64, n),
	}
	for i, r := range filas {
		cat, num := e.EncodeFeatures(r.Tarifa, r.Ubigeo, r.Mes, r.Antiguedad)
		copy(d.Cat[i*NumCat:], cat[:])
		copy(d.Num[i*NumNum:], num[:])
		d.Y[i] = (math.Log1p(r.ConsumoKWh) - e.YMean) / e.YStd
		d.KWh[i] = r.ConsumoKWh
	}
	return d
}

// EncodeFeatures codifica una sola consulta (lo usará la API de la PC4).
func (e *Encoder) EncodeFeatures(tarifa string, ubigeo, mes int, antiguedad float64) ([NumCat]int32, [NumNum]float64) {
	if e.idxTarifa == nil {
		e.indexar()
	}
	nt, nd := len(e.Tarifas), len(e.Distritos)
	cat := [NumCat]int32{-1, -1, -1}
	if i, ok := e.idxTarifa[tarifa]; ok {
		cat[0] = int32(i)
	}
	if i, ok := e.idxDistrito[ubigeo]; ok {
		cat[1] = int32(nt + i)
	}
	if mes >= 1 && mes <= 12 {
		cat[2] = int32(nt + nd + mes - 1)
	}
	return cat, [NumNum]float64{(antiguedad - e.AntMean) / e.AntStd}
}

// ToKWh pasa una salida normalizada de la red a kWh.
func (e *Encoder) ToKWh(yNorm float64) float64 {
	v := math.Expm1(yNorm*e.YStd + e.YMean)
	if v < 0 {
		return 0
	}
	return v
}
