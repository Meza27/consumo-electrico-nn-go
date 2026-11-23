package api

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"consumo-electrico-nn-go/internal/cluster"
	"consumo-electrico-nn-go/internal/mlp"
)

// grupoDemanda resume a los suministros de un (distrito, tarifa): cuántos hay
// en un mes típico y su antigüedad media.
type grupoDemanda struct {
	Ubigeo      int
	Distrito    string
	Provincia   string
	Tarifa      string
	Suministros float64
	AntMedia    float64
}

// DistritoDemanda es la demanda estimada de un distrito en un mes.
type DistritoDemanda struct {
	Ubigeo      int     `json:"ubigeo"`
	Distrito    string  `json:"distrito"`
	Provincia   string  `json:"provincia"`
	Suministros int     `json:"suministros"`
	MWh         float64 `json:"mwh"`
	KWhPromedio float64 `json:"kwh_promedio"`
}

type demandaResultado struct {
	TotalMWh    float64           `json:"total_mwh"`
	Suministros int               `json:"suministros"`
	Distritos   []DistritoDemanda `json:"distritos"`
}

// cargarGrupos agrega el CSV limpio por (ubigeo, tarifa).
func cargarGrupos(ruta string) ([]grupoDemanda, error) {
	f, err := os.Open(ruta)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	if !sc.Scan() {
		return nil, fmt.Errorf("%s vacío", ruta)
	}
	col := map[string]int{}
	for i, c := range strings.Split(sc.Text(), ",") {
		col[c] = i
	}
	type acum struct {
		g     grupoDemanda
		n     int
		sumAn float64
	}
	grupos := map[string]*acum{}
	meses := map[string]bool{}
	for sc.Scan() {
		c := strings.Split(sc.Text(), ",")
		if len(c) < len(col) {
			continue
		}
		clave := c[col["ubigeo"]] + "|" + c[col["tarifa"]]
		a, ok := grupos[clave]
		if !ok {
			ub, _ := strconv.Atoi(c[col["ubigeo"]])
			a = &acum{g: grupoDemanda{Ubigeo: ub, Distrito: c[col["distrito"]], Provincia: c[col["provincia"]], Tarifa: c[col["tarifa"]]}}
			grupos[clave] = a
		}
		ant, _ := strconv.ParseFloat(c[col["antiguedad_anios"]], 64)
		a.n++
		a.sumAn += ant
		meses[c[col["anio"]]+c[col["mes"]]] = true
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	nm := float64(max(len(meses), 1))
	out := make([]grupoDemanda, 0, len(grupos))
	for _, a := range grupos {
		a.g.Suministros = float64(a.n) / nm
		a.g.AntMedia = a.sumAn / float64(a.n)
		out = append(out, a.g)
	}
	return out, nil
}

// calcularDemanda estima el consumo total de cada distrito en un mes:
// suma, por cada (distrito, tarifa), suministros × predicción del MLP.
func (s *Server) calcularDemanda(m *modeloActivo, mes int) demandaResultado {
	por := map[int]*DistritoDemanda{}
	var total float64
	var sum float64
	for _, g := range s.grupos {
		kwh := m.Saved.PredictKWh(g.Tarifa, g.Ubigeo, mes, g.AntMedia) * g.Suministros
		d, ok := por[g.Ubigeo]
		if !ok {
			d = &DistritoDemanda{Ubigeo: g.Ubigeo, Distrito: g.Distrito, Provincia: g.Provincia}
			por[g.Ubigeo] = d
		}
		d.MWh += kwh / 1000
		d.Suministros += int(g.Suministros + 0.5)
		total += kwh / 1000
		sum += g.Suministros
	}
	res := demandaResultado{TotalMWh: total, Suministros: int(sum + 0.5)}
	for _, d := range por {
		if d.Suministros > 0 {
			d.KWhPromedio = d.MWh * 1000 / float64(d.Suministros)
		}
		res.Distritos = append(res.Distritos, *d)
	}
	sort.Slice(res.Distritos, func(i, j int) bool { return res.Distritos[i].MWh > res.Distritos[j].MWh })
	return res
}

func savedDesdeMongoLike(r *cluster.Result) (*mlp.SavedModel, error) {
	return &mlp.SavedModel{Model: r.Model, Encoder: r.Encoder}, nil
}
