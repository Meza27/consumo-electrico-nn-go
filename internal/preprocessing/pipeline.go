package preprocessing

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// Config son los parámetros de la limpieza.
type Config struct {
	DirEntrada   string // carpeta con los CSV crudos
	Salida       string // CSV limpio que se va a escribir
	ResumenJSON  string // resumen en JSON ("" para no escribirlo)
	Workers      int    // goroutines que limpian
	TamanoBloque int    // líneas por bloque
}

// Summary es el resumen de la limpieza.
type Summary struct {
	Archivos         []string `json:"archivos"`
	FilasLeidas      int      `json:"filas_leidas"`
	FilasValidas     int      `json:"filas_validas"`
	FilasDescartadas int      `json:"filas_descartadas"`
	// Filas válidas con facturación negativa (ajustes). Se conservan; solo se cuentan.
	FacturacionNegativa int            `json:"facturacion_negativa_conservadas"`
	Motivos             map[string]int `json:"motivos_descarte"`
	Workers             int            `json:"workers"`
	SegundosTotales     float64        `json:"segundos"`
}

// Run ejecuta todo el pipeline: productor -> workers -> reductor.
func Run(cfg Config) (Summary, error) {
	inicio := time.Now()
	if cfg.Workers < 1 {
		cfg.Workers = 1
	}
	if cfg.TamanoBloque < 1 {
		cfg.TamanoBloque = 5000
	}

	archivos, err := ListarCSV(cfg.DirEntrada)
	if err != nil {
		return Summary{}, err
	}
	if len(archivos) == 0 {
		return Summary{}, fmt.Errorf("no hay archivos .csv en %s", cfg.DirEntrada)
	}

	if err := os.MkdirAll(filepath.Dir(cfg.Salida), 0o755); err != nil {
		return Summary{}, err
	}
	f, err := os.Create(cfg.Salida)
	if err != nil {
		return Summary{}, err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	bloques := make(chan Block, cfg.Workers*2)
	resultados := make(chan BlockResult, cfg.Workers*2)
	errLectura := make(chan error, 1)

	// 1) Productor: lee los archivos y llena el channel de bloques.
	go func() {
		errLectura <- StreamBlocks(archivos, cfg.TamanoBloque, bloques)
		close(bloques)
	}()

	// 2) Workers: cada uno saca bloques y los limpia, sin compartir nada.
	// (Se usa un channel "listo" en vez de WaitGroup para cerrar resultados.)
	listo := make(chan struct{})
	for i := 0; i < cfg.Workers; i++ {
		go func() {
			for b := range bloques {
				resultados <- ProcessBlock(b)
			}
			listo <- struct{}{}
		}()
	}
	go func() {
		for i := 0; i < cfg.Workers; i++ {
			<-listo
		}
		close(resultados)
	}()

	// 3) Reductor (esta misma goroutine): único que escribe y cuenta.
	resumen := Summary{Archivos: archivos, Motivos: map[string]int{}, Workers: cfg.Workers}
	for _, c := range Header()[:len(Header())-1] {
		w.WriteString(c)
		w.WriteByte(',')
	}
	w.WriteString(Header()[len(Header())-1])
	w.WriteByte('\n')

	pendientes := map[int]BlockResult{} // bloques que llegaron antes de su turno
	siguiente := 0
	vistos := map[uint64]struct{}{} // (cliente, año, mes) ya escritos

	escribir := func(r BlockResult) {
		resumen.FilasLeidas += r.Leidas
		for m, n := range r.Descartes {
			resumen.Motivos[m] += n
		}
		for _, rec := range r.Registros {
			clave := uint64(rec.CodigoCliente)*1000000 + uint64(rec.Anio)*100 + uint64(rec.Mes)
			if _, existe := vistos[clave]; existe {
				resumen.Motivos[Duplicado]++
				continue
			}
			vistos[clave] = struct{}{}
			resumen.FilasValidas++
			if rec.Facturacion < 0 {
				resumen.FacturacionNegativa++
			}
			escribirRegistro(w, rec)
		}
	}
	for r := range resultados {
		pendientes[r.Seq] = r
		for {
			sig, ok := pendientes[siguiente]
			if !ok {
				break
			}
			delete(pendientes, siguiente)
			escribir(sig)
			siguiente++
		}
	}

	if err := <-errLectura; err != nil {
		return Summary{}, err
	}
	if err := w.Flush(); err != nil {
		return Summary{}, err
	}

	resumen.FilasDescartadas = resumen.FilasLeidas - resumen.FilasValidas
	resumen.SegundosTotales = time.Since(inicio).Seconds()

	if cfg.ResumenJSON != "" {
		datos, _ := json.MarshalIndent(resumen, "", "  ")
		if err := os.WriteFile(cfg.ResumenJSON, datos, 0o644); err != nil {
			return resumen, err
		}
	}
	return resumen, nil
}

func escribirRegistro(w *bufio.Writer, r CleanRecord) {
	w.WriteString(strconv.Itoa(r.CodigoCliente))
	w.WriteByte(',')
	w.WriteString(strconv.Itoa(r.Ubigeo))
	for _, t := range []string{r.Departamento, r.Provincia, r.Distrito, r.Tarifa, r.Estado} {
		w.WriteByte(',')
		w.WriteString(t)
	}
	w.WriteByte(',')
	w.WriteString(strconv.Itoa(r.Anio))
	w.WriteByte(',')
	w.WriteString(strconv.Itoa(r.Mes))
	w.WriteByte(',')
	w.WriteString(strconv.FormatFloat(r.AntiguedadAnios, 'f', 3, 64))
	w.WriteByte(',')
	w.WriteString(strconv.FormatFloat(r.ConsumoKWh, 'f', -1, 64))
	w.WriteByte(',')
	w.WriteString(strconv.FormatFloat(r.Facturacion, 'f', -1, 64))
	w.WriteByte('\n')
}
