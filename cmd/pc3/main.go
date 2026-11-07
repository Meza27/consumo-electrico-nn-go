// Comando pc3: limpia los CSV mensuales de Electro Puno de forma concurrente.
//
// Uso:
//
//	go run ./cmd/pc3 -in data/raw/puno -out data/processed/dataset_limpio.csv -workers 8
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"consumo-electrico-nn-go/internal/platform"
	"consumo-electrico-nn-go/internal/preprocessing"
)

func main() {
	in := flag.String("in", "data/raw/puno", "carpeta con los CSV crudos")
	out := flag.String("out", "data/processed/dataset_limpio.csv", "CSV limpio de salida")
	resumen := flag.String("summary", "data/processed/resumen_limpieza.json", "resumen en JSON")
	workers := flag.Int("workers", runtime.NumCPU(), "cantidad de goroutines que limpian")
	bloque := flag.Int("block", 5000, "líneas por bloque")
	flag.Parse()
	platform.SinEcoQoS()

	s, err := preprocessing.Run(preprocessing.Config{
		DirEntrada:   *in,
		Salida:       *out,
		ResumenJSON:  *resumen,
		Workers:      *workers,
		TamanoBloque: *bloque,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	fmt.Printf("Archivos:     %d\n", len(s.Archivos))
	fmt.Printf("Filas leídas: %d\n", s.FilasLeidas)
	fmt.Printf("Filas válidas: %d\n", s.FilasValidas)
	fmt.Printf("Descartadas:  %d\n", s.FilasDescartadas)
	fmt.Printf("Facturación negativa (ajustes, conservadas): %d\n", s.FacturacionNegativa)
	for m, n := range s.Motivos {
		fmt.Printf("  - %s: %d\n", m, n)
	}
	fmt.Printf("Workers: %d | Tiempo: %.2f s\n", s.Workers, s.SegundosTotales)
}
