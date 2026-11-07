package preprocessing

import (
	"math"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// aUTF8 arregla el texto de los archivos de Electro Puno: vienen en Latin-1
// (la "Ó" de CÓDIGO llega como un solo byte), y Go espera UTF-8. Si la línea
// ya es UTF-8 válido no se toca.
func aUTF8(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	runas := make([]rune, 0, len(s))
	for i := 0; i < len(s); i++ {
		runas = append(runas, rune(s[i]))
	}
	return string(runas)
}

// parseNumero convierte "58.9" (o "58,9") a número y rechaza NaN e infinitos.
func parseNumero(s string) (float64, error) {
	if !strings.Contains(s, ".") {
		s = strings.Replace(s, ",", ".", 1)
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, strconv.ErrSyntax
	}
	return v, nil
}

// ProcessLine valida y transforma una línea cruda. Devuelve el registro limpio
// o, si la fila no sirve, el motivo del descarte. No usa nada compartido, así
// que cada worker puede llamarla por su cuenta.
func ProcessLine(raw string) (CleanRecord, string) {
	linea := aUTF8(raw)

	// Una fila hecha solo de ';' y espacios cuenta como vacía.
	if strings.TrimSpace(strings.ReplaceAll(linea, ";", "")) == "" {
		return CleanRecord{}, FilaVacia
	}

	campos := strings.Split(linea, ";")
	// Quitamos el ';' de más del final, si lo trae.
	if len(campos) == ColumnasEsperadas+1 && strings.TrimSpace(campos[ColumnasEsperadas]) == "" {
		campos = campos[:ColumnasEsperadas]
	}
	if len(campos) != ColumnasEsperadas {
		return CleanRecord{}, ColumnasIncorrectas
	}
	for i := range campos {
		campos[i] = strings.TrimSpace(campos[i])
	}

	// Orden de columnas: 0 codigo, 1 ubigeo, 2 depto, 3 provincia, 4 distrito,
	// 5 fecha_alta, 6 tarifa, 7 periodo, 8 consumo, 9 facturación, 10 estado,
	// 11 fecha_corte (no se usa).
	codigo, err := strconv.Atoi(campos[0])
	if err != nil || codigo <= 0 {
		return CleanRecord{}, ErrorDeParseo
	}

	if len(campos[1]) != 6 {
		return CleanRecord{}, UbigeoInvalido
	}
	ubigeo, err := strconv.Atoi(campos[1])
	if err != nil {
		return CleanRecord{}, UbigeoInvalido
	}

	departamento := strings.ToUpper(campos[2])
	provincia := strings.ToUpper(campos[3])
	distrito := strings.ToUpper(campos[4])
	tarifa := strings.ToUpper(campos[6])
	estado := strings.ToUpper(campos[10])
	if departamento == "" || provincia == "" || distrito == "" || tarifa == "" || estado == "" {
		return CleanRecord{}, TextoVacio
	}

	// Periodo en formato AAAAMM.
	if len(campos[7]) != 6 {
		return CleanRecord{}, PeriodoInvalido
	}
	periodo, err := strconv.Atoi(campos[7])
	if err != nil {
		return CleanRecord{}, PeriodoInvalido
	}
	anio, mes := periodo/100, periodo%100
	if anio < 2000 || anio > 2100 || mes < 1 || mes > 12 {
		return CleanRecord{}, PeriodoInvalido
	}

	// La fecha de alta puede venir como 1/1/1989 o 01/01/1989.
	alta, err := time.Parse("2/1/2006", campos[5])
	if err != nil {
		return CleanRecord{}, FechaAltaInvalida
	}
	inicioMes := time.Date(anio, time.Month(mes), 1, 0, 0, 0, 0, time.UTC)
	if !alta.Before(inicioMes.AddDate(0, 1, 0)) {
		return CleanRecord{}, AltaPosterior
	}
	antiguedad := inicioMes.Sub(alta).Hours() / 24 / 365.25
	if antiguedad < 0 {
		antiguedad = 0 // se dio de alta dentro del mismo mes
	}

	consumo, err := parseNumero(campos[8])
	if err != nil {
		return CleanRecord{}, ErrorDeParseo
	}
	facturacion, err := parseNumero(campos[9])
	if err != nil {
		return CleanRecord{}, ErrorDeParseo
	}
	if consumo < 0 {
		return CleanRecord{}, ConsumoNegativo
	}
	// La facturación negativa NO se descarta: son ajustes/notas de crédito de
	// la empresa y el consumo de esa fila sigue siendo válido. Además la
	// facturación no se usa como feature del modelo (sería fuga de información).

	return CleanRecord{
		CodigoCliente:   codigo,
		Ubigeo:          ubigeo,
		Departamento:    departamento,
		Provincia:       provincia,
		Distrito:        distrito,
		Tarifa:          tarifa,
		Estado:          estado,
		Anio:            anio,
		Mes:             mes,
		AntiguedadAnios: antiguedad,
		ConsumoKWh:      consumo,
		Facturacion:     facturacion,
	}, ""
}

// ProcessBlock limpia todas las líneas de un bloque.
func ProcessBlock(b Block) BlockResult {
	res := BlockResult{
		Seq:       b.Seq,
		Archivo:   b.Archivo,
		Leidas:    len(b.Lines),
		Registros: make([]CleanRecord, 0, len(b.Lines)),
		Descartes: make(map[string]int),
	}
	for _, linea := range b.Lines {
		rec, motivo := ProcessLine(linea)
		if motivo != "" {
			res.Descartes[motivo]++
			continue
		}
		res.Registros = append(res.Registros, rec)
	}
	return res
}
