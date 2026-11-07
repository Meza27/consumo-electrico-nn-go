// Package preprocessing carga y limpia los archivos mensuales de consumo
// eléctrico de Electro Puno (datosabiertos.gob.pe) usando varias goroutines.
//
// El flujo es: una goroutine lee los archivos y los corta en bloques de
// líneas, varias goroutines "worker" limpian cada bloque en paralelo, y una
// sola goroutine "reductora" junta los resultados, escribe el CSV limpio y
// arma el resumen. Como solo el reductor toca el resultado final, no hacen
// falta mutex.
package preprocessing

// ColumnasEsperadas es el número de columnas reales de cada archivo. Casi todas
// las filas terminan con un ';' de más, así que al separar por ';' pueden
// salir 12 o 13 campos.
const ColumnasEsperadas = 12

// Motivos por los que se descarta una fila.
const (
	FilaVacia           = "fila_vacia"
	ColumnasIncorrectas = "columnas_incorrectas"
	ErrorDeParseo       = "error_de_parseo"
	UbigeoInvalido      = "ubigeo_invalido"
	TextoVacio          = "texto_vacio"
	PeriodoInvalido     = "periodo_invalido"
	FechaAltaInvalida   = "fecha_alta_invalida"
	AltaPosterior       = "alta_posterior_al_periodo"
	ConsumoNegativo     = "consumo_negativo"
	Duplicado           = "duplicado"
)

// CleanRecord es una fila ya validada y transformada.
type CleanRecord struct {
	CodigoCliente   int
	Ubigeo          int
	Departamento    string
	Provincia       string
	Distrito        string
	Tarifa          string
	Estado          string
	Anio            int
	Mes             int
	AntiguedadAnios float64 // años entre la fecha de alta y el periodo
	ConsumoKWh      float64
	Facturacion     float64 // puede ser negativa (ajustes); no es feature del modelo
}

// Header devuelve las columnas del CSV limpio, en orden.
func Header() []string {
	return []string{
		"codigo_cliente", "ubigeo", "departamento", "provincia", "distrito",
		"tarifa", "estado_cliente", "anio", "mes", "antiguedad_anios",
		"consumo_kwh", "facturacion",
	}
}

// Block es un grupo de líneas crudas de un archivo. Seq numera los bloques en
// el orden en que se leyeron, para poder escribir el resultado en ese mismo
// orden aunque los workers terminen desordenados.
type Block struct {
	Seq     int
	Archivo string
	Lines   []string
}

// BlockResult es lo que devuelve un worker por cada bloque.
type BlockResult struct {
	Seq       int
	Archivo   string
	Leidas    int
	Registros []CleanRecord
	Descartes map[string]int
}
