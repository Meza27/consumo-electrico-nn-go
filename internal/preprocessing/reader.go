package preprocessing

import (
	"bufio"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ListarCSV busca todos los .csv dentro de la carpeta (también en subcarpetas,
// por si los .zip se descomprimieron en carpetas) y los ordena por nombre para
// que el resultado sea siempre el mismo.
func ListarCSV(dir string) ([]string, error) {
	var archivos []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".csv") {
			archivos = append(archivos, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(archivos)
	return archivos, nil
}

// esEncabezado dice si la primera línea del archivo es el encabezado: si el
// primer campo no es un número, es el título de las columnas.
func esEncabezado(linea string) bool {
	linea = strings.TrimPrefix(linea, "\xef\xbb\xbf") // marca BOM (UTF-8) al inicio
	primero := linea
	if i := strings.IndexByte(linea, ';'); i >= 0 {
		primero = linea[:i]
	}
	_, err := strconv.Atoi(strings.TrimSpace(primero))
	return err != nil
}

// StreamBlocks es el productor: lee los archivos uno por uno, línea a línea
// (sin cargarlos enteros en memoria), y manda las líneas en bloques por el
// channel. No cierra el channel; eso lo hace quien lo llama.
func StreamBlocks(archivos []string, tamanoBloque int, salida chan<- Block) error {
	seq := 0
	for _, ruta := range archivos {
		if err := leerArchivo(ruta, tamanoBloque, &seq, salida); err != nil {
			return err
		}
	}
	return nil
}

func leerArchivo(ruta string, tamanoBloque int, seq *int, salida chan<- Block) error {
	f, err := os.Open(ruta)
	if err != nil {
		return err
	}
	defer f.Close()

	lector := bufio.NewReaderSize(f, 1<<20)
	bloque := make([]string, 0, tamanoBloque)
	primera := true

	for {
		linea, err := lector.ReadString('\n')
		if len(linea) > 0 {
			linea = strings.TrimRight(linea, "\r\n")
			if primera {
				primera = false
				if esEncabezado(linea) {
					linea = "\x00" // marca para saltarla
				}
			}
			if linea != "\x00" {
				bloque = append(bloque, linea)
				if len(bloque) == tamanoBloque {
					salida <- Block{Seq: *seq, Archivo: ruta, Lines: bloque}
					*seq++
					bloque = make([]string, 0, tamanoBloque)
				}
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	if len(bloque) > 0 {
		salida <- Block{Seq: *seq, Archivo: ruta, Lines: bloque}
		*seq++
	}
	return nil
}
