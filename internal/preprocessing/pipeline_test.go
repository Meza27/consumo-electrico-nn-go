package preprocessing

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const enc = "C\xd3DIGO_CLIENTE;UBIGEO;DEPARTAMENTO;PROVINCIA;DISTRITO;FECHA_ALTA;TARIFA;PERIODO;CONSUMO;FACTURACI\xd3N;ESTADO_CLIENTE;FECHA_CORTE;\r\n"

func TestProcessLine(t *testing.T) {
	casos := []struct {
		nombre, linea, motivo string
	}{
		{"valida", "1001;210101;PUNO;PUNO;PUNO;15/03/1989;BT5B;202301;58;45.9;NORMAL;31/01/2023;", ""},
		{"valida sin ; final", "1001;210101;PUNO;PUNO;PUNO;5/3/1989;BT5B;202301;58;45.9;NORMAL;31/01/2023", ""},
		{"vacia", ";;;;;;;;;;;;", FilaVacia},
		{"pocas columnas", "1001;210101;PUNO", ColumnasIncorrectas},
		{"consumo texto", "1001;210101;PUNO;PUNO;PUNO;15/03/1989;BT5B;202301;abc;45.9;NORMAL;31/01/2023;", ErrorDeParseo},
		{"consumo negativo", "1001;210101;PUNO;PUNO;PUNO;15/03/1989;BT5B;202301;-5;45.9;NORMAL;31/01/2023;", ConsumoNegativo},
		{"periodo malo", "1001;210101;PUNO;PUNO;PUNO;15/03/1989;BT5B;202313;58;45.9;NORMAL;31/01/2023;", PeriodoInvalido},
		{"alta futura", "1001;210101;PUNO;PUNO;PUNO;15/03/2030;BT5B;202301;58;45.9;NORMAL;31/01/2023;", AltaPosterior},
		{"facturacion negativa (ajuste) se conserva", "1001;210101;PUNO;PUNO;PUNO;15/03/1989;BT5B;202301;58;-12.4;NORMAL;31/01/2023;", ""},
		{"ubigeo corto", "1001;2101;PUNO;PUNO;PUNO;15/03/1989;BT5B;202301;58;45.9;NORMAL;31/01/2023;", UbigeoInvalido},
	}
	for _, c := range casos {
		t.Run(c.nombre, func(t *testing.T) {
			_, motivo := ProcessLine(c.linea)
			if motivo != c.motivo {
				t.Fatalf("esperaba %q y salió %q", c.motivo, motivo)
			}
		})
	}
}

func TestLatin1(t *testing.T) {
	// "Ó" suelta (byte 0xD3) debe convertirse bien, sin romper la fila.
	rec, motivo := ProcessLine("1;210101;PUNO;PUNO;PUNO;1/1/2000;BT5B;202301;10;5;NORMAL;31/01/2023;")
	if motivo != "" || rec.Mes != 1 {
		t.Fatal("fila simple falló")
	}
	rec, motivo = ProcessLine("1;210101;PUNO;PUNO;A\xd1ANCHO;1/1/2000;BT5B;202301;10;5;NORMAL;31/01/2023;")
	if motivo != "" || rec.Distrito != "AÑANCHO" {
		t.Fatalf("Latin-1 mal convertido: %q (%s)", rec.Distrito, motivo)
	}
}

// crea una carpeta con n archivos de filasPorArchivo filas cada uno.
func datosPrueba(t *testing.T, n, filas int) string {
	dir := t.TempDir()
	for a := 0; a < n; a++ {
		var sb strings.Builder
		sb.WriteString(enc)
		for i := 0; i < filas; i++ {
			fmt.Fprintf(&sb, "%d;210101;PUNO;PUNO;PUNO;15/03/1989;BT5B;2023%02d;%d;%d.5;NORMAL;31/01/2023;\r\n",
				1000+i, a+1, 50+i%40, 20+i%40)
		}
		sb.WriteString(";;;;;;;;;;;;\r\n")
		sb.WriteString("basura;x\r\n")
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("mes%02d.csv", a+1)), []byte(sb.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func ejecutar(t *testing.T, dir string, workers int) (Summary, string) {
	salida := filepath.Join(t.TempDir(), "limpio.csv")
	s, err := Run(Config{DirEntrada: dir, Salida: salida, Workers: workers, TamanoBloque: 7})
	if err != nil {
		t.Fatal(err)
	}
	datos, _ := os.ReadFile(salida)
	return s, string(datos)
}

func TestRunConteos(t *testing.T) {
	dir := datosPrueba(t, 3, 100)
	s, _ := ejecutar(t, dir, 4)
	if s.FilasLeidas != 3*102 || s.FilasValidas != 300 || s.FilasDescartadas != 6 {
		t.Fatalf("conteos mal: %+v", s)
	}
	if s.Motivos[FilaVacia] != 3 || s.Motivos[ColumnasIncorrectas] != 3 {
		t.Fatalf("motivos mal: %+v", s.Motivos)
	}
}

// Con 1 worker o con 8, el CSV tiene que quedar idéntico (mismo orden).
func TestMismoResultadoConVariosWorkers(t *testing.T) {
	dir := datosPrueba(t, 3, 200)
	_, a := ejecutar(t, dir, 1)
	_, b := ejecutar(t, dir, 8)
	if a != b {
		t.Fatal("el resultado cambia según la cantidad de workers")
	}
}

// Las filas con facturación negativa son ajustes: deben quedar en el CSV y
// contarse aparte en el resumen, no como descarte.
func TestFacturacionNegativaSeConserva(t *testing.T) {
	dir := t.TempDir()
	filas := "1;210101;PUNO;PUNO;PUNO;1/1/2000;BT5B;202301;10;-5.5;NORMAL;31/01/2023;\r\n" +
		"2;210101;PUNO;PUNO;PUNO;1/1/2000;BT5B;202301;20;8;NORMAL;31/01/2023;\r\n"
	os.WriteFile(filepath.Join(dir, "a.csv"), []byte(enc+filas), 0o644)
	s, csv := ejecutar(t, dir, 2)
	if s.FilasValidas != 2 || s.FilasDescartadas != 0 || s.FacturacionNegativa != 1 {
		t.Fatalf("la facturación negativa no se conservó: %+v", s)
	}
	if !strings.Contains(csv, ",10,-5.5\n") {
		t.Fatalf("la fila con facturación negativa no está en el CSV:\n%s", csv)
	}
}

func TestDuplicados(t *testing.T) {
	dir := t.TempDir()
	fila := "1;210101;PUNO;PUNO;PUNO;1/1/2000;BT5B;202301;10;5;NORMAL;31/01/2023;\r\n"
	os.WriteFile(filepath.Join(dir, "a.csv"), []byte(enc+fila+fila), 0o644)
	s, _ := ejecutar(t, dir, 2)
	if s.FilasValidas != 1 || s.Motivos[Duplicado] != 1 {
		t.Fatalf("duplicado no detectado: %+v", s)
	}
}
