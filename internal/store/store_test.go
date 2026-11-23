package store

import (
	"bufio"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Codificar y decodificar un documento BSON debe devolver los mismos valores.
func TestBSONIdaYVuelta(t *testing.T) {
	ahora := time.UnixMilli(time.Now().UnixMilli()).UTC()
	oid := NewObjectID()
	b, err := Marshal(D{{"s", "hola ñ"}, {"f", 3.25}, {"i", 7}, {"g", int64(1) << 40}, {"b", true},
		{"t", ahora}, {"n", nil}, {"o", oid}, {"arr", []float64{1.5, -2}}, {"doc", D{{"x", "y"}}}})
	if err != nil {
		t.Fatal(err)
	}
	m, err := Unmarshal(b)
	if err != nil {
		t.Fatal(err)
	}
	arr := m["arr"].([]any)
	if m["s"] != "hola ñ" || m["f"] != 3.25 || m["i"] != int32(7) || m["g"] != int64(1)<<40 || m["b"] != true ||
		!m["t"].(time.Time).Equal(ahora) || m["n"] != nil || m["o"] != oid || arr[1] != -2.0 || m["doc"].(M)["x"] != "y" {
		t.Fatalf("documento distinto: %#v", m)
	}
}

// Documento de referencia de la especificación BSON: {"hello": "world"}.
func TestBSONEspecificacion(t *testing.T) {
	b, _ := Marshal(D{{"hello", "world"}})
	esperado := "\x16\x00\x00\x00\x02hello\x00\x06\x00\x00\x00world\x00\x00"
	if string(b) != esperado {
		t.Fatalf("bytes distintos: %q", b)
	}
	if _, err := Unmarshal(b[:10]); err == nil {
		t.Fatal("un documento truncado debía dar error")
	}
}

func TestRESP(t *testing.T) {
	r := bufio.NewReader(strings.NewReader("+OK\r\n:42\r\n$5\r\nhello\r\n$-1\r\n*2\r\n$1\r\na\r\n:1\r\n-ERR malo\r\n"))
	esperado := []any{"OK", int64(42), "hello", nil}
	for _, e := range esperado {
		v, err := leerRESP(r)
		if err != nil || v != e {
			t.Fatalf("esperaba %v, salió %v (%v)", e, v, err)
		}
	}
	v, _ := leerRESP(r)
	if arr := v.([]any); arr[0] != "a" || arr[1] != int64(1) {
		t.Fatalf("arreglo mal leído: %v", v)
	}
	if _, err := leerRESP(r); err == nil || !strings.Contains(err.Error(), "ERR malo") {
		t.Fatalf("esperaba error de Redis, salió %v", err)
	}
}

// Prueba de integración con MongoDB y Redis reales. Se ejecuta solo si están
// definidas MONGO_ADDR y REDIS_ADDR (por ejemplo, con docker compose).
func TestIntegracion(t *testing.T) {
	ma, ra := os.Getenv("MONGO_ADDR"), os.Getenv("REDIS_ADDR")
	if ma == "" || ra == "" {
		t.Skip("define MONGO_ADDR y REDIS_ADDR para probar contra bases reales")
	}
	s, err := Open(ma, "test_consumo", ra)
	if err != nil {
		t.Fatal(err)
	}
	s.Mongo.Command(D{{"drop", "items"}})
	if _, err := s.Mongo.InsertOne("items", D{{"k", 1}, {"v", "a"}}); err != nil {
		t.Fatal(err)
	}
	s.Mongo.InsertMany("items", []D{{{"k", 2}}, {{"k", 3}}})
	if n, _ := s.Mongo.Count("items", nil); n != 3 {
		t.Fatalf("count = %d", n)
	}
	docs, _ := s.Mongo.Find("items", D{{"k", D{{"$gte", 2}}}}, D{{"k", -1}}, 10)
	if len(docs) != 2 || ToInt(docs[0]["k"]) != 3 {
		t.Fatalf("find: %v", docs)
	}
	if err := s.Redis.Set("t:k", "v", time.Minute); err != nil {
		t.Fatal(err)
	}
	if v, _ := s.Redis.Get("t:k"); v != "v" {
		t.Fatalf("redis get = %q", v)
	}
	if _, err := s.Redis.Get("t:no-existe"); err != ErrNil {
		t.Fatalf("esperaba ErrNil, salió %v", err)
	}
	id, err := s.SaveModel("job-test", []int{3, 2, 1}, []float64{0.5, math.Pi}, "{}", map[string]any{"r2": 0.25})
	if err != nil {
		t.Fatal(err)
	}
	m, _ := s.ActiveModel()
	if m == nil || m["_id"].(ObjectID).Hex() != id || m["params"].([]any)[1] != math.Pi {
		t.Fatalf("modelo activo incorrecto: %v", m)
	}
}
