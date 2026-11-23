// Comando loadtest: prueba de carga de POST /api/predict con N goroutines
// concurrentes. Mide la latencia de cada petición vista por el cliente y
// reporta percentiles para comprobar el requisito de < 100 ms.
//
//	go run ./cmd/loadtest -url http://localhost:8080 -n 5000 -c 50
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

func main() {
	base := flag.String("url", "http://localhost:8080", "URL base de la API")
	n := flag.Int("n", 5000, "peticiones totales")
	c := flag.Int("c", 50, "goroutines concurrentes")
	user := flag.String("user", "demo", "usuario")
	pass := flag.String("pass", "demo123", "contraseña")
	out := flag.String("out", "", "archivo JSON con el resultado (opcional)")
	flag.Parse()

	// 1) Login para obtener el JWT.
	body, _ := json.Marshal(map[string]string{"username": *user, "password": *pass})
	resp, err := http.Post(*base+"/api/auth/login", "application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintln(os.Stderr, "login:", err)
		os.Exit(1)
	}
	var lr struct{ Token string }
	json.NewDecoder(resp.Body).Decode(&lr)
	resp.Body.Close()
	if lr.Token == "" {
		fmt.Fprintln(os.Stderr, "login fallido")
		os.Exit(1)
	}

	// 2) Productor -> channel de trabajos -> C goroutines -> channel de resultados.
	ubigeos := []int{211101, 210101, 210801, 211201, 210201, 210701, 211001, 210501}
	tarifas := []string{"BT5B", "BT5B", "BT5B", "BT5D", "BT6"}
	trabajos := make(chan []byte, *c)
	type res struct {
		ms float64
		ok bool
	}
	resultados := make(chan res, *c)
	cliente := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{MaxIdleConnsPerHost: *c}}
	go func() {
		rng := rand.New(rand.NewSource(1))
		for i := 0; i < *n; i++ {
			b, _ := json.Marshal(map[string]any{"tarifa": tarifas[rng.Intn(len(tarifas))], "ubigeo": ubigeos[rng.Intn(len(ubigeos))],
				"mes": 1 + rng.Intn(12), "antiguedad": float64(rng.Intn(400)) / 10})
			trabajos <- b
		}
		close(trabajos)
	}()
	var wg sync.WaitGroup
	for w := 0; w < *c; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for b := range trabajos {
				req, _ := http.NewRequest("POST", *base+"/api/predict", bytes.NewReader(b))
				req.Header.Set("Authorization", "Bearer "+lr.Token)
				req.Header.Set("Content-Type", "application/json")
				ini := time.Now()
				r, err := cliente.Do(req)
				ms := float64(time.Since(ini).Microseconds()) / 1000
				ok := err == nil && r.StatusCode == 200
				if r != nil {
					var tmp map[string]any
					json.NewDecoder(r.Body).Decode(&tmp)
					r.Body.Close()
				}
				resultados <- res{ms, ok}
			}
		}()
	}
	go func() { wg.Wait(); close(resultados) }()

	ini := time.Now()
	var lats []float64
	errores := 0
	for r := range resultados {
		if r.ok {
			lats = append(lats, r.ms)
		} else {
			errores++
		}
	}
	total := time.Since(ini).Seconds()
	sort.Float64s(lats)
	p := func(q float64) float64 {
		if len(lats) == 0 {
			return 0
		}
		return lats[min(len(lats)-1, int(q*float64(len(lats))))]
	}
	bajo100 := sort.SearchFloat64s(lats, 100)
	resumen := map[string]any{"requests": *n, "concurrency": *c, "ok": len(lats), "errors": errores,
		"seconds": total, "req_per_sec": float64(len(lats)) / total,
		"p50_ms": p(0.50), "p95_ms": p(0.95), "p99_ms": p(0.99), "max_ms": p(1),
		"pct_under_100ms": 100 * float64(bajo100) / float64(max(len(lats), 1))}
	fmt.Printf("Peticiones: %d (concurrencia %d) | OK: %d | errores: %d | %.2f s | %.0f req/s\n",
		*n, *c, len(lats), errores, total, resumen["req_per_sec"])
	fmt.Printf("Latencia (ms): p50 %.2f | p95 %.2f | p99 %.2f | máx %.2f | < 100 ms: %.2f %%\n",
		p(0.5), p(0.95), p(0.99), p(1), resumen["pct_under_100ms"])
	if *out != "" {
		b, _ := json.MarshalIndent(resumen, "", "  ")
		os.WriteFile(*out, b, 0o644)
	}
}
