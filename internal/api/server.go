// Package api es la API REST/WebSocket coordinadora (PC4): expone el modelo,
// coordina el entrenamiento distribuido en los nodos ML por TCP y persiste
// resultados en MongoDB (persistente) y Redis (caché).
package api

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"consumo-electrico-nn-go/internal/cluster"
	"consumo-electrico-nn-go/internal/mlp"
	"consumo-electrico-nn-go/internal/store"
)

//go:embed openapi.yaml
var openapiYAML []byte

//go:embed swagger.html
var swaggerHTML []byte

// Config es la configuración de la API.
type Config struct {
	Addr      string
	NodeAddrs []string
	DataPath  string // CSV limpio (para entrenar y para la demanda por distrito)
	ModelPath string // modelo inicial (results/model.json de la PC3)
	JWTSecret []byte
	AdminUser string
	AdminPass string
	DemoUser  string
	DemoPass  string
}

// modeloActivo es el modelo que sirve predicciones. Se reemplaza entero de
// forma atómica cuando termina un entrenamiento (atomic.Pointer): las
// peticiones en curso siguen usando el anterior, sin bloqueos.
type modeloActivo struct {
	ID      string
	Saved   *mlp.SavedModel
	Metrics any
	Created time.Time
}

// Server es la API.
type Server struct {
	cfg   Config
	st    *store.Store
	coord *cluster.Coordinator
	hub   *Hub
	model atomic.Pointer[modeloActivo]
	grupos []grupoDemanda

	entrenando atomic.Bool // solo un entrenamiento a la vez

	latMu sync.Mutex // protege el buffer circular de latencias
	lats  []float64
	latI  int
	reqs  atomic.Int64
	ini   time.Time
}

// NewServer prepara la API: usuarios, modelo inicial y datos de demanda.
func NewServer(cfg Config, st *store.Store, coord *cluster.Coordinator) (*Server, error) {
	s := &Server{cfg: cfg, st: st, coord: coord, hub: NewHub(), lats: make([]float64, 0, 10000), ini: time.Now()}
	for _, u := range []store.User{{Username: cfg.AdminUser, Role: "admin"}, {Username: cfg.DemoUser, Role: "user"}} {
		pass := cfg.AdminPass
		if u.Role == "user" {
			pass = cfg.DemoPass
		}
		if u.Username == "" {
			continue
		}
		if existente, _ := st.FindUser(u.Username); existente == nil {
			u.PasswordHash = HashPassword(pass)
			if err := st.UpsertUser(u); err != nil {
				return nil, fmt.Errorf("crear usuario %s: %w", u.Username, err)
			}
		}
	}
	if err := s.cargarModelo(); err != nil {
		log.Printf("aviso: sin modelo inicial: %v", err)
	}
	if g, err := cargarGrupos(cfg.DataPath); err == nil {
		s.grupos = g
		log.Printf("demanda: %d grupos (distrito, tarifa) cargados", len(g))
	} else {
		log.Printf("aviso: sin datos de demanda: %v", err)
	}
	return s, nil
}

// cargarModelo toma el modelo activo de MongoDB; si no hay, importa el modelo
// de la PC3 (results/model.json) y lo registra como primer modelo.
func (s *Server) cargarModelo() error {
	if d, err := s.st.ActiveModel(); err == nil && d != nil {
		sm, err := savedDesdeMongo(d)
		if err != nil {
			return err
		}
		id := d["_id"].(store.ObjectID).Hex()
		s.model.Store(&modeloActivo{ID: id, Saved: sm, Metrics: d["metrics"], Created: tiempo(d["created_at"])})
		s.st.Redis.Set(store.KeyActiveModel, id, 0)
		log.Printf("modelo activo %s cargado desde MongoDB", id)
		return nil
	}
	sm, err := mlp.Load(s.cfg.ModelPath)
	if err != nil {
		return err
	}
	encJSON, _ := json.Marshal(sm.Encoder)
	id, err := s.st.SaveModel("pc3-inicial", sm.Model.Sizes, sm.Model.Params, string(encJSON), sm.Metrics)
	if err != nil {
		return err
	}
	s.model.Store(&modeloActivo{ID: id, Saved: sm, Metrics: sm.Metrics, Created: time.Now()})
	log.Printf("modelo de la PC3 importado a MongoDB con id %s", id)
	return nil
}

func savedDesdeMongo(d store.M) (*mlp.SavedModel, error) {
	var enc mlp.Encoder
	if err := json.Unmarshal([]byte(store.ToString(d["encoder"])), &enc); err != nil {
		return nil, err
	}
	var sizes []int
	for _, v := range d["sizes"].([]any) {
		sizes = append(sizes, store.ToInt(v))
	}
	raw := d["params"].([]any)
	m := mlp.NewMLP(sizes, 1)
	if len(raw) != m.NumParams() {
		return nil, fmt.Errorf("el modelo guardado tiene %d parámetros y se esperaban %d", len(raw), m.NumParams())
	}
	for i, v := range raw {
		m.Params[i] = store.ToFloat(v)
	}
	return &mlp.SavedModel{Model: m, Encoder: &enc}, nil
}

func tiempo(v any) time.Time {
	t, _ := v.(time.Time)
	return t
}

// Handler arma las rutas (patrones de net/http de Go 1.22: "MÉTODO /ruta").
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", s.health)
	mux.HandleFunc("POST /api/auth/login", s.login)
	mux.HandleFunc("POST /api/predict", s.auth("", s.predict))
	mux.HandleFunc("GET /api/demand", s.auth("", s.demand))
	mux.HandleFunc("GET /api/model", s.auth("", s.modelInfo))
	mux.HandleFunc("GET /api/predictions", s.auth("", s.predictions))
	mux.HandleFunc("GET /api/predictions/stats", s.auth("", s.predictionStats))
	mux.HandleFunc("POST /api/train", s.auth("admin", s.train))
	mux.HandleFunc("GET /api/train", s.auth("", s.listJobs))
	mux.HandleFunc("GET /api/train/{id}", s.auth("", s.getJob))
	mux.HandleFunc("GET /api/cluster/nodes", s.auth("admin", s.clusterNodes))
	mux.HandleFunc("GET /api/cluster/logs", s.auth("admin", s.clusterLogs))
	mux.HandleFunc("GET /api/metrics", s.auth("admin", s.metrics))
	mux.HandleFunc("GET /ws", s.auth("", s.hub.ServeWS))
	mux.HandleFunc("GET /swagger/openapi.yaml", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml")
		w.Write(openapiYAML)
	})
	mux.HandleFunc("GET /swagger/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(swaggerHTML)
	})
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/swagger/", http.StatusFound) })
	return s.middleware(mux)
}

// middleware: CORS (para el frontend de la TB2), medición de latencia y log.
func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		ini := time.Now()
		next.ServeHTTP(w, r)
		if r.URL.Path != "/ws" {
			s.registrarLatencia(float64(time.Since(ini).Microseconds()) / 1000)
		}
	})
}

func (s *Server) registrarLatencia(ms float64) {
	s.reqs.Add(1)
	s.latMu.Lock()
	if len(s.lats) < cap(s.lats) {
		s.lats = append(s.lats, ms)
	} else {
		s.lats[s.latI] = ms
		s.latI = (s.latI + 1) % len(s.lats)
	}
	s.latMu.Unlock()
}

// Start lanza tareas en segundo plano: sondeo del clúster cada 10 s.
func (s *Server) Start() {
	go func() {
		for {
			nodos := s.coord.Probe()
			b, _ := json.Marshal(nodos)
			s.st.Redis.Set(store.KeyClusterNodes, string(b), 30*time.Second)
			s.hub.Publish("cluster", nodos)
			time.Sleep(10 * time.Second)
		}
	}()
}

// ---------------------------------------------------------------- utilidades

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func queryInt(r *http.Request, k string, def int) int {
	if v, err := strconv.Atoi(r.URL.Query().Get(k)); err == nil {
		return v
	}
	return def
}

func percentil(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	c := append([]float64(nil), xs...)
	sort.Float64s(c)
	return c[min(len(c)-1, int(p*float64(len(c))))]
}

// limpiarIDs convierte ObjectID y fechas a texto para responder en JSON.
func limpiarIDs(docs []store.M) []store.M {
	for _, d := range docs {
		limpiarDoc(d)
	}
	return docs
}

func limpiarDoc(d store.M) store.M {
	for k, v := range d {
		switch x := v.(type) {
		case store.ObjectID:
			d[k] = x.Hex()
		case store.M:
			limpiarDoc(x)
		}
	}
	return d
}

func runtimeStats() map[string]any {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return map[string]any{"goroutines": runtime.NumGoroutine(), "heap_mb": float64(m.HeapAlloc) / (1 << 20),
		"cpus": runtime.NumCPU(), "gc_cycles": m.NumGC}
}
