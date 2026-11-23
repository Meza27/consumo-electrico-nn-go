package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"consumo-electrico-nn-go/internal/cluster"
	"consumo-electrico-nn-go/internal/store"
)

// GET /api/health (público): estado de bases de datos, nodos y modelo.
func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	estado := map[string]any{"status": "ok", "uptime_sec": int(time.Since(s.ini).Seconds())}
	if _, err := s.st.Mongo.Ping(); err != nil {
		estado["mongo"], estado["status"] = "error: "+err.Error(), "degraded"
	} else {
		estado["mongo"] = "ok"
	}
	if _, err := s.st.Redis.Do("PING"); err != nil {
		estado["redis"], estado["status"] = "error: "+err.Error(), "degraded"
	} else {
		estado["redis"] = "ok"
	}
	vivos := 0
	for _, n := range s.coord.Status() {
		if n.Alive {
			vivos++
		}
	}
	estado["nodes_alive"], estado["nodes_total"] = vivos, len(s.cfg.NodeAddrs)
	if m := s.model.Load(); m != nil {
		estado["model_id"] = m.ID
	}
	writeJSON(w, http.StatusOK, estado)
}

// POST /api/auth/login {username, password} -> {token, role, expires_in}
func (s *Server) login(w http.ResponseWriter, r *http.Request) {
	var in struct{ Username, Password string }
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil || in.Username == "" {
		writeErr(w, http.StatusBadRequest, "se esperaba {\"username\", \"password\"}")
		return
	}
	u, err := s.st.FindUser(in.Username)
	if err != nil || u == nil || !CheckPassword(in.Password, u.PasswordHash) {
		writeErr(w, http.StatusUnauthorized, "usuario o contraseña incorrectos")
		return
	}
	tok, err := NewJWT(s.cfg.JWTSecret, u.Username, u.Role, 8*time.Hour)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"token": tok, "role": u.Role, "expires_in": 8 * 3600})
}

// PredictRequest es la consulta de predicción.
type PredictRequest struct {
	Tarifa     string  `json:"tarifa"`
	Ubigeo     int     `json:"ubigeo"`
	Mes        int     `json:"mes"`
	Antiguedad float64 `json:"antiguedad"`
}

// POST /api/predict: consumo mensual estimado (kWh) de un suministro.
// Primero consulta la caché de Redis; si no está, calcula con el MLP, guarda en
// caché y encola el registro para MongoDB (sin esperar a la base de datos).
func (s *Server) predict(w http.ResponseWriter, r *http.Request) {
	ini := time.Now()
	var in PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeErr(w, http.StatusBadRequest, "JSON inválido")
		return
	}
	in.Tarifa = strings.ToUpper(strings.TrimSpace(in.Tarifa))
	if in.Tarifa == "" || in.Mes < 1 || in.Mes > 12 || in.Ubigeo < 10000 || in.Antiguedad < 0 || in.Antiguedad > 120 {
		writeErr(w, http.StatusBadRequest, "campos: tarifa (p. ej. BT5B), ubigeo (6 dígitos), mes (1-12), antiguedad (años, 0-120)")
		return
	}
	m := s.model.Load()
	if m == nil {
		writeErr(w, http.StatusServiceUnavailable, "no hay modelo entrenado")
		return
	}
	in.Antiguedad = float64(int(in.Antiguedad*10+0.5)) / 10 // redondeo a 0.1 años: mejora el uso de la caché
	clave := store.KeyPrediction(m.ID, in.Tarifa, in.Ubigeo, in.Mes, in.Antiguedad)
	var kwh float64
	cached := false
	if v, err := s.st.Redis.Get(clave); err == nil {
		kwh, _ = strconv.ParseFloat(v, 64)
		cached = true
	} else {
		kwh = s.predecir(m, in)
		s.st.Redis.Set(clave, strconv.FormatFloat(kwh, 'f', 4, 64), time.Hour)
	}
	lat := float64(time.Since(ini).Microseconds()) / 1000
	conocido := s.conocido(m, in)
	c := claimsDe(r)
	p := store.Prediction{ModelID: m.ID, Tarifa: in.Tarifa, Ubigeo: in.Ubigeo, Mes: in.Mes, Antiguedad: in.Antiguedad,
		KWh: kwh, Cached: cached, LatencyMs: lat, Username: c.Sub, CreatedAt: time.Now()}
	s.st.SavePrediction(p)
	s.hub.Publish("prediction", p)
	writeJSON(w, http.StatusOK, map[string]any{"kwh": kwh, "cached": cached, "latency_ms": lat, "model_id": m.ID,
		"input": in, "known_categories": conocido})
}

func (s *Server) predecir(m *modeloActivo, in PredictRequest) float64 {
	return m.Saved.PredictKWh(in.Tarifa, in.Ubigeo, in.Mes, in.Antiguedad)
}

// conocido indica si la tarifa y el distrito existían en el entrenamiento.
func (s *Server) conocido(m *modeloActivo, in PredictRequest) map[string]bool {
	t, d := false, false
	for _, x := range m.Saved.Encoder.Tarifas {
		t = t || x == in.Tarifa
	}
	for _, x := range m.Saved.Encoder.Distritos {
		d = d || x == in.Ubigeo
	}
	return map[string]bool{"tarifa": t, "ubigeo": d}
}

// GET /api/demand?mes=7&top=20: demanda estimada por distrito para un mes.
// Responde la pregunta de impacto social: ¿qué zonas tendrán más demanda?
func (s *Server) demand(w http.ResponseWriter, r *http.Request) {
	ini := time.Now()
	mes := queryInt(r, "mes", int(time.Now().Month()))
	top := queryInt(r, "top", 20)
	if mes < 1 || mes > 12 {
		writeErr(w, http.StatusBadRequest, "mes debe estar entre 1 y 12")
		return
	}
	m := s.model.Load()
	if m == nil || len(s.grupos) == 0 {
		writeErr(w, http.StatusServiceUnavailable, "no hay modelo o datos de demanda")
		return
	}
	clave := store.KeyDemand(m.ID, mes)
	var res demandaResultado
	cached := false
	if v, err := s.st.Redis.Get(clave); err == nil && json.Unmarshal([]byte(v), &res) == nil {
		cached = true
	} else {
		res = s.calcularDemanda(m, mes)
		b, _ := json.Marshal(res)
		s.st.Redis.Set(clave, string(b), time.Hour)
	}
	if top > 0 && top < len(res.Distritos) {
		res.Distritos = res.Distritos[:top]
	}
	writeJSON(w, http.StatusOK, map[string]any{"mes": mes, "model_id": m.ID, "cached": cached,
		"latency_ms": float64(time.Since(ini).Microseconds()) / 1000, "total_mwh": res.TotalMWh,
		"suministros": res.Suministros, "distritos": res.Distritos})
}

// GET /api/model: modelo activo y sus métricas.
func (s *Server) modelInfo(w http.ResponseWriter, r *http.Request) {
	m := s.model.Load()
	if m == nil {
		writeErr(w, http.StatusNotFound, "no hay modelo")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"model_id": m.ID, "sizes": m.Saved.Model.Sizes, "params": m.Saved.Model.NumParams(),
		"metrics": m.Metrics, "created_at": m.Created, "tarifas": m.Saved.Encoder.Tarifas, "distritos": len(m.Saved.Encoder.Distritos)})
}

// GET /api/predictions?limit=20
func (s *Server) predictions(w http.ResponseWriter, r *http.Request) {
	docs, err := s.st.ListPredictions(min(max(queryInt(r, "limit", 20), 1), 200))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, limpiarIDs(docs))
}

// GET /api/predictions/stats
func (s *Server) predictionStats(w http.ResponseWriter, r *http.Request) {
	st, err := s.st.PredictionStats()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, st)
}

// TrainRequest son los parámetros de un entrenamiento distribuido.
type TrainRequest struct {
	Epochs int     `json:"epochs"`
	Batch  int     `json:"batch"`
	LR     float64 `json:"lr"`
	Hidden []int   `json:"hidden"`
	Limit  int     `json:"limit"` // 0 = todas las filas
}

// POST /api/train (admin): lanza un entrenamiento distribuido en segundo plano.
func (s *Server) train(w http.ResponseWriter, r *http.Request) {
	in := TrainRequest{Epochs: 5, Batch: 4096, LR: 0.003, Hidden: []int{64, 32}}
	if r.ContentLength != 0 {
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
			writeErr(w, http.StatusBadRequest, "JSON inválido")
			return
		}
	}
	if in.Epochs < 1 || in.Epochs > 100 || in.Batch < 32 || in.LR <= 0 || in.LR > 1 || len(in.Hidden) == 0 || len(in.Hidden) > 4 {
		writeErr(w, http.StatusBadRequest, "parámetros fuera de rango (epochs 1-100, batch >= 32, 0 < lr <= 1, 1-4 capas ocultas)")
		return
	}
	if !s.entrenando.CompareAndSwap(false, true) {
		writeErr(w, http.StatusConflict, "ya hay un entrenamiento en curso")
		return
	}
	id := fmt.Sprintf("job-%s", time.Now().Format("20060102-150405"))
	cfg := cluster.JobConfig{JobID: id, DataPath: s.cfg.DataPath, Limit: in.Limit, TestFrac: 0.2, Seed: 42,
		Epochs: in.Epochs, Batch: in.Batch, LR: in.LR, Hidden: in.Hidden}
	if err := s.st.CreateJob(id, claimsDe(r).Sub, cfg, s.cfg.NodeAddrs); err != nil {
		s.entrenando.Store(false)
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	go s.ejecutarEntrenamiento(cfg)
	writeJSON(w, http.StatusAccepted, map[string]any{"job_id": id, "status": "running", "nodes": s.cfg.NodeAddrs,
		"follow": "/api/train/" + id + " o WebSocket /ws"})
}

func (s *Server) ejecutarEntrenamiento(cfg cluster.JobConfig) {
	defer s.entrenando.Store(false)
	s.st.Log("inicio de " + cfg.JobID)
	s.hub.Publish("log", "inicio de "+cfg.JobID)
	res, err := s.coord.Train(context.Background(), cfg, func(p cluster.Progress) {
		b, _ := json.Marshal(p)
		s.st.Redis.Set(store.KeyJobProgress(cfg.JobID), string(b), 24*time.Hour)
		s.hub.Publish("train_progress", p)
		if p.EpochDone {
			msg := fmt.Sprintf("%s: época %d/%d, pérdida %.4f, %.1f s, %d nodos", cfg.JobID, p.Epoch, p.Epochs, p.Loss, p.EpochSecond, p.AliveNodes)
			s.st.Log(msg)
			s.hub.Publish("log", msg)
		}
	})
	if err != nil {
		s.st.FinishJob(cfg.JobID, nil, err.Error())
		s.st.Log(cfg.JobID + " falló: " + err.Error())
		s.hub.Publish("train_done", map[string]any{"job_id": cfg.JobID, "error": err.Error()})
		return
	}
	s.st.FinishJob(cfg.JobID, res, "")
	s.st.SaveNodeMetrics(cfg.JobID, res.Nodes)
	encJSON, _ := json.Marshal(res.Encoder)
	metricas := map[string]any{"test": res.Test, "baseline_test": res.Baseline, "train_seconds": res.Seconds}
	id, err := s.st.SaveModel(cfg.JobID, res.Model.Sizes, res.Model.Params, string(encJSON), metricas)
	if err == nil {
		sm, _ := savedDesdeMongoLike(res)
		s.model.Store(&modeloActivo{ID: id, Saved: sm, Metrics: metricas, Created: time.Now()}) // cambio atómico
	}
	msg := fmt.Sprintf("%s terminado en %.1f s: R² log %.3f, MAE %.2f kWh; nuevo modelo %s", cfg.JobID, res.Seconds, res.Test.Log.R2, res.Test.KWh.MAE, id)
	s.st.Log(msg)
	s.hub.Publish("train_done", map[string]any{"job_id": cfg.JobID, "model_id": id, "result": res})
}

// GET /api/train: últimos entrenamientos.
func (s *Server) listJobs(w http.ResponseWriter, r *http.Request) {
	docs, err := s.st.ListJobs(min(max(queryInt(r, "limit", 10), 1), 50))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	for _, d := range docs { // el historial completo se pide por id
		if res, ok := d["result"].(store.M); ok {
			delete(res, "history")
		}
	}
	writeJSON(w, http.StatusOK, limpiarIDs(docs))
}

// GET /api/train/{id}: estado, progreso en vivo (Redis) y resultado (MongoDB).
func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	d, err := s.st.GetJob(id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if d == nil {
		writeErr(w, http.StatusNotFound, "no existe "+id)
		return
	}
	if v, err := s.st.Redis.Get(store.KeyJobProgress(id)); err == nil {
		var p any
		json.Unmarshal([]byte(v), &p)
		d["progress"] = p
	}
	writeJSON(w, http.StatusOK, limpiarDoc(d))
}

// GET /api/cluster/nodes (admin): estado de cada nodo ML (caché de 30 s en Redis).
func (s *Server) clusterNodes(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("refresh") != "1" {
		if v, err := s.st.Redis.Get(store.KeyClusterNodes); err == nil {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.Header().Set("X-Cache", "HIT")
			w.Write([]byte(v))
			return
		}
	}
	nodos := s.coord.Probe()
	b, _ := json.Marshal(nodos)
	s.st.Redis.Set(store.KeyClusterNodes, string(b), 30*time.Second)
	writeJSON(w, http.StatusOK, nodos)
}

// GET /api/cluster/logs (admin): últimos eventos del clúster.
func (s *Server) clusterLogs(w http.ResponseWriter, r *http.Request) {
	l, err := s.st.Redis.LRange(store.KeyClusterLogs, 0, min(max(queryInt(r, "limit", 50), 1), 200)-1)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, l)
}

// GET /api/metrics (admin): latencia de la API y uso del proceso.
func (s *Server) metrics(w http.ResponseWriter, r *http.Request) {
	s.latMu.Lock()
	lats := append([]float64(nil), s.lats...)
	s.latMu.Unlock()
	total, _ := s.st.Redis.Get(store.KeyPredCount)
	writeJSON(w, http.StatusOK, map[string]any{
		"requests": s.reqs.Load(), "latency_ms": map[string]float64{"p50": percentil(lats, 0.50), "p95": percentil(lats, 0.95),
			"p99": percentil(lats, 0.99), "max": percentil(lats, 1)},
		"window": len(lats), "predictions_total": total, "ws_clients": s.hub.Clients(), "training": s.entrenando.Load(),
		"process": runtimeStats(), "uptime_sec": int(time.Since(s.ini).Seconds())})
}
