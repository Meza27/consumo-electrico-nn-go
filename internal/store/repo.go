package store

import (
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// Colecciones de MongoDB (entidades persistentes).
const (
	ColUsers       = "users"        // usuarios y roles (login con JWT)
	ColJobs        = "training_jobs" // entrenamientos distribuidos solicitados
	ColModels      = "models"       // modelos entrenados (pesos + encoder + métricas)
	ColPredictions = "predictions"  // historial de predicciones servidas por la API
	ColNodeMetrics = "node_metrics" // métricas de cada nodo ML por entrenamiento
)

// Claves de Redis (caché y estado compartido de corta vida).
const (
	KeyClusterNodes = "cluster:nodes"     // JSON con el estado de los nodos (TTL 30 s)
	KeyClusterLogs  = "cluster:logs"      // lista con los últimos 200 eventos
	KeyActiveModel  = "model:active"      // id del modelo activo (versión de la caché)
	KeyPredCount    = "stats:predictions" // contador de predicciones
)

// KeyPrediction es la clave de caché de una predicción.
func KeyPrediction(modelo, tarifa string, ubigeo, mes int, antiguedad float64) string {
	return fmt.Sprintf("pred:%s:%s:%d:%d:%.1f", modelo, tarifa, ubigeo, mes, antiguedad)
}

// KeyJobProgress es la clave del progreso de un entrenamiento.
func KeyJobProgress(id string) string { return "job:" + id + ":progress" }

// KeyDemand es la clave de la demanda agregada por distrito de un mes.
func KeyDemand(modelo string, mes int) string { return fmt.Sprintf("demand:%s:%d", modelo, mes) }

// Store agrupa las dos bases de datos y un escritor asíncrono de predicciones.
type Store struct {
	Mongo *Mongo
	Redis *Redis

	preds chan D // cola para insertar predicciones sin frenar la respuesta HTTP
}

// Open conecta con MongoDB y Redis, crea índices y lanza el escritor asíncrono.
func Open(mongoAddr, db, redisAddr string) (*Store, error) {
	m, err := DialMongo(mongoAddr, db, 8)
	if err != nil {
		return nil, fmt.Errorf("mongo %s: %w", mongoAddr, err)
	}
	r, err := DialRedis(redisAddr, 8)
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("redis %s: %w", redisAddr, err)
	}
	s := &Store{Mongo: m, Redis: r, preds: make(chan D, 4096)}
	for _, ix := range []struct {
		col, name string
		keys      D
		unique    bool
	}{
		{ColUsers, "username_unique", D{{"username", 1}}, true},
		{ColJobs, "created_at", D{{"created_at", -1}}, false},
		{ColModels, "active_created", D{{"active", 1}, {"created_at", -1}}, false},
		{ColPredictions, "created_at", D{{"created_at", -1}}, false},
		{ColPredictions, "ubigeo_mes", D{{"ubigeo", 1}, {"mes", 1}}, false},
		{ColNodeMetrics, "job_id", D{{"job_id", 1}}, false},
	} {
		if err := m.CreateIndex(ix.col, ix.name, ix.keys, ix.unique); err != nil {
			log.Printf("índice %s.%s: %v", ix.col, ix.name, err)
		}
	}
	go s.escritorPredicciones()
	return s, nil
}

// escritorPredicciones junta las predicciones de la cola y las inserta en
// lotes (InsertMany) cada 200 ms o cada 500 documentos. Así la latencia de
// /api/predict no depende de MongoDB.
func (s *Store) escritorPredicciones() {
	var lote []D
	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	vaciar := func() {
		if len(lote) == 0 {
			return
		}
		if err := s.Mongo.InsertMany(ColPredictions, lote); err != nil {
			log.Printf("guardar predicciones: %v", err)
		}
		lote = lote[:0]
	}
	for {
		select {
		case d, ok := <-s.preds:
			if !ok {
				vaciar()
				return
			}
			lote = append(lote, d)
			if len(lote) >= 500 {
				vaciar()
			}
		case <-tick.C:
			vaciar()
		}
	}
}

// ---------------------------------------------------------------- usuarios

// User es un usuario de la API.
type User struct {
	Username     string `json:"username"`
	PasswordHash string `json:"-"`
	Role         string `json:"role"`
}

// UpsertUser crea o actualiza un usuario.
func (s *Store) UpsertUser(u User) error {
	return s.Mongo.UpdateOne(ColUsers, D{{"username", u.Username}},
		D{{"$set", D{{"username", u.Username}, {"password_hash", u.PasswordHash}, {"role", u.Role}}},
			{"$setOnInsert", D{{"created_at", time.Now()}}}}, true)
}

// FindUser busca un usuario por nombre.
func (s *Store) FindUser(username string) (*User, error) {
	d, err := s.Mongo.FindOne(ColUsers, D{{"username", username}}, nil)
	if err != nil || d == nil {
		return nil, err
	}
	return &User{Username: ToString(d["username"]), PasswordHash: ToString(d["password_hash"]), Role: ToString(d["role"])}, nil
}

// ---------------------------------------------------------------- entrenamientos

// CreateJob registra un entrenamiento.
func (s *Store) CreateJob(id, user string, params any, nodos []string) error {
	pj, _ := json.Marshal(params)
	var pm M
	json.Unmarshal(pj, &pm)
	_, err := s.Mongo.InsertOne(ColJobs, D{{"_id", id}, {"status", "running"}, {"params", jsonToD(pm)},
		{"nodes", nodos}, {"created_by", user}, {"created_at", time.Now()}})
	return err
}

// FinishJob guarda el resultado (o el error) de un entrenamiento.
func (s *Store) FinishJob(id string, resultado any, errMsg string) error {
	set := D{{"finished_at", time.Now()}}
	if errMsg != "" {
		set = append(set, E{"status", "failed"}, E{"error", errMsg})
	} else {
		rj, _ := json.Marshal(resultado)
		var rm M
		json.Unmarshal(rj, &rm)
		set = append(set, E{"status", "done"}, E{"result", jsonToD(rm)})
	}
	return s.Mongo.UpdateOne(ColJobs, D{{"_id", id}}, D{{"$set", set}}, false)
}

// GetJob devuelve un entrenamiento.
func (s *Store) GetJob(id string) (M, error) {
	return s.Mongo.FindOne(ColJobs, D{{"_id", id}}, nil)
}

// ListJobs devuelve los últimos entrenamientos.
func (s *Store) ListJobs(limit int) ([]M, error) {
	return s.Mongo.Find(ColJobs, nil, D{{"created_at", -1}}, limit)
}

// ---------------------------------------------------------------- modelos

// SaveModel guarda un modelo, lo marca como activo y desactiva los anteriores.
// Devuelve su id (que también versiona la caché de Redis).
func (s *Store) SaveModel(jobID string, sizes []int, params []float64, encoderJSON string, metricas any) (string, error) {
	mj, _ := json.Marshal(metricas)
	var mm M
	json.Unmarshal(mj, &mm)
	if err := s.Mongo.UpdateMany(ColModels, D{{"active", true}}, D{{"$set", D{{"active", false}}}}); err != nil {
		return "", err
	}
	id, err := s.Mongo.InsertOne(ColModels, D{{"job_id", jobID}, {"sizes", sizes}, {"params", params},
		{"encoder", encoderJSON}, {"metrics", jsonToD(mm)}, {"active", true}, {"created_at", time.Now()}})
	if err != nil {
		return "", err
	}
	s.Redis.Set(KeyActiveModel, id.Hex(), 0)
	return id.Hex(), nil
}

// ActiveModel devuelve el modelo activo (o nil si no hay ninguno).
func (s *Store) ActiveModel() (M, error) {
	return s.Mongo.FindOne(ColModels, D{{"active", true}}, D{{"created_at", -1}})
}

// ---------------------------------------------------------------- predicciones

// Prediction es una predicción servida.
type Prediction struct {
	ModelID    string    `json:"model_id"`
	Tarifa     string    `json:"tarifa"`
	Ubigeo     int       `json:"ubigeo"`
	Mes        int       `json:"mes"`
	Antiguedad float64   `json:"antiguedad"`
	KWh        float64   `json:"kwh"`
	Cached     bool      `json:"cached"`
	LatencyMs  float64   `json:"latency_ms"`
	Username   string    `json:"username"`
	CreatedAt  time.Time `json:"created_at"`
}

// SavePrediction encola la predicción para insertarla en MongoDB (no bloquea).
func (s *Store) SavePrediction(p Prediction) {
	d := D{{"model_id", p.ModelID}, {"tarifa", p.Tarifa}, {"ubigeo", p.Ubigeo}, {"mes", p.Mes},
		{"antiguedad", p.Antiguedad}, {"kwh", p.KWh}, {"cached", p.Cached}, {"latency_ms", p.LatencyMs},
		{"username", p.Username}, {"created_at", p.CreatedAt}}
	select {
	case s.preds <- d:
	default: // cola llena: se descarta antes que frenar a la API
	}
	s.Redis.Incr(KeyPredCount)
}

// ListPredictions devuelve las últimas predicciones.
func (s *Store) ListPredictions(limit int) ([]M, error) {
	return s.Mongo.Find(ColPredictions, nil, D{{"created_at", -1}}, limit)
}

// PredictionStats agrega las predicciones: total, % desde caché y latencia media.
func (s *Store) PredictionStats() (M, error) {
	r, err := s.Mongo.Aggregate(ColPredictions, []D{{{"$group", D{{"_id", nil},
		{"total", D{{"$sum", 1}}},
		{"cached", D{{"$sum", D{{"$cond", A{"$cached", 1, 0}}}}}},
		{"avg_latency_ms", D{{"$avg", "$latency_ms"}}},
		{"max_latency_ms", D{{"$max", "$latency_ms"}}}}}}})
	if err != nil || len(r) == 0 {
		return M{"total": 0}, err
	}
	delete(r[0], "_id")
	return r[0], nil
}

// ---------------------------------------------------------------- nodos

// SaveNodeMetrics guarda las métricas por nodo de un entrenamiento.
func (s *Store) SaveNodeMetrics(jobID string, nodos any) error {
	nj, _ := json.Marshal(nodos)
	var arr []M
	json.Unmarshal(nj, &arr)
	docs := make([]D, 0, len(arr))
	for _, n := range arr {
		docs = append(docs, append(D{{"job_id", jobID}, {"created_at", time.Now()}}, jsonToD(n)...))
	}
	return s.Mongo.InsertMany(ColNodeMetrics, docs)
}

// Log agrega un evento del clúster a Redis (para el panel de administración).
func (s *Store) Log(msg string) {
	s.Redis.LPush(KeyClusterLogs, time.Now().Format("2006-01-02 15:04:05")+" "+msg, 200)
}

// jsonToD convierte un mapa decodificado de JSON a documento BSON.
func jsonToD(m M) D {
	d := make(D, 0, len(m))
	for k, v := range m {
		d = append(d, E{k, jsonVal(v)})
	}
	return d
}

func jsonVal(v any) any {
	switch x := v.(type) {
	case M:
		return jsonToD(x)
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = jsonVal(e)
		}
		return out
	}
	return v
}
