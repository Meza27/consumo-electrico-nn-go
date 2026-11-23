package cluster

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sort"
	"sync"
	"time"

	"consumo-electrico-nn-go/internal/mlp"
)

// NodeStatus es el estado de un nodo visto por el coordinador.
type NodeStatus struct {
	Addr          string    `json:"addr"`
	ID            string    `json:"id"`
	Hostname      string    `json:"hostname"`
	Cores         int       `json:"cores"`
	Workers       int       `json:"workers"`
	Alive         bool      `json:"alive"`
	ShardRows     int       `json:"shard_rows"`
	Steps         int64     `json:"steps"`
	AvgLatencyMs  float64   `json:"avg_latency_ms"`  // ida y vuelta de un STEP
	AvgComputeMs  float64   `json:"avg_compute_ms"`  // tiempo de cómputo dentro del nodo
	PingMs        float64   `json:"ping_ms"`
	LastError     string    `json:"last_error,omitempty"`
	LastSeen      time.Time `json:"last_seen"`
}

// JobConfig son los parámetros de un entrenamiento distribuido.
type JobConfig struct {
	JobID    string  `json:"job_id"`
	DataPath string  `json:"data_path"`
	Limit    int     `json:"limit"`
	TestFrac float64 `json:"test_frac"`
	Seed     int64   `json:"seed"`
	Epochs   int     `json:"epochs"`
	Batch    int     `json:"batch"`
	LR       float64 `json:"lr"`
	Hidden   []int   `json:"hidden"`
}

// Progress se emite durante el entrenamiento (para WebSocket y Redis).
type Progress struct {
	JobID       string  `json:"job_id"`
	Epoch       int     `json:"epoch"`
	Epochs      int     `json:"epochs"`
	Batch       int     `json:"batch"`
	Batches     int     `json:"batches"`
	Loss        float64 `json:"loss"`
	AliveNodes  int     `json:"alive_nodes"`
	ElapsedSec  float64 `json:"elapsed_sec"`
	EpochDone   bool    `json:"epoch_done"`
	EpochSecond float64 `json:"epoch_seconds,omitempty"`
}

// Result es el resultado de un entrenamiento distribuido.
type Result struct {
	Model    *mlp.MLP         `json:"-"`
	Encoder  *mlp.Encoder     `json:"-"`
	History  []mlp.EpochStats `json:"history"`
	Test     mlp.Evaluation   `json:"test"`
	Baseline mlp.Evaluation   `json:"baseline_test"`
	Seconds  float64          `json:"train_seconds"`
	Nodes    []NodeStatus     `json:"nodes"`
	TrainRow int              `json:"train_rows"`
	TestRows int              `json:"test_rows"`
	// Tiempo medio por paso en el coordinador y la parte que es comunicación.
	AvgStepMs float64 `json:"avg_step_ms"`
	AvgCommMs float64 `json:"avg_comm_ms"`
}

// Coordinator gestiona los nodos ML.
type Coordinator struct {
	Addrs       []string
	DialTimeout time.Duration
	StepTimeout time.Duration
	Logf        func(string, ...any)

	mu     sync.Mutex // protege status
	status map[string]*NodeStatus
}

// NewCoordinator crea un coordinador para las direcciones dadas (host:puerto).
func NewCoordinator(addrs []string, logf func(string, ...any)) *Coordinator {
	c := &Coordinator{Addrs: addrs, DialTimeout: 3 * time.Second, StepTimeout: 60 * time.Second,
		Logf: logf, status: map[string]*NodeStatus{}}
	for _, a := range addrs {
		c.status[a] = &NodeStatus{Addr: a}
	}
	return c
}

// Status devuelve una copia del estado de los nodos, ordenada por dirección.
func (c *Coordinator) Status() []NodeStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]NodeStatus, 0, len(c.status))
	for _, s := range c.status {
		out = append(out, *s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Addr < out[j].Addr })
	return out
}

func (c *Coordinator) update(addr string, f func(*NodeStatus)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	s, ok := c.status[addr]
	if !ok {
		s = &NodeStatus{Addr: addr}
		c.status[addr] = s
	}
	f(s)
}

// dial abre la conexión y lee el HELLO del nodo.
func (c *Coordinator) dial(addr string) (*Conn, *Hello, error) {
	raw, err := net.DialTimeout("tcp", addr, c.DialTimeout)
	if err != nil {
		return nil, nil, err
	}
	conn := NewConn(raw)
	m, err := conn.Recv(5 * time.Second)
	if err != nil || m.Type != MsgHello {
		conn.Close()
		return nil, nil, fmt.Errorf("handshake con %s falló: %v", addr, err)
	}
	return conn, m.Hello, nil
}

// Probe consulta a todos los nodos en paralelo (HELLO + PING) y actualiza su
// estado. Lo usa el endpoint de métricas del clúster.
func (c *Coordinator) Probe() []NodeStatus {
	var wg sync.WaitGroup
	for _, addr := range c.Addrs {
		wg.Add(1)
		go func(addr string) {
			defer wg.Done()
			ini := time.Now()
			conn, h, err := c.dial(addr)
			if err != nil {
				c.update(addr, func(s *NodeStatus) { s.Alive = false; s.LastError = err.Error() })
				return
			}
			defer conn.Close()
			conn.Send(Envelope{Type: MsgPing}, 3*time.Second)
			m, err := conn.Recv(3 * time.Second)
			ping := float64(time.Since(ini).Microseconds()) / 1000
			c.update(addr, func(s *NodeStatus) {
				s.ID, s.Hostname, s.Cores, s.Workers = h.ID, h.Hostname, h.Cores, h.Workers
				s.Alive = err == nil && m.Type == MsgPong
				s.PingMs = ping
				s.LastSeen = time.Now()
				if err != nil {
					s.LastError = err.Error()
				} else {
					s.LastError = ""
				}
			})
		}(addr)
	}
	wg.Wait()
	return c.Status()
}

type nodoActivo struct {
	addr   string
	conn   *Conn
	shard  int
	alive  bool
	steps  int64
	latSum float64
	cmpSum float64
}

type respuesta struct {
	idx       int
	grad      []float64
	loss      float64
	n         int
	latMs     float64
	computeMs float64
	err       error
}

// Train ejecuta un entrenamiento distribuido síncrono.
func (c *Coordinator) Train(ctx context.Context, cfg JobConfig, onProgress func(Progress)) (*Result, error) {
	if cfg.Epochs < 1 {
		cfg.Epochs = 1
	}
	if cfg.Batch < 1 {
		cfg.Batch = 4096
	}
	if cfg.LR <= 0 {
		cfg.LR = 0.003
	}
	if cfg.TestFrac <= 0 {
		cfg.TestFrac = 0.2
	}
	if len(cfg.Hidden) == 0 {
		cfg.Hidden = []int{64, 32}
	}
	if onProgress == nil {
		onProgress = func(Progress) {}
	}

	// 1) El coordinador lee los datos solo para ajustar el encoder con train y
	//    evaluar con test. Los nodos cargan su propio shard.
	filas, err := mlp.LoadCSV(cfg.DataPath, cfg.Limit)
	if err != nil {
		return nil, err
	}
	train, test := mlp.Split(filas, cfg.TestFrac, cfg.Seed)
	filas = nil
	enc := mlp.FitEncoder(train)
	dTest := enc.Encode(test)
	nTrain := len(train)
	train = nil
	sizes := append([]int{enc.InputDim()}, cfg.Hidden...)
	sizes = append(sizes, 1)

	// 2) Conectar con los nodos disponibles.
	var nodos []*nodoActivo
	for _, addr := range c.Addrs {
		conn, h, err := c.dial(addr)
		if err != nil {
			c.Logf("nodo %s no disponible: %v", addr, err)
			c.update(addr, func(s *NodeStatus) { s.Alive = false; s.LastError = err.Error() })
			continue
		}
		c.update(addr, func(s *NodeStatus) {
			s.ID, s.Hostname, s.Cores, s.Workers, s.Alive, s.LastError = h.ID, h.Hostname, h.Cores, h.Workers, true, ""
			s.LastSeen = time.Now()
		})
		nodos = append(nodos, &nodoActivo{addr: addr, conn: conn, alive: true})
	}
	if len(nodos) == 0 {
		return nil, errors.New("no hay nodos ML disponibles")
	}
	defer func() {
		for _, n := range nodos {
			n.conn.Send(Envelope{Type: MsgStop}, 2*time.Second)
			n.conn.Close()
		}
	}()
	K := len(nodos)

	// 3) Asignar shards en paralelo y esperar READY de todos (barrera).
	errs := make(chan error, K)
	for i, n := range nodos {
		n.shard = i
		go func(n *nodoActivo) {
			err := n.conn.Send(Envelope{Type: MsgAssign, Assign: &Assign{JobID: cfg.JobID, DataPath: cfg.DataPath,
				Limit: cfg.Limit, TestFrac: cfg.TestFrac, Seed: cfg.Seed, ShardIndex: n.shard, ShardCount: K,
				Encoder: *enc, Sizes: sizes}}, 10*time.Second)
			if err == nil {
				var m Envelope
				m, err = n.conn.Recv(5 * time.Minute)
				if err == nil && m.Type != MsgReady {
					err = fmt.Errorf("nodo %s: %s", n.addr, m.Error)
				}
				if err == nil {
					c.update(n.addr, func(s *NodeStatus) { s.ShardRows = m.Ready.Rows })
				}
			}
			errs <- err
		}(n)
	}
	for range nodos {
		if err := <-errs; err != nil {
			return nil, err
		}
	}
	c.Logf("trabajo %s: %d nodos listos, %d filas de train", cfg.JobID, K, nTrain)

	// 4) Bucle de entrenamiento síncrono.
	model := mlp.NewMLP(sizes, cfg.Seed)
	opt := mlp.NewAdam(model.NumParams(), cfg.LR)
	total := make([]float64, model.NumParams())
	resps := make(chan respuesta, K)
	porNodo := make([][]int32, K)
	batches := (nTrain + cfg.Batch - 1) / cfg.Batch
	inicio := time.Now()
	var hist []mlp.EpochStats
	var pasos int
	var stepMsSum, commMsSum float64

	for ep := 1; ep <= cfg.Epochs; ep++ {
		iniEp := time.Now()
		perm := mlp.EpochOrder(nTrain, cfg.Seed, ep)
		var perdida float64
		var filasEp int
		for b := 0; b < batches; b++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			iniStep := time.Now()
			lote := perm[b*cfg.Batch : min((b+1)*cfg.Batch, nTrain)]
			// Scatter: cada fila va al nodo dueño de su shard.
			for i := range porNodo {
				porNodo[i] = porNodo[i][:0]
			}
			for _, g := range lote {
				porNodo[g%K] = append(porNodo[g%K], int32(g))
			}
			enviados := 0
			for i, n := range nodos {
				if !n.alive || len(porNodo[i]) == 0 {
					continue
				}
				enviados++
				go c.paso(i, n, model.Params, porNodo[i], resps)
			}
			// Barrera: esperar la respuesta de cada nodo.
			rs := make([]respuesta, 0, enviados)
			for ; enviados > 0; enviados-- {
				rs = append(rs, <-resps)
			}
			sort.Slice(rs, func(i, j int) bool { return rs[i].idx < rs[j].idx }) // suma en orden fijo
			// Reducción: sumar gradientes de los nodos.
			for k := range total {
				total[k] = 0
			}
			nLote := 0
			var maxComp float64
			for _, r := range rs {
				n := nodos[r.idx]
				if r.err != nil {
					n.alive = false
					c.Logf("nodo %s falló y queda fuera del entrenamiento: %v", n.addr, r.err)
					c.update(n.addr, func(s *NodeStatus) { s.Alive = false; s.LastError = r.err.Error() })
					continue
				}
				for k, v := range r.grad {
					total[k] += v
				}
				perdida += r.loss
				nLote += r.n
				n.steps++
				n.latSum += r.latMs
				n.cmpSum += r.computeMs
				maxComp = max(maxComp, r.computeMs)
			}
			if nLote == 0 {
				return nil, errors.New("todos los nodos fallaron")
			}
			escala := 1 / float64(nLote)
			for k := range total {
				total[k] *= escala
			}
			opt.Step(model.Params, total) // ÚNICA escritura de los pesos
			filasEp += nLote
			stepMs := float64(time.Since(iniStep).Microseconds()) / 1000
			stepMsSum += stepMs
			commMsSum += max(0, stepMs-maxComp)
			pasos++
			if b%25 == 0 || b == batches-1 {
				onProgress(Progress{JobID: cfg.JobID, Epoch: ep, Epochs: cfg.Epochs, Batch: b + 1, Batches: batches,
					Loss: perdida / float64(max(filasEp, 1)), AliveNodes: vivos(nodos), ElapsedSec: time.Since(inicio).Seconds()})
			}
		}
		st := mlp.EpochStats{Epoch: ep, Loss: perdida / float64(max(filasEp, 1)), Segundos: time.Since(iniEp).Seconds()}
		hist = append(hist, st)
		for _, n := range nodos {
			n := n
			c.update(n.addr, func(s *NodeStatus) {
				s.Steps = n.steps
				if n.steps > 0 {
					s.AvgLatencyMs = n.latSum / float64(n.steps)
					s.AvgComputeMs = n.cmpSum / float64(n.steps)
				}
				s.LastSeen = time.Now()
			})
		}
		onProgress(Progress{JobID: cfg.JobID, Epoch: ep, Epochs: cfg.Epochs, Batch: batches, Batches: batches,
			Loss: st.Loss, AliveNodes: vivos(nodos), ElapsedSec: time.Since(inicio).Seconds(), EpochDone: true, EpochSecond: st.Segundos})
		c.Logf("época %d/%d | pérdida %.4f | %.2f s | nodos vivos %d", ep, cfg.Epochs, st.Loss, st.Segundos, vivos(nodos))
	}
	segundos := time.Since(inicio).Seconds()

	// 5) Evaluación en test (en el coordinador, en paralelo con goroutines).
	res := &Result{Model: model, Encoder: enc, History: hist, Seconds: segundos, TrainRow: nTrain, TestRows: dTest.Len(),
		Test: mlp.Evaluate(model, enc, dTest, 8), Baseline: mlp.Baseline(enc, dTest)}
	if pasos > 0 {
		res.AvgStepMs = stepMsSum / float64(pasos)
		res.AvgCommMs = commMsSum / float64(pasos)
	}
	estados := c.Status()
	for _, s := range estados {
		for _, n := range nodos {
			if n.addr == s.Addr {
				res.Nodes = append(res.Nodes, s)
			}
		}
	}
	return res, nil
}

// paso envía un STEP a un nodo y espera su GRAD. Corre en su propia goroutine
// (fan-out); el resultado vuelve por el channel resps (fan-in).
func (c *Coordinator) paso(i int, n *nodoActivo, params []float64, filas []int32, resps chan<- respuesta) {
	ini := time.Now()
	err := n.conn.Send(Envelope{Type: MsgStep, Step: &Step{Params: params, Rows: filas}}, c.StepTimeout)
	if err != nil {
		resps <- respuesta{idx: i, err: err}
		return
	}
	m, err := n.conn.Recv(c.StepTimeout)
	if err == nil && m.Type != MsgGrad {
		err = fmt.Errorf("respuesta inesperada %s: %s", m.Type, m.Error)
	}
	if err != nil {
		resps <- respuesta{idx: i, err: err}
		return
	}
	resps <- respuesta{idx: i, grad: m.Grad.Grad, loss: m.Grad.Loss, n: m.Grad.N,
		latMs: float64(time.Since(ini).Microseconds()) / 1000, computeMs: m.Grad.ComputeMs}
}

func vivos(ns []*nodoActivo) int {
	c := 0
	for _, n := range ns {
		if n.alive {
			c++
		}
	}
	return c
}
