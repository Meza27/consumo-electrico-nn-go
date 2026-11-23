package cluster

import (
	"fmt"
	"log"
	"net"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"consumo-electrico-nn-go/internal/mlp"
)

// Node es un nodo ML: escucha en un puerto TCP y atiende al coordinador.
type Node struct {
	ID      string
	Workers int
	Logf    func(format string, args ...any)

	inicio time.Time
	steps  atomic.Int64

	mu    sync.Mutex // protege cache (filas leídas del CSV, por ruta)
	cache map[string][]mlp.Row
}

// NewNode crea un nodo con P goroutines de cómputo.
func NewNode(id string, workers int) *Node {
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	return &Node{ID: id, Workers: workers, Logf: log.Printf, inicio: time.Now(), cache: map[string][]mlp.Row{}}
}

// ListenAndServe acepta conexiones del coordinador. Cada conexión se atiende en
// su propia goroutine (un trabajo de entrenamiento por conexión).
func (n *Node) ListenAndServe(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	n.Logf("nodo %s escuchando en %s con %d workers", n.ID, ln.Addr(), n.Workers)
	return n.Serve(ln)
}

// Serve atiende un listener ya abierto (útil en pruebas).
func (n *Node) Serve(ln net.Listener) error {
	for {
		c, err := ln.Accept()
		if err != nil {
			return err
		}
		go n.handle(NewConn(c))
	}
}

func (n *Node) handle(c *Conn) {
	defer c.Close()
	host, _ := os.Hostname()
	if err := c.Send(Envelope{Type: MsgHello, Hello: &Hello{ID: n.ID, Hostname: host, Cores: runtime.NumCPU(), Workers: n.Workers}}, 10*time.Second); err != nil {
		return
	}

	var (
		pool  *mlp.GradPool
		model *mlp.MLP
		K, k  int
		rows  int
	)
	defer func() {
		if pool != nil {
			pool.Close()
		}
	}()

	for {
		m, err := c.Recv(0)
		if err != nil {
			if pool != nil {
				n.Logf("conexión con el coordinador cerrada: %v", err)
			}
			return
		}
		switch m.Type {
		case MsgAssign:
			a := m.Assign
			ini := time.Now()
			d, err := n.cargarShard(a)
			if err != nil {
				c.Send(Envelope{Type: MsgError, Error: err.Error()}, 10*time.Second)
				continue
			}
			if pool != nil {
				pool.Close()
			}
			model = mlp.NewMLP(a.Sizes, 1)
			pool = mlp.NewGradPool(model, d, n.Workers)
			K, k, rows = a.ShardCount, a.ShardIndex, d.Len()
			n.Logf("coordinador %s -> trabajo %s: shard %d/%d con %d filas (%.2f s)", c.RemoteAddr(), a.JobID, k+1, K, rows, time.Since(ini).Seconds())
			c.Send(Envelope{Type: MsgReady, Ready: &Ready{Rows: rows, LoadSeconds: time.Since(ini).Seconds()}}, 10*time.Second)

		case MsgStep:
			if pool == nil {
				c.Send(Envelope{Type: MsgError, Error: "STEP sin ASSIGN"}, 10*time.Second)
				continue
			}
			ini := time.Now()
			copy(model.Params, m.Step.Params) // los workers están esperando: no hay carrera
			locales := make([]int, len(m.Step.Rows))
			for j, g := range m.Step.Rows {
				locales[j] = int(g) / K // fila global g -> posición en el shard
			}
			grad, loss := pool.Compute(locales)
			n.steps.Add(1)
			err := c.Send(Envelope{Type: MsgGrad, Grad: &Grad{Grad: grad, Loss: loss, N: len(locales),
				ComputeMs: float64(time.Since(ini).Microseconds()) / 1000}}, 30*time.Second)
			if err != nil {
				return
			}

		case MsgPing:
			c.Send(Envelope{Type: MsgPong, Pong: &Pong{Rows: rows, Steps: n.steps.Load(), UptimeSec: time.Since(n.inicio).Seconds()}}, 10*time.Second)

		case MsgStop:
			n.Logf("trabajo terminado (%d pasos acumulados)", n.steps.Load())
			if pool != nil {
				pool.Close()
				pool = nil
			}
		}
	}
}

// cargarShard lee el CSV limpio (una sola vez por ruta), repite el MISMO split
// train/test que el coordinador (misma semilla) y codifica solo las filas de
// train que le tocan a este nodo: la fila global i es del nodo i % K.
func (n *Node) cargarShard(a *Assign) (*mlp.Dataset, error) {
	if a.ShardCount < 1 || a.ShardIndex < 0 || a.ShardIndex >= a.ShardCount {
		return nil, fmt.Errorf("shard inválido %d/%d", a.ShardIndex, a.ShardCount)
	}
	clave := fmt.Sprintf("%s|%d", a.DataPath, a.Limit)
	n.mu.Lock()
	filas, ok := n.cache[clave]
	n.mu.Unlock()
	if !ok {
		var err error
		filas, err = mlp.LoadCSV(a.DataPath, a.Limit)
		if err != nil {
			return nil, err
		}
		n.mu.Lock()
		n.cache[clave] = filas
		n.mu.Unlock()
	}
	train, _ := mlp.Split(filas, a.TestFrac, a.Seed)
	shard := make([]mlp.Row, 0, len(train)/a.ShardCount+1)
	for i := a.ShardIndex; i < len(train); i += a.ShardCount {
		shard = append(shard, train[i])
	}
	enc := a.Encoder
	return enc.Encode(shard), nil
}
