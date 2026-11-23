// Package cluster implementa el entrenamiento DISTRIBUIDO del MLP entre varios
// nodos ML comunicados por TCP (PC4).
//
// Topología en estrella (parameter server síncrono):
//
//	                coordinador (API)
//	              /       |        \        TCP (gob)
//	         nodo 1    nodo 2 ...  nodo K
//
// Cada nodo guarda solo SU fragmento (shard) del conjunto de entrenamiento: la
// fila global i pertenece al nodo i % K. En cada paso el coordinador envía a
// cada nodo los pesos actuales y las filas del mini-lote que le pertenecen; el
// nodo calcula el gradiente con su GradPool (goroutines) y lo devuelve. El
// coordinador espera a todos (barrera), suma los gradientes, promedia y es el
// ÚNICO que actualiza los pesos con Adam.
package cluster

import (
	"encoding/gob"
	"net"
	"time"

	"consumo-electrico-nn-go/internal/mlp"
)

// Tipos de mensaje del protocolo.
const (
	MsgHello  = "HELLO"  // nodo -> coordinador, al conectarse
	MsgAssign = "ASSIGN" // coordinador -> nodo: shard + encoder + arquitectura
	MsgReady  = "READY"  // nodo -> coordinador: shard cargado
	MsgStep   = "STEP"   // coordinador -> nodo: pesos + filas del lote
	MsgGrad   = "GRAD"   // nodo -> coordinador: gradiente sumado + pérdida
	MsgPing   = "PING"   // coordinador -> nodo: chequeo de salud
	MsgPong   = "PONG"   // nodo -> coordinador
	MsgStop   = "STOP"   // coordinador -> nodo: fin del trabajo
	MsgError  = "ERROR"  // nodo -> coordinador
)

// Hello describe al nodo.
type Hello struct {
	ID       string
	Hostname string
	Cores    int
	Workers  int
}

// Assign le dice al nodo qué parte de los datos le toca y con qué codificador.
// El codificador se ajustó SOLO con train en el coordinador.
type Assign struct {
	JobID      string
	DataPath   string
	Limit      int
	TestFrac   float64
	Seed       int64
	ShardIndex int
	ShardCount int
	Encoder    mlp.Encoder
	Sizes      []int
}

// Ready confirma que el shard está cargado.
type Ready struct {
	Rows        int
	LoadSeconds float64
}

// Step es un paso de entrenamiento: pesos actuales y filas GLOBALES del lote
// que pertenecen a este nodo.
type Step struct {
	Params []float64
	Rows   []int32
}

// Grad es la respuesta a un Step.
type Grad struct {
	Grad      []float64
	Loss      float64
	N         int
	ComputeMs float64
}

// Pong es la respuesta a un Ping.
type Pong struct {
	Rows      int
	Steps     int64
	UptimeSec float64
}

// Envelope es lo que viaja por el socket (un mensaje por envío).
type Envelope struct {
	Type   string
	Hello  *Hello
	Assign *Assign
	Ready  *Ready
	Step   *Step
	Grad   *Grad
	Pong   *Pong
	Error  string
}

// Conn envuelve una conexión TCP con codificación gob en ambos sentidos.
// gob es un formato binario de la librería estándar: los []float64 viajan
// compactos y sin conversión a texto.
type Conn struct {
	raw net.Conn
	enc *gob.Encoder
	dec *gob.Decoder
}

// NewConn prepara los codificadores sobre una conexión ya abierta.
func NewConn(c net.Conn) *Conn {
	if tc, ok := c.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
		tc.SetKeepAlive(true)
	}
	return &Conn{raw: c, enc: gob.NewEncoder(c), dec: gob.NewDecoder(c)}
}

// Send envía un mensaje con un plazo máximo.
func (c *Conn) Send(m Envelope, timeout time.Duration) error {
	if timeout > 0 {
		c.raw.SetWriteDeadline(time.Now().Add(timeout))
	}
	return c.enc.Encode(&m)
}

// Recv espera un mensaje con un plazo máximo (0 = sin plazo).
func (c *Conn) Recv(timeout time.Duration) (Envelope, error) {
	if timeout > 0 {
		c.raw.SetReadDeadline(time.Now().Add(timeout))
	} else {
		c.raw.SetReadDeadline(time.Time{})
	}
	var m Envelope
	err := c.dec.Decode(&m)
	return m, err
}

// Close cierra la conexión.
func (c *Conn) Close() error { return c.raw.Close() }

// RemoteAddr devuelve la dirección del otro extremo.
func (c *Conn) RemoteAddr() string { return c.raw.RemoteAddr().String() }
