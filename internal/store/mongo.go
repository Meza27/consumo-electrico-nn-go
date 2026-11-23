package store

import (
	"bufio"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"sync/atomic"
	"time"
)

// Mongo es un cliente mínimo de MongoDB (protocolo OP_MSG, MongoDB 3.6+).
// Mantiene un pool de conexiones TCP en un channel con buffer: cada consulta
// toma una conexión, la usa en exclusiva y la devuelve (sin mutex).
type Mongo struct {
	addr    string
	DB      string
	pool    chan *mongoConn
	reqID   atomic.Int32
	Timeout time.Duration
}

type mongoConn struct {
	c net.Conn
	r *bufio.Reader
}

const opMsg = 2013

// DialMongo abre `size` conexiones contra addr (host:27017) y prueba con ping.
func DialMongo(addr, db string, size int) (*Mongo, error) {
	if size < 1 {
		size = 4
	}
	m := &Mongo{addr: addr, DB: db, pool: make(chan *mongoConn, size), Timeout: 10 * time.Second}
	for i := 0; i < size; i++ {
		c, err := m.nueva()
		if err != nil {
			m.Close()
			return nil, err
		}
		m.pool <- c
	}
	if _, err := m.Command(D{{"ping", 1}}); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

func (m *Mongo) nueva() (*mongoConn, error) {
	c, err := net.DialTimeout("tcp", m.addr, 5*time.Second)
	if err != nil {
		return nil, err
	}
	return &mongoConn{c: c, r: bufio.NewReader(c)}, nil
}

// Close cierra las conexiones libres del pool.
func (m *Mongo) Close() {
	for {
		select {
		case c := <-m.pool:
			c.c.Close()
		default:
			return
		}
	}
}

// Ping comprueba la conexión con el servidor.
func (m *Mongo) Ping() (M, error) { return m.Command(D{{"ping", 1}}) }

// Command ejecuta un comando sobre la base m.DB y devuelve la respuesta.
func (m *Mongo) Command(cmd D) (M, error) {
	cmd = append(cmd, E{"$db", m.DB})
	body, err := Marshal(cmd)
	if err != nil {
		return nil, err
	}
	c := <-m.pool
	resp, err := m.roundTrip(c, body)
	if err != nil {
		// La conexión puede haber quedado en mal estado: se reemplaza.
		c.c.Close()
		if nc, e := m.nueva(); e == nil {
			c = nc
		}
		m.pool <- c
		return nil, err
	}
	m.pool <- c
	if ToFloat(resp["ok"]) != 1 {
		return resp, fmt.Errorf("mongo: %v (code %v)", resp["errmsg"], resp["code"])
	}
	if we, ok := resp["writeErrors"].([]any); ok && len(we) > 0 {
		if e, ok := we[0].(M); ok {
			return resp, fmt.Errorf("mongo: %v", e["errmsg"])
		}
	}
	return resp, nil
}

// roundTrip envía un OP_MSG y lee la respuesta.
//
//	header: longitud(int32) requestID(int32) responseTo(int32) opCode(int32=2013)
//	cuerpo: flagBits(uint32) + sección tipo 0 (un documento BSON)
func (m *Mongo) roundTrip(c *mongoConn, doc []byte) (M, error) {
	id := m.reqID.Add(1)
	msg := make([]byte, 16+4+1+len(doc))
	binary.LittleEndian.PutUint32(msg[0:], uint32(len(msg)))
	binary.LittleEndian.PutUint32(msg[4:], uint32(id))
	binary.LittleEndian.PutUint32(msg[12:], opMsg)
	msg[20] = 0 // sección tipo 0
	copy(msg[21:], doc)
	c.c.SetDeadline(time.Now().Add(m.Timeout))
	if _, err := c.c.Write(msg); err != nil {
		return nil, err
	}
	var hdr [16]byte
	if _, err := io.ReadFull(c.r, hdr[:]); err != nil {
		return nil, err
	}
	n := int(binary.LittleEndian.Uint32(hdr[0:]))
	if binary.LittleEndian.Uint32(hdr[12:]) != opMsg || n < 21 || n > 48<<20 {
		return nil, errors.New("mongo: respuesta inválida")
	}
	resto := make([]byte, n-16)
	if _, err := io.ReadFull(c.r, resto); err != nil {
		return nil, err
	}
	flags := binary.LittleEndian.Uint32(resto)
	cuerpo := resto[4:]
	if flags&1 == 1 { // checksumPresent: los últimos 4 bytes son CRC
		cuerpo = cuerpo[:len(cuerpo)-4]
	}
	if len(cuerpo) < 1 || cuerpo[0] != 0 {
		return nil, errors.New("mongo: sección no soportada")
	}
	return Unmarshal(cuerpo[1:])
}

// ---------------------------------------------------------------- operaciones

var oidCounter atomic.Uint32

func init() {
	var b [4]byte
	rand.Read(b[:])
	oidCounter.Store(binary.BigEndian.Uint32(b[:]))
}

// NewObjectID genera un ObjectID: 4 bytes de tiempo + 5 aleatorios + 3 de contador.
func NewObjectID() ObjectID {
	var o ObjectID
	binary.BigEndian.PutUint32(o[0:], uint32(time.Now().Unix()))
	rand.Read(o[4:9])
	c := oidCounter.Add(1)
	o[9], o[10], o[11] = byte(c>>16), byte(c>>8), byte(c)
	return o
}

// InsertOne inserta un documento y devuelve su _id.
func (m *Mongo) InsertOne(coll string, doc D) (ObjectID, error) {
	id := NewObjectID()
	if doc.Get("_id") == nil {
		doc = append(D{{"_id", id}}, doc...)
	}
	_, err := m.Command(D{{"insert", coll}, {"documents", A{doc}}})
	return id, err
}

// InsertMany inserta varios documentos de una vez.
func (m *Mongo) InsertMany(coll string, docs []D) error {
	if len(docs) == 0 {
		return nil
	}
	arr := make(A, len(docs))
	for i, d := range docs {
		if d.Get("_id") == nil {
			d = append(D{{"_id", NewObjectID()}}, d...)
		}
		arr[i] = d
	}
	_, err := m.Command(D{{"insert", coll}, {"documents", arr}})
	return err
}

// Find devuelve hasta `limit` documentos (un solo lote, sin cursores abiertos).
func (m *Mongo) Find(coll string, filter, sort D, limit int) ([]M, error) {
	if filter == nil {
		filter = D{}
	}
	cmd := D{{"find", coll}, {"filter", filter}}
	if len(sort) > 0 {
		cmd = append(cmd, E{"sort", sort})
	}
	if limit > 0 {
		cmd = append(cmd, E{"limit", limit}, E{"batchSize", limit})
	}
	cmd = append(cmd, E{"singleBatch", true})
	resp, err := m.Command(cmd)
	if err != nil {
		return nil, err
	}
	return lote(resp), nil
}

// FindOne devuelve el primer documento o nil.
func (m *Mongo) FindOne(coll string, filter, sort D) (M, error) {
	docs, err := m.Find(coll, filter, sort, 1)
	if err != nil || len(docs) == 0 {
		return nil, err
	}
	return docs[0], nil
}

// Count cuenta los documentos que cumplen el filtro.
func (m *Mongo) Count(coll string, filter D) (int, error) {
	if filter == nil {
		filter = D{}
	}
	resp, err := m.Command(D{{"count", coll}, {"query", filter}})
	if err != nil {
		return 0, err
	}
	return ToInt(resp["n"]), nil
}

// UpdateOne aplica `update` (p. ej. D{{"$set", D{...}}}) al primer documento.
func (m *Mongo) UpdateOne(coll string, filter, update D, upsert bool) error {
	_, err := m.Command(D{{"update", coll}, {"updates", A{D{{"q", filter}, {"u", update}, {"upsert", upsert}}}}})
	return err
}

// UpdateMany aplica `update` a todos los documentos del filtro.
func (m *Mongo) UpdateMany(coll string, filter, update D) error {
	_, err := m.Command(D{{"update", coll}, {"updates", A{D{{"q", filter}, {"u", update}, {"multi", true}}}}})
	return err
}

// Aggregate ejecuta un pipeline de agregación.
func (m *Mongo) Aggregate(coll string, pipeline []D) ([]M, error) {
	arr := make(A, len(pipeline))
	for i, p := range pipeline {
		arr[i] = p
	}
	resp, err := m.Command(D{{"aggregate", coll}, {"pipeline", arr}, {"cursor", D{{"batchSize", 1000}}}})
	if err != nil {
		return nil, err
	}
	return lote(resp), nil
}

// CreateIndex crea un índice (si ya existe no hace nada).
func (m *Mongo) CreateIndex(coll, name string, keys D, unique bool) error {
	_, err := m.Command(D{{"createIndexes", coll}, {"indexes", A{D{{"key", keys}, {"name", name}, {"unique", unique}}}}})
	return err
}

func lote(resp M) []M {
	cur, _ := resp["cursor"].(M)
	batch, _ := cur["firstBatch"].([]any)
	out := make([]M, 0, len(batch))
	for _, d := range batch {
		if doc, ok := d.(M); ok {
			out = append(out, doc)
		}
	}
	return out
}
