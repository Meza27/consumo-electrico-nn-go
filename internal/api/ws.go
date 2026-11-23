package api

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// Event es un mensaje JSON que se envía a los clientes WebSocket.
type Event struct {
	Type string    `json:"type"` // train_progress, train_done, cluster, prediction, log
	Time time.Time `json:"time"`
	Data any       `json:"data"`
}

// Hub reparte eventos a todos los clientes WebSocket conectados. Una sola
// goroutine (run) es dueña del conjunto de clientes: registrar, quitar y
// difundir pasan por channels, así que no hace falta mutex.
type Hub struct {
	register   chan *wsClient
	unregister chan *wsClient
	broadcast  chan []byte
	count      chan chan int
}

type wsClient struct {
	conn net.Conn
	send chan []byte
}

// NewHub crea el hub y lanza su goroutine.
func NewHub() *Hub {
	h := &Hub{register: make(chan *wsClient), unregister: make(chan *wsClient),
		broadcast: make(chan []byte, 256), count: make(chan chan int)}
	go h.run()
	return h
}

func (h *Hub) run() {
	clientes := map[*wsClient]bool{}
	for {
		select {
		case c := <-h.register:
			clientes[c] = true
		case c := <-h.unregister:
			if clientes[c] {
				delete(clientes, c)
				close(c.send)
			}
		case msg := <-h.broadcast:
			for c := range clientes {
				select {
				case c.send <- msg:
				default: // cliente lento: se desconecta antes que frenar a todos
					delete(clientes, c)
					close(c.send)
				}
			}
		case resp := <-h.count:
			resp <- len(clientes)
		}
	}
}

// Publish envía un evento a todos los clientes (no bloquea).
func (h *Hub) Publish(tipo string, data any) {
	b, err := json.Marshal(Event{Type: tipo, Time: time.Now(), Data: data})
	if err != nil {
		return
	}
	select {
	case h.broadcast <- b:
	default:
	}
}

// Clients devuelve cuántos clientes hay conectados.
func (h *Hub) Clients() int {
	r := make(chan int)
	h.count <- r
	return <-r
}

const wsGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// acceptKey calcula Sec-WebSocket-Accept (RFC 6455, sección 4.2.2).
func acceptKey(key string) string {
	h := sha1.Sum([]byte(key + wsGUID))
	return base64.StdEncoding.EncodeToString(h[:])
}

// ServeWS hace el handshake de WebSocket y registra al cliente.
func (h *Hub) ServeWS(w http.ResponseWriter, r *http.Request) {
	if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") || r.Header.Get("Sec-WebSocket-Key") == "" {
		http.Error(w, "se esperaba una conexión WebSocket", http.StatusBadRequest)
		return
	}
	hj, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "hijack no soportado", http.StatusInternalServerError)
		return
	}
	conn, rw, err := hj.Hijack()
	if err != nil {
		return
	}
	rw.WriteString("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" +
		"Sec-WebSocket-Accept: " + acceptKey(r.Header.Get("Sec-WebSocket-Key")) + "\r\n\r\n")
	if err := rw.Flush(); err != nil {
		conn.Close()
		return
	}
	c := &wsClient{conn: conn, send: make(chan []byte, 64)}
	h.register <- c
	go c.escribir()
	go c.leer(h, rw.Reader)
}

// escribir envía los mensajes del channel como frames de texto y un ping cada 30 s.
func (c *wsClient) escribir() {
	ping := time.NewTicker(30 * time.Second)
	defer func() { ping.Stop(); c.conn.Close() }()
	for {
		select {
		case msg, ok := <-c.send:
			if !ok {
				writeFrame(c.conn, 0x8, nil) // close
				return
			}
			if err := writeFrame(c.conn, 0x1, msg); err != nil {
				return
			}
		case <-ping.C:
			if err := writeFrame(c.conn, 0x9, nil); err != nil {
				return
			}
		}
	}
}

// leer consume los frames del cliente (que llegan enmascarados) para detectar
// el cierre de la conexión.
func (c *wsClient) leer(h *Hub, r *bufio.Reader) {
	defer func() { h.unregister <- c }()
	for {
		op, _, err := readFrame(r)
		if err != nil || op == 0x8 {
			return
		}
	}
}

// writeFrame escribe un frame sin máscara (servidor -> cliente).
func writeFrame(w io.Writer, opcode byte, payload []byte) error {
	hdr := []byte{0x80 | opcode}
	n := len(payload)
	switch {
	case n < 126:
		hdr = append(hdr, byte(n))
	case n < 1<<16:
		hdr = append(hdr, 126, byte(n>>8), byte(n))
	default:
		hdr = append(hdr, 127)
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], uint64(n))
		hdr = append(hdr, b[:]...)
	}
	if c, ok := w.(net.Conn); ok {
		c.SetWriteDeadline(time.Now().Add(10 * time.Second))
	}
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// readFrame lee un frame (cliente -> servidor, con máscara de 4 bytes).
func readFrame(r *bufio.Reader) (byte, []byte, error) {
	var h [2]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		return 0, nil, err
	}
	op := h[0] & 0x0F
	masked := h[1]&0x80 != 0
	n := uint64(h[1] & 0x7F)
	switch n {
	case 126:
		var b [2]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, nil, err
		}
		n = uint64(binary.BigEndian.Uint16(b[:]))
	case 127:
		var b [8]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, nil, err
		}
		n = binary.BigEndian.Uint64(b[:])
	}
	if n > 1<<20 {
		return 0, nil, errors.New("frame demasiado grande")
	}
	var mask [4]byte
	if masked {
		if _, err := io.ReadFull(r, mask[:]); err != nil {
			return 0, nil, err
		}
	}
	p := make([]byte, n)
	if _, err := io.ReadFull(r, p); err != nil {
		return 0, nil, err
	}
	if masked {
		for i := range p {
			p[i] ^= mask[i%4]
		}
	}
	return op, p, nil
}
