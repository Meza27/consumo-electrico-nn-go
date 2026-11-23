package api

import (
	"bufio"
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestJWT(t *testing.T) {
	sec := []byte("secreto")
	tok, err := NewJWT(sec, "admin", "admin", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ParseJWT(sec, tok)
	if err != nil || c.Sub != "admin" || c.Role != "admin" {
		t.Fatalf("token válido rechazado: %v %+v", err, c)
	}
	if _, err := ParseJWT([]byte("otro"), tok); err == nil {
		t.Fatal("aceptó un token firmado con otro secreto")
	}
	partes := strings.Split(tok, ".")
	falso := partes[0] + "." + b64.EncodeToString([]byte(`{"sub":"x","role":"admin","exp":9999999999}`)) + "." + partes[2]
	if _, err := ParseJWT(sec, falso); err == nil {
		t.Fatal("aceptó un payload modificado")
	}
	vencido, _ := NewJWT(sec, "a", "user", -time.Minute)
	if _, err := ParseJWT(sec, vencido); err == nil {
		t.Fatal("aceptó un token vencido")
	}
}

func TestPassword(t *testing.T) {
	h := HashPassword("admin123")
	if !CheckPassword("admin123", h) || CheckPassword("admin124", h) {
		t.Fatal("verificación de contraseña incorrecta")
	}
	if HashPassword("admin123") == h {
		t.Fatal("dos hashes iguales: falta la sal aleatoria")
	}
}

// Ejemplo de la RFC 6455 (sección 1.3).
func TestWebSocketAcceptKey(t *testing.T) {
	if got := acceptKey("dGhlIHNhbXBsZSBub25jZQ=="); got != "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" {
		t.Fatalf("accept key = %s", got)
	}
}

func TestWebSocketFrames(t *testing.T) {
	for _, n := range []int{5, 300, 70000} {
		var buf bytes.Buffer
		msg := bytes.Repeat([]byte("a"), n)
		if err := writeFrame(&buf, 0x1, msg); err != nil {
			t.Fatal(err)
		}
		op, p, err := readFrame(bufio.NewReader(&buf))
		if err != nil || op != 0x1 || !bytes.Equal(p, msg) {
			t.Fatalf("n=%d: frame mal leído (%v)", n, err)
		}
	}
	// Frame enmascarado como los que envía un navegador: "Hello" (RFC 6455, 5.7).
	enmascarado := []byte{0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58}
	op, p, err := readFrame(bufio.NewReader(bytes.NewReader(enmascarado)))
	if err != nil || op != 1 || string(p) != "Hello" {
		t.Fatalf("frame enmascarado mal leído: %q %v", p, err)
	}
}

func TestHub(t *testing.T) {
	h := NewHub()
	if h.Clients() != 0 {
		t.Fatal("hub nuevo con clientes")
	}
	c := &wsClient{send: make(chan []byte, 4)}
	h.register <- c
	h.Publish("log", "hola")
	select {
	case m := <-c.send:
		if !strings.Contains(string(m), `"type":"log"`) {
			t.Fatalf("evento inesperado: %s", m)
		}
	case <-time.After(time.Second):
		t.Fatal("el evento no llegó")
	}
	h.unregister <- c
	if h.Clients() != 0 {
		t.Fatal("el cliente no se quitó")
	}
}
