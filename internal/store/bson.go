// Package store implementa la persistencia del sistema con la librería
// estándar de Go:
//
//   - MongoDB (persistente): cliente mínimo que habla el protocolo OP_MSG y el
//     formato binario BSON directamente sobre TCP.
//   - Redis (caché): cliente mínimo del protocolo RESP sobre TCP.
//
// Solo se implementa lo que el sistema necesita (insertar, buscar, contar,
// actualizar, índices; GET/SET/HSET/...).
package store

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"time"
)

// E es un par clave-valor de un documento BSON.
type E struct {
	Key string
	Val any
}

// D es un documento BSON ORDENADO (en los comandos de MongoDB el primer campo
// es el nombre del comando, por eso el orden importa).
type D []E

// M es un documento sin orden (lo que devuelve la decodificación).
type M = map[string]any

// A es un arreglo BSON.
type A = []any

// ObjectID es el identificador de 12 bytes de MongoDB.
type ObjectID [12]byte

// Hex devuelve el ObjectID en hexadecimal.
func (o ObjectID) Hex() string { return hex.EncodeToString(o[:]) }

// Get busca una clave en un documento ordenado.
func (d D) Get(k string) any {
	for _, e := range d {
		if e.Key == k {
			return e.Val
		}
	}
	return nil
}

// ---------------------------------------------------------------- codificación

// Marshal codifica un documento (D o M) a BSON.
func Marshal(doc any) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeDoc(&buf, doc); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writeDoc(buf *bytes.Buffer, doc any) error {
	ini := buf.Len()
	buf.Write([]byte{0, 0, 0, 0}) // tamaño, se completa al final
	switch d := doc.(type) {
	case D:
		for _, e := range d {
			if err := writeElem(buf, e.Key, e.Val); err != nil {
				return err
			}
		}
	case M:
		for k, v := range d {
			if err := writeElem(buf, k, v); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("bson: documento de tipo %T no soportado", doc)
	}
	buf.WriteByte(0)
	binary.LittleEndian.PutUint32(buf.Bytes()[ini:], uint32(buf.Len()-ini))
	return nil
}

func cstring(buf *bytes.Buffer, s string) {
	buf.WriteString(s)
	buf.WriteByte(0)
}

func writeElem(buf *bytes.Buffer, k string, v any) error {
	var tmp [8]byte
	switch x := v.(type) {
	case nil:
		buf.WriteByte(0x0A)
		cstring(buf, k)
	case float64:
		buf.WriteByte(0x01)
		cstring(buf, k)
		binary.LittleEndian.PutUint64(tmp[:], math.Float64bits(x))
		buf.Write(tmp[:])
	case float32:
		return writeElem(buf, k, float64(x))
	case string:
		buf.WriteByte(0x02)
		cstring(buf, k)
		binary.LittleEndian.PutUint32(tmp[:4], uint32(len(x)+1))
		buf.Write(tmp[:4])
		cstring(buf, x)
	case D, M:
		buf.WriteByte(0x03)
		cstring(buf, k)
		return writeDoc(buf, x)
	case []any:
		buf.WriteByte(0x04)
		cstring(buf, k)
		return writeArray(buf, x)
	case []float64:
		arr := make([]any, len(x))
		for i, f := range x {
			arr[i] = f
		}
		return writeElem(buf, k, arr)
	case []string:
		arr := make([]any, len(x))
		for i, s := range x {
			arr[i] = s
		}
		return writeElem(buf, k, arr)
	case []int:
		arr := make([]any, len(x))
		for i, n := range x {
			arr[i] = n
		}
		return writeElem(buf, k, arr)
	case []D:
		arr := make([]any, len(x))
		for i, d := range x {
			arr[i] = d
		}
		return writeElem(buf, k, arr)
	case ObjectID:
		buf.WriteByte(0x07)
		cstring(buf, k)
		buf.Write(x[:])
	case bool:
		buf.WriteByte(0x08)
		cstring(buf, k)
		if x {
			buf.WriteByte(1)
		} else {
			buf.WriteByte(0)
		}
	case time.Time:
		buf.WriteByte(0x09)
		cstring(buf, k)
		binary.LittleEndian.PutUint64(tmp[:], uint64(x.UnixMilli()))
		buf.Write(tmp[:])
	case int32:
		buf.WriteByte(0x10)
		cstring(buf, k)
		binary.LittleEndian.PutUint32(tmp[:4], uint32(x))
		buf.Write(tmp[:4])
	case int:
		if x >= math.MinInt32 && x <= math.MaxInt32 {
			return writeElem(buf, k, int32(x))
		}
		return writeElem(buf, k, int64(x))
	case int64:
		buf.WriteByte(0x12)
		cstring(buf, k)
		binary.LittleEndian.PutUint64(tmp[:], uint64(x))
		buf.Write(tmp[:])
	default:
		return fmt.Errorf("bson: tipo %T no soportado en %q", v, k)
	}
	return nil
}

func writeArray(buf *bytes.Buffer, arr []any) error {
	d := make(D, len(arr))
	for i, v := range arr {
		d[i] = E{fmt.Sprint(i), v}
	}
	return writeDoc(buf, d)
}

// ---------------------------------------------------------------- decodificación

var errCorto = errors.New("bson: documento truncado")

// Unmarshal decodifica un documento BSON a M (mapa).
func Unmarshal(b []byte) (M, error) {
	m, _, err := readDoc(b)
	return m, err
}

func readDoc(b []byte) (M, int, error) {
	if len(b) < 5 {
		return nil, 0, errCorto
	}
	n := int(binary.LittleEndian.Uint32(b))
	if n > len(b) || n < 5 {
		return nil, 0, errCorto
	}
	out := M{}
	p := 4
	for p < n-1 {
		t := b[p]
		p++
		fin := bytes.IndexByte(b[p:n], 0)
		if fin < 0 {
			return nil, 0, errCorto
		}
		k := string(b[p : p+fin])
		p += fin + 1
		v, used, err := readVal(t, b[p:n])
		if err != nil {
			return nil, 0, fmt.Errorf("campo %q: %w", k, err)
		}
		out[k] = v
		p += used
	}
	return out, n, nil
}

func readVal(t byte, b []byte) (any, int, error) {
	need := func(k int) error {
		if len(b) < k {
			return errCorto
		}
		return nil
	}
	switch t {
	case 0x01:
		if err := need(8); err != nil {
			return nil, 0, err
		}
		return math.Float64frombits(binary.LittleEndian.Uint64(b)), 8, nil
	case 0x02, 0x0D, 0x0E: // string, javascript, symbol
		if err := need(4); err != nil {
			return nil, 0, err
		}
		l := int(binary.LittleEndian.Uint32(b))
		if err := need(4 + l); err != nil || l < 1 {
			return nil, 0, errCorto
		}
		return string(b[4 : 4+l-1]), 4 + l, nil
	case 0x03:
		m, n, err := readDoc(b)
		return m, n, err
	case 0x04:
		m, n, err := readDoc(b)
		if err != nil {
			return nil, 0, err
		}
		arr := make([]any, len(m))
		for k, v := range m {
			var i int
			fmt.Sscan(k, &i)
			if i >= 0 && i < len(arr) {
				arr[i] = v
			}
		}
		return arr, n, nil
	case 0x05: // binario
		if err := need(5); err != nil {
			return nil, 0, err
		}
		l := int(binary.LittleEndian.Uint32(b))
		if err := need(5 + l); err != nil {
			return nil, 0, err
		}
		return append([]byte(nil), b[5:5+l]...), 5 + l, nil
	case 0x06, 0x0A, 0xFF, 0x7F: // undefined, null, min/max key
		return nil, 0, nil
	case 0x07:
		if err := need(12); err != nil {
			return nil, 0, err
		}
		var o ObjectID
		copy(o[:], b[:12])
		return o, 12, nil
	case 0x08:
		if err := need(1); err != nil {
			return nil, 0, err
		}
		return b[0] == 1, 1, nil
	case 0x09:
		if err := need(8); err != nil {
			return nil, 0, err
		}
		return time.UnixMilli(int64(binary.LittleEndian.Uint64(b))).UTC(), 8, nil
	case 0x10:
		if err := need(4); err != nil {
			return nil, 0, err
		}
		return int32(binary.LittleEndian.Uint32(b)), 4, nil
	case 0x11, 0x12: // timestamp, int64
		if err := need(8); err != nil {
			return nil, 0, err
		}
		return int64(binary.LittleEndian.Uint64(b)), 8, nil
	case 0x13: // decimal128 (no se usa: se devuelve en bruto)
		if err := need(16); err != nil {
			return nil, 0, err
		}
		return append([]byte(nil), b[:16]...), 16, nil
	}
	return nil, 0, fmt.Errorf("bson: tipo 0x%02x no soportado", t)
}

// ---------------------------------------------------------------- utilidades

// ToFloat convierte un número BSON (double, int32, int64) a float64.
func ToFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int32:
		return float64(x)
	case int64:
		return float64(x)
	case int:
		return float64(x)
	}
	return 0
}

// ToInt convierte un número BSON a int.
func ToInt(v any) int { return int(ToFloat(v)) }

// ToString devuelve el texto o "".
func ToString(v any) string {
	s, _ := v.(string)
	return s
}
