package store

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"time"
)

// Redis es un cliente mínimo del protocolo RESP (Redis 2+), con un pool de
// conexiones en un channel con buffer, igual que el cliente de MongoDB.
type Redis struct {
	addr    string
	pool    chan *redisConn
	Timeout time.Duration
}

type redisConn struct {
	c net.Conn
	r *bufio.Reader
	w *bufio.Writer
}

// ErrNil indica que la clave no existe.
var ErrNil = errors.New("redis: nil")

// DialRedis abre `size` conexiones contra addr (host:6379) y prueba con PING.
func DialRedis(addr string, size int) (*Redis, error) {
	if size < 1 {
		size = 4
	}
	r := &Redis{addr: addr, pool: make(chan *redisConn, size), Timeout: 3 * time.Second}
	for i := 0; i < size; i++ {
		c, err := r.nueva()
		if err != nil {
			r.Close()
			return nil, err
		}
		r.pool <- c
	}
	if _, err := r.Do("PING"); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

func (r *Redis) nueva() (*redisConn, error) {
	c, err := net.DialTimeout("tcp", r.addr, 3*time.Second)
	if err != nil {
		return nil, err
	}
	return &redisConn{c: c, r: bufio.NewReader(c), w: bufio.NewWriter(c)}, nil
}

// Close cierra las conexiones libres.
func (r *Redis) Close() {
	for {
		select {
		case c := <-r.pool:
			c.c.Close()
		default:
			return
		}
	}
}

// Do envía un comando y devuelve la respuesta: string, int64, []any, nil o error.
//
// Formato RESP de un comando: *<n>\r\n y por cada argumento $<len>\r\n<arg>\r\n
func (r *Redis) Do(args ...string) (any, error) {
	c := <-r.pool
	c.c.SetDeadline(time.Now().Add(r.Timeout))
	fmt.Fprintf(c.w, "*%d\r\n", len(args))
	for _, a := range args {
		fmt.Fprintf(c.w, "$%d\r\n%s\r\n", len(a), a)
	}
	err := c.w.Flush()
	var v any
	if err == nil {
		v, err = leerRESP(c.r)
	}
	var re respError
	if err != nil && !errors.As(err, &re) {
		c.c.Close() // error de red: se reemplaza la conexión
		if nc, e := r.nueva(); e == nil {
			c = nc
		}
	}
	r.pool <- c
	return v, err
}

type respError string

func (e respError) Error() string { return "redis: " + string(e) }

// leerRESP interpreta una respuesta: +simple, -error, :entero, $bulk, *arreglo.
func leerRESP(r *bufio.Reader) (any, error) {
	linea, err := r.ReadString('\n')
	if err != nil {
		return nil, err
	}
	if len(linea) < 3 {
		return nil, errors.New("redis: respuesta vacía")
	}
	linea = linea[:len(linea)-2]
	switch linea[0] {
	case '+':
		return linea[1:], nil
	case '-':
		return nil, respError(linea[1:])
	case ':':
		return strconv.ParseInt(linea[1:], 10, 64)
	case '$':
		n, err := strconv.Atoi(linea[1:])
		if err != nil {
			return nil, err
		}
		if n < 0 {
			return nil, nil
		}
		buf := make([]byte, n+2)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		return string(buf[:n]), nil
	case '*':
		n, err := strconv.Atoi(linea[1:])
		if err != nil {
			return nil, err
		}
		if n < 0 {
			return nil, nil
		}
		arr := make([]any, n)
		for i := range arr {
			if arr[i], err = leerRESP(r); err != nil {
				return nil, err
			}
		}
		return arr, nil
	}
	return nil, fmt.Errorf("redis: tipo de respuesta desconocido %q", linea[0])
}

// Get devuelve el valor de la clave o ErrNil.
func (r *Redis) Get(k string) (string, error) {
	v, err := r.Do("GET", k)
	if err != nil {
		return "", err
	}
	if v == nil {
		return "", ErrNil
	}
	return v.(string), nil
}

// Set guarda un valor con expiración (ttl 0 = sin expiración).
func (r *Redis) Set(k, v string, ttl time.Duration) error {
	if ttl > 0 {
		_, err := r.Do("SET", k, v, "EX", strconv.Itoa(int(ttl.Seconds())))
		return err
	}
	_, err := r.Do("SET", k, v)
	return err
}

// Del borra claves.
func (r *Redis) Del(keys ...string) error {
	_, err := r.Do(append([]string{"DEL"}, keys...)...)
	return err
}

// Incr incrementa un contador.
func (r *Redis) Incr(k string) (int64, error) {
	v, err := r.Do("INCR", k)
	if err != nil {
		return 0, err
	}
	return v.(int64), nil
}

// HSet guarda campos en un hash: HSet(k, campo1, valor1, campo2, valor2...).
func (r *Redis) HSet(k string, campoValor ...string) error {
	_, err := r.Do(append([]string{"HSET", k}, campoValor...)...)
	return err
}

// HGetAll devuelve todos los campos de un hash.
func (r *Redis) HGetAll(k string) (map[string]string, error) {
	v, err := r.Do("HGETALL", k)
	if err != nil {
		return nil, err
	}
	arr, _ := v.([]any)
	out := map[string]string{}
	for i := 0; i+1 < len(arr); i += 2 {
		out[fmt.Sprint(arr[i])] = fmt.Sprint(arr[i+1])
	}
	return out, nil
}

// LPush agrega al inicio de una lista y la recorta a `max` elementos.
func (r *Redis) LPush(k, v string, max int) error {
	if _, err := r.Do("LPUSH", k, v); err != nil {
		return err
	}
	if max > 0 {
		_, err := r.Do("LTRIM", k, "0", strconv.Itoa(max-1))
		return err
	}
	return nil
}

// LRange devuelve un rango de una lista.
func (r *Redis) LRange(k string, ini, fin int) ([]string, error) {
	v, err := r.Do("LRANGE", k, strconv.Itoa(ini), strconv.Itoa(fin))
	if err != nil {
		return nil, err
	}
	arr, _ := v.([]any)
	out := make([]string, 0, len(arr))
	for _, x := range arr {
		out = append(out, fmt.Sprint(x))
	}
	return out, nil
}
