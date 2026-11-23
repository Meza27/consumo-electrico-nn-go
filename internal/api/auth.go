package api

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Claims es el contenido del token JWT.
type Claims struct {
	Sub  string `json:"sub"`
	Role string `json:"role"`
	Iat  int64  `json:"iat"`
	Exp  int64  `json:"exp"`
}

var b64 = base64.RawURLEncoding

// NewJWT firma un token HS256: base64(header).base64(payload).base64(firma).
func NewJWT(secret []byte, user, role string, ttl time.Duration) (string, error) {
	h := b64.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	now := time.Now()
	p, err := json.Marshal(Claims{Sub: user, Role: role, Iat: now.Unix(), Exp: now.Add(ttl).Unix()})
	if err != nil {
		return "", err
	}
	msg := h + "." + b64.EncodeToString(p)
	mac := hmac.New(sha256.New, secret)
	mac.Write([]byte(msg))
	return msg + "." + b64.EncodeToString(mac.Sum(nil)), nil
}

// ParseJWT verifica la firma y la expiración.
func ParseJWT(secret []byte, token string) (*Claims, error) {
	partes := strings.Split(token, ".")
	if len(partes) != 3 {
		return nil, errors.New("token mal formado")
	}
	mac := hmac.New(sha256.New, secret)
	mac.Write([]byte(partes[0] + "." + partes[1]))
	firma, err := b64.DecodeString(partes[2])
	if err != nil || !hmac.Equal(firma, mac.Sum(nil)) {
		return nil, errors.New("firma inválida")
	}
	p, err := b64.DecodeString(partes[1])
	if err != nil {
		return nil, err
	}
	var c Claims
	if err := json.Unmarshal(p, &c); err != nil {
		return nil, err
	}
	if time.Now().Unix() > c.Exp {
		return nil, errors.New("token expirado")
	}
	return &c, nil
}

const iteraciones = 20000

// HashPassword guarda la contraseña como sha256 iterado con sal aleatoria:
// "sha256$<iteraciones>$<sal>$<hash>". Nunca se guarda la contraseña en claro.
func HashPassword(pass string) string {
	sal := make([]byte, 16)
	rand.Read(sal)
	return fmt.Sprintf("sha256$%d$%s$%s", iteraciones, hex.EncodeToString(sal), hex.EncodeToString(derivar(pass, sal, iteraciones)))
}

// CheckPassword compara en tiempo constante.
func CheckPassword(pass, guardado string) bool {
	p := strings.Split(guardado, "$")
	if len(p) != 4 || p[0] != "sha256" {
		return false
	}
	n, err1 := strconv.Atoi(p[1])
	sal, err2 := hex.DecodeString(p[2])
	h, err3 := hex.DecodeString(p[3])
	if err1 != nil || err2 != nil || err3 != nil {
		return false
	}
	return subtle.ConstantTimeCompare(derivar(pass, sal, n), h) == 1
}

func derivar(pass string, sal []byte, n int) []byte {
	sum := sha256.Sum256(append(append([]byte{}, sal...), pass...))
	for i := 1; i < n; i++ {
		sum = sha256.Sum256(sum[:])
	}
	return sum[:]
}

type ctxKey struct{}

// claimsDe devuelve el usuario autenticado de la petición.
func claimsDe(r *http.Request) *Claims {
	c, _ := r.Context().Value(ctxKey{}).(*Claims)
	return c
}

// auth exige un JWT válido (cabecera "Authorization: Bearer <token>" o, para
// WebSocket, el parámetro ?token=). Si role != "", exige ese rol.
func (s *Server) auth(role string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		tok := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if tok == "" || tok == r.Header.Get("Authorization") {
			tok = r.URL.Query().Get("token")
		}
		c, err := ParseJWT(s.cfg.JWTSecret, tok)
		if err != nil {
			writeErr(w, http.StatusUnauthorized, "no autorizado: "+err.Error())
			return
		}
		if role != "" && c.Role != role {
			writeErr(w, http.StatusForbidden, "se requiere rol "+role)
			return
		}
		next(w, r.WithContext(context.WithValue(r.Context(), ctxKey{}, c)))
	}
}
