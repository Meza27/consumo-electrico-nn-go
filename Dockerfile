# Etapa 1: compilar los binarios con la imagen oficial de Go (solo librería estándar).
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
# Los tests corren dentro del build: si alguno falla, la imagen no se construye.
RUN go vet ./... && go test ./...
RUN for c in pc3 train node api loadtest clusterbench; do \
      CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/$c ./cmd/$c || exit 1; \
    done

# Etapa 2: imagen final pequeña, solo con los binarios.
FROM alpine:3.20
RUN adduser -D -u 10001 app
WORKDIR /app
COPY --from=build /out/ /usr/local/bin/
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p data results && chown -R app /app
USER app
# Los datos NO van dentro de la imagen: se montan como volumen en /app/data.
VOLUME ["/app/data", "/app/results"]
ENTRYPOINT ["docker-entrypoint.sh"]
# Puertos: 8080 (API) y 9000 (nodo ML).
EXPOSE 8080 9000
# Por defecto: limpiar los CSV y entrenar el modelo concurrente (PC3).
# En Docker Compose cada servicio elige su modo: api, node o prep.
CMD ["all"]
