# Etapa 1: compilar los binarios con la imagen oficial de Go (solo librería estándar).
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
# Los tests corren dentro del build: si alguno falla, la imagen no se construye.
RUN go vet ./... && go test ./...
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/pc3 ./cmd/pc3 \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/train ./cmd/train

# Etapa 2: imagen final pequeña, solo con los binarios.
FROM alpine:3.20
RUN adduser -D -u 10001 app
WORKDIR /app
COPY --from=build /out/pc3 /out/train /usr/local/bin/
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p data results && chown -R app /app
USER app
# Los datos NO van dentro de la imagen: se montan como volumen en /app/data.
VOLUME ["/app/data", "/app/results"]
ENTRYPOINT ["docker-entrypoint.sh"]
# Por defecto: limpiar los CSV y entrenar el modelo concurrente.
CMD ["all"]
