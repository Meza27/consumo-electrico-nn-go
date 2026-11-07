// Package platform agrupa ajustes del sistema operativo para medir tiempos de forma justa.
//go:build !windows

package platform

// SinEcoQoS no hace nada fuera de Windows.
func SinEcoQoS() {}
