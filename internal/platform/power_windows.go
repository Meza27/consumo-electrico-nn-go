//go:build windows

package platform

import (
	"syscall"
	"unsafe"
)

// SinEcoQoS le pide a Windows 11 que NO aplique "power throttling" (EcoQoS) a
// este proceso. Sin esto, cuando la consola no está en primer plano, Windows
// puede mover los hilos a los núcleos de eficiencia (E-cores) a baja
// frecuencia y el speedup medido se aplana. Afecta solo a este proceso; no
// cambia ninguna configuración del sistema.
func SinEcoQoS() {
	type powerThrottling struct{ Version, ControlMask, StateMask uint32 }
	const processPowerThrottling = 4 // PROCESS_INFORMATION_CLASS
	estado := powerThrottling{Version: 1, ControlMask: 1 /* EXECUTION_SPEED */, StateMask: 0 /* desactivado */}
	h, err := syscall.GetCurrentProcess()
	if err != nil {
		return
	}
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("SetProcessInformation")
	if proc.Find() != nil {
		return
	}
	proc.Call(uintptr(h), processPowerThrottling, uintptr(unsafe.Pointer(&estado)), unsafe.Sizeof(estado))
}
