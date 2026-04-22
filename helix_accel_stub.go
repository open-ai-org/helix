//go:build !(darwin && cgo)

package helix

func helixAccelAvailable() bool { return false }
func helixClipGradsAccel(h *HelixOptimizer, maxNorm float32) float32 { return 0 }
func helixDNAStepAccel(pair helixPair, r Rung, lr, bc1, bc2 float32, h *HelixOptimizer) {}
func helixAdamStepAccel(p HelixParam, lr, bc1, bc2 float32, h *HelixOptimizer) {}
func helixCheckpointCopy(dst, src []float32) { copy(dst, src) }

var useHelixAccel = false
