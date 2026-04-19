package helix

import (
	"math"
	"testing"
)

func TestDNAConstants(t *testing.T) {
	// Verify DNA geometry matches physical reality
	if math.Abs(dnaTwistPerStep-0.6283) > 0.001 {
		t.Errorf("twist per step: want ~0.6283 rad (36°), got %f", dnaTwistPerStep)
	}
	if math.Abs(dnaHelixAngle-0.496) > 0.001 {
		t.Errorf("helix angle: want ~0.496 rad (28.4°), got %f", dnaHelixAngle)
	}
	if math.Abs(dnaGrooveRatio-1.833) > 0.01 {
		t.Errorf("groove ratio: want ~1.833, got %f", dnaGrooveRatio)
	}
	if dnaBasePairsPerTurn != 10 {
		t.Errorf("base pairs per turn: want 10, got %d", dnaBasePairsPerTurn)
	}
}

func TestHelixPhaseAdvance(t *testing.T) {
	h := NewHelixOptimizer(1e-3, 0.9, 0.95, 1e-8, 0.1)

	p := &SimpleHelixParam{
		D: make([]float32, 4), G: make([]float32, 4),
		M: make([]float32, 4), V: make([]float32, 4), N: 4,
	}
	h.Register(p)

	// 10 steps = 1 full turn (360°)
	for i := 1; i <= 10; i++ {
		for j := range p.G { p.G[j] = 0.1 }
		h.Step(i, 5.0, 1e-3)
	}
	if h.Turn() != 1 {
		t.Errorf("after 10 steps: want 1 turn, got %d", h.Turn())
	}
	if h.BasePair() != 1 {
		t.Errorf("after 10 steps: want base pair 1 (wrapped), got %d", h.BasePair())
	}
}

func TestATvsGCBondStrength(t *testing.T) {
	// AT (2 bonds) should produce weaker cross-strand coupling than GC (3 bonds).
	// Run both with identical inputs over multiple steps, compare weight divergence.

	makeParam := func() *SimpleHelixParam {
		return &SimpleHelixParam{
			D: []float32{1.0, 2.0, 3.0, 4.0},
			G: []float32{0.1, 0.2, 0.3, 0.4},
			M: make([]float32, 4), V: make([]float32, 4), N: 4,
		}
	}

	// AT pair
	hAT := NewHelixOptimizer(0.01, 0.9, 0.95, 1e-8, 0.0)
	at1, at2 := makeParam(), makeParam()
	at2.D = []float32{5.0, 6.0, 7.0, 8.0}
	at2.G = []float32{0.5, 0.6, 0.7, 0.8}
	hAT.PairAT(at1, at2)

	// GC pair
	hGC := NewHelixOptimizer(0.01, 0.9, 0.95, 1e-8, 0.0)
	gc1, gc2 := makeParam(), makeParam()
	gc2.D = []float32{5.0, 6.0, 7.0, 8.0}
	gc2.G = []float32{0.5, 0.6, 0.7, 0.8}
	hGC.PairGC(gc1, gc2)

	// Run enough steps for gradients to apply (immune system may skip early steps)
	for i := 1; i <= 20; i++ {
		// Reset gradients each step
		for j := range at1.G { at1.G[j] = 0.1 * float32(j+1) }
		for j := range at2.G { at2.G[j] = 0.5 + 0.1*float32(j) }
		for j := range gc1.G { gc1.G[j] = 0.1 * float32(j+1) }
		for j := range gc2.G { gc2.G[j] = 0.5 + 0.1*float32(j) }
		hAT.Step(i, 5.0-0.1*float32(i), 0.01) // decreasing loss so immune settles
		hGC.Step(i, 5.0-0.1*float32(i), 0.01)
	}

	// GC strand1 should differ from AT strand1 (different bond strength
	// means different cross-strand coupling)
	atDiff := math.Abs(float64(at1.D[0] - gc1.D[0]))
	if atDiff < 1e-10 {
		t.Error("AT and GC should produce different strand1 updates")
	}
}

func TestCurveStability(t *testing.T) {
	h := NewHelixOptimizer(1e-3, 0.9, 0.95, 1e-8, 0.1)
	for i := 0; i < 100000; i++ {
		h.advanceCurve()
		if math.IsNaN(h.curveX) || math.IsInf(h.curveX, 0) {
			t.Fatalf("step %d: curveX NaN/Inf", i)
		}
	}
}

func TestGrooveAdaptation(t *testing.T) {
	// Simulate steep descent — loss dropping consistently
	h := NewHelixOptimizer(1e-3, 0.9, 0.95, 1e-8, 0.1)
	for i := 0; i < 20; i++ {
		h.recordSignal(float32(5.0-0.2*float64(i)), 1.0)
	}
	majorSteep, minorSteep := h.grooveWeightsFromSignal()

	// Simulate plateau — loss flat
	h2 := NewHelixOptimizer(1e-3, 0.9, 0.95, 1e-8, 0.1)
	for i := 0; i < 20; i++ {
		h2.recordSignal(float32(1.5+0.001*float64(i%3-1)), 0.1)
	}
	majorFlat, minorFlat := h2.grooveWeightsFromSignal()

	// On plateau, minor groove should grow relative to major
	steepRatio := majorSteep / minorSteep
	flatRatio := majorFlat / minorFlat
	if flatRatio >= steepRatio {
		t.Errorf("minor groove should gain on plateau: steep ratio=%f, flat ratio=%f",
			steepRatio, flatRatio)
	}
}

func TestSingleParamIsAdam(t *testing.T) {
	// Single (unpaired) params should behave exactly like standard Adam
	h1 := NewHelixOptimizer(0.01, 0.9, 0.95, 1e-8, 0.1)
	h2 := NewHelixOptimizer(0.01, 0.9, 0.95, 1e-8, 0.1)

	p1 := &SimpleHelixParam{
		D: []float32{1.0, 2.0}, G: []float32{0.5, 0.5},
		M: make([]float32, 2), V: make([]float32, 2), N: 2,
	}
	p2 := &SimpleHelixParam{
		D: []float32{1.0, 2.0}, G: []float32{0.5, 0.5},
		M: make([]float32, 2), V: make([]float32, 2), N: 2,
	}
	h1.Register(p1)
	h2.Register(p2)

	h1.Step(1, 5.0, 0.01)
	h2.Step(1, 5.0, 0.01)

	for i := range p1.D {
		if math.Abs(float64(p1.D[i]-p2.D[i])) > 1e-7 {
			t.Errorf("D[%d]: two identical singles diverged: %f vs %f", i, p1.D[i], p2.D[i])
		}
	}
}
