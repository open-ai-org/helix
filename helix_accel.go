//go:build darwin && cgo

package helix

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <string.h>

// Gradient norm via cblas_snrm2 (AMX-accelerated on Apple Silicon).
static float helix_grad_norm(float** grads, int* sizes, int nBufs) {
    float normSq = 0.0f;
    for (int i = 0; i < nBufs; i++) {
        float n = cblas_snrm2(sizes[i], grads[i], 1);
        normSq += n * n;
    }
    return sqrtf(normSq);
}

// Scale all gradient buffers (AMX cblas_sscal).
static void helix_grad_scale(float** grads, int* sizes, int nBufs, float scale) {
    for (int i = 0; i < nBufs; i++) {
        cblas_sscal(sizes[i], scale, grads[i], 1);
    }
}

// Clip gradients to maxNorm. Returns pre-clip norm.
static float helix_clip_grads(float** grads, int* sizes, int nBufs, float maxNorm) {
    float norm = helix_grad_norm(grads, sizes, nBufs);
    if (norm > maxNorm) {
        helix_grad_scale(grads, sizes, nBufs, maxNorm / norm);
    }
    return norm;
}

// DNA rung Adam step — paired parameters, 6-point coupling.
static void helix_dna_step(
    float* d1, float* g1, float* m1, float* v1, int n1,
    float* d2, float* g2, float* m2, float* v2, int n2,
    float backbone1, float glyco1, float hbond1,
    float hbond2, float glyco2, float backbone2,
    float bondStrength,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd) {

    int coupled = n1 < n2 ? n1 : n2;
    float ob1 = 1.0f - beta1;
    float ob2 = 1.0f - beta2;
    float wd1 = wd * backbone1;
    float wd2 = wd * backbone2;

    for (int i = 0; i < coupled; i++) {
        float signal1 = g1[i] * glyco1;
        float crossMom = g2[i] * hbond1 * bondStrength;
        float crossVel = g1[i] * hbond2 * bondStrength;
        float signal2 = g2[i] * glyco2;

        float effGrad1 = signal1 + crossMom;
        float mi1 = beta1 * m1[i] + ob1 * effGrad1;
        float vi1 = beta2 * v1[i] + ob2 * effGrad1 * effGrad1;
        m1[i] = mi1; v1[i] = vi1;
        d1[i] -= lr * (mi1 / bc1 / (sqrtf(vi1 / bc2) + eps) + wd1 * d1[i]);

        float effGrad2 = signal2 + crossVel;
        float mi2 = beta1 * m2[i] + ob1 * effGrad2;
        float vi2 = beta2 * v2[i] + ob2 * effGrad2 * effGrad2;
        m2[i] = mi2; v2[i] = vi2;
        d2[i] -= lr * (mi2 / bc1 / (sqrtf(vi2 / bc2) + eps) + wd2 * d2[i]);
    }
    for (int i = coupled; i < n1; i++) {
        float g = g1[i];
        float mi = beta1 * m1[i] + ob1 * g;
        float vi = beta2 * v1[i] + ob2 * g * g;
        m1[i] = mi; v1[i] = vi;
        d1[i] -= lr * (mi / bc1 / (sqrtf(vi / bc2) + eps) + wd * d1[i]);
    }
    for (int i = coupled; i < n2; i++) {
        float g = g2[i];
        float mi = beta1 * m2[i] + ob1 * g;
        float vi = beta2 * v2[i] + ob2 * g * g;
        m2[i] = mi; v2[i] = vi;
        d2[i] -= lr * (mi / bc1 / (sqrtf(vi / bc2) + eps) + wd * d2[i]);
    }
}

// Standard Adam step (unpaired).
static void helix_adam_step_c(
    float* d, float* g, float* m, float* v, int n,
    float lr, float beta1, float beta2, float bc1, float bc2,
    float eps, float wd) {
    float ob1 = 1.0f - beta1;
    float ob2 = 1.0f - beta2;
    for (int i = 0; i < n; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + ob1 * gi;
        float vi = beta2 * v[i] + ob2 * gi * gi;
        m[i] = mi; v[i] = vi;
        d[i] -= lr * (mi / bc1 / (sqrtf(vi / bc2) + eps) + wd * d[i]);
    }
}
*/
import "C"
import (
	"math"
	"unsafe"
)

// helixAccelAvailable returns true on darwin with Accelerate.
func helixAccelAvailable() bool { return true }

// helixClipGradsAccel clips gradients using AMX-accelerated cblas_snrm2/sscal.
// Uses C-allocated arrays to avoid cgo pointer rule violations.
func helixClipGradsAccel(h *HelixOptimizer, maxNorm float32) float32 {
	nBufs := 0
	for range h.pairs { nBufs += 2 }
	nBufs += len(h.singles)

	// Allocate C arrays for pointers and sizes
	cPtrs := C.malloc(C.size_t(nBufs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	cSizes := C.malloc(C.size_t(nBufs) * C.size_t(unsafe.Sizeof(C.int(0))))
	defer C.free(cPtrs)
	defer C.free(cSizes)

	ptrs := (*[1 << 20]*C.float)(cPtrs)
	sizes := (*[1 << 20]C.int)(cSizes)

	idx := 0
	for _, pair := range h.pairs {
		g1 := pair.strand1.GradHelix()
		ptrs[idx] = (*C.float)(unsafe.Pointer(&g1[0]))
		sizes[idx] = C.int(len(g1))
		idx++
		g2 := pair.strand2.GradHelix()
		ptrs[idx] = (*C.float)(unsafe.Pointer(&g2[0]))
		sizes[idx] = C.int(len(g2))
		idx++
	}
	for _, p := range h.singles {
		g := p.GradHelix()
		ptrs[idx] = (*C.float)(unsafe.Pointer(&g[0]))
		sizes[idx] = C.int(len(g))
		idx++
	}

	norm := float32(C.helix_clip_grads(
		(**C.float)(cPtrs),
		(*C.int)(cSizes),
		C.int(nBufs),
		C.float(maxNorm),
	))
	return norm
}

// helixDNAStepAccel runs the 6-point rung Adam update in C.
func helixDNAStepAccel(pair helixPair, r Rung, lr, bc1, bc2 float32, h *HelixOptimizer) {
	bondStrength := float32(pair.strength) / 5.0

	C.helix_dna_step(
		(*C.float)(unsafe.Pointer(&pair.strand1.DataHelix()[0])),
		(*C.float)(unsafe.Pointer(&pair.strand1.GradHelix()[0])),
		(*C.float)(unsafe.Pointer(&pair.strand1.MomHelix()[0])),
		(*C.float)(unsafe.Pointer(&pair.strand1.VelHelix()[0])),
		C.int(pair.strand1.SizeHelix()),

		(*C.float)(unsafe.Pointer(&pair.strand2.DataHelix()[0])),
		(*C.float)(unsafe.Pointer(&pair.strand2.GradHelix()[0])),
		(*C.float)(unsafe.Pointer(&pair.strand2.MomHelix()[0])),
		(*C.float)(unsafe.Pointer(&pair.strand2.VelHelix()[0])),
		C.int(pair.strand2.SizeHelix()),

		C.float(r.Backbone1), C.float(r.Glyco1), C.float(r.Hbond1),
		C.float(r.Hbond2), C.float(r.Glyco2), C.float(r.Backbone2),
		C.float(bondStrength),

		C.float(lr), C.float(h.beta1), C.float(h.beta2),
		C.float(bc1), C.float(bc2), C.float(h.eps), C.float(h.weightDecay),
	)
}

// helixAdamStepAccel runs standard Adam in C for unpaired params.
func helixAdamStepAccel(p HelixParam, lr, bc1, bc2 float32, h *HelixOptimizer) {
	C.helix_adam_step_c(
		(*C.float)(unsafe.Pointer(&p.DataHelix()[0])),
		(*C.float)(unsafe.Pointer(&p.GradHelix()[0])),
		(*C.float)(unsafe.Pointer(&p.MomHelix()[0])),
		(*C.float)(unsafe.Pointer(&p.VelHelix()[0])),
		C.int(p.SizeHelix()),
		C.float(lr), C.float(h.beta1), C.float(h.beta2),
		C.float(bc1), C.float(bc2), C.float(h.eps), C.float(h.weightDecay),
	)
}

// helixCheckpointCopy uses memcpy for fast checkpoint save/restore.
func helixCheckpointCopy(dst, src []float32) {
	if len(dst) != len(src) || len(dst) == 0 {
		return
	}
	C.memcpy(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), C.size_t(len(dst)*4))
}

// useAccel is set at init time on darwin.
var useHelixAccel = true

func init() {
	_ = math.Pi // keep import
}
