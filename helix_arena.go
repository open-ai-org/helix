package helix

// HelixArena provides zero-copy checkpoint/restore via pointer swapping.
// The full GPU-backed implementation (Metal shared memory) is in the
// mongoose-tensor extension package. This stub supports the optimizer's
// checkpoint/restore interface with a Swap() method.
type HelixArena struct {
	swapped bool
}

// Swap alternates the live and checkpoint buffers.
func (a *HelixArena) Swap() {
	a.swapped = !a.swapped
}
