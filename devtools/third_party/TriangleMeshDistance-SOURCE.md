# TriangleMeshDistance

`TriangleMeshDistance.h` is an MIT-licensed single-header library by
José Antonio Fernández Fernández (copyright 2021), obtained from SdfLib commit
`8db373ef71d65be24badf6ae10750a932bbc223b`, at
`libs/InteractiveComputerGraphics/InteractiveComputerGraphics/TriangleMeshDistance.h`.
The original header's SHA-256 is
`c9a4fabb0e54a3b2d4ef4eb9deb941387d9916f8bd138d0ed5ff7c329175c58c`.

The local header's SHA-256 is
`e1ff3c81ffb856a8d53a2576704cc83b55bfd63f8bcc14ca8ae06cb8838886f3`.
It adds `<stdexcept>` and three guards in `signed_distance`: reject a missing or
nonfinite nearest-triangle result before indexing; throw on an invalid nearest
entity rather than using an uninitialized pseudonormal; reject nonfinite
pseudonormals. The local file also ends with a newline and strips six inherited
trailing-whitespace lines. The pre-whitespace-normalization local SHA-256 was
`1c631199a4462c3443aac0ef9647b007c38dfff874e518d1dd6f40a0c50e752a`;
the existing final-library node recheck records that source snapshot.
No successful finite
query arithmetic or BVH search was changed. The original MIT notice is intact.

GCC 13.3 identified the unchecked switch default with `-Wmaybe-uninitialized`.
All normal point/triangle feature branches assign one of the seven enumerators,
but this does not make the old error path safe: an extreme finite query can
overflow BVH arithmetic and leave `triangle_id=-1`, which the old implementation
indexed before the wrapper could check it. The guards turn that failure into a
C ABI error; warnings are not suppressed. The finite-overflow query regression
and the analytic checks run through the resulting shared library.

Source repository: <https://github.com/UPC-ViRVIG/SdfLib>. The vendored header,
not a checkout of that repository, is the build input. No SdfLib code, octree,
Hermite interpolation, or process-wide sample cache is required at runtime.

`../_mesh_distance.cpp` builds the header's double-precision BVH once. It uses
the selected triangle and pseudonormal sign, then recomputes closest distance
from plane projection and three clamped segments in double precision. This
residual calculation is the tested mechanism extracted from the experiment's
v4 primitive adaptation; it avoids cancellation in the header's expanded
quadratic distance. Strict face-interior signs use the oriented plane directly.
The header's BVH face selection remains the source of nearest-triangle identity;
the wrapper is not a Boolean mesh operation or an exact-arithmetic certificate.

Validation and the public input contract are described in
[`../mesh_distance.md`](../mesh_distance.md) and checked by
`tests/verify_mesh_distance.py`.
