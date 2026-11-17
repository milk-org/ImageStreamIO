# Change Log

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- 2017-12-14 Posting semlog sempaphore in functions ImageStreamIO_sempost, ImageStreamIO_sempost_loop (Sevin, Guyon)
- 2018-07-03 Add GPUIPC feature need to be validate with non-regression tests (Sevin)
- 2019-03-10 python module
- 2023-06-24 Operational and several other tests (Carcich)
- 2026-03-27 Python: zero-copy `IMAGE.view()`; GPU streams are returned as cupy arrays (Deo)
- 2026-05-03 Process-local refcounted CUDA IPC handle registry: the same GPU SHM can be opened several times in one process (Deo)
- 2026-02-04 `Coverage` CMake build type (Deo)

### Changed

- 2017-12-14 Change return value in functions ImageStreamIO_sempost, ImageStreamIO_sempost_loop to be compiliant with standards (Sevin, Guyon)
- 2018-03-27 initialize shmfd and memsize to 0 when not shared, remove free semptr in ImageStreamIO_createsem which can produce segfault (Sevin)
- 2018-05-03 update CMakeLists.txt with 3.0 conventions
- 2018-07-04 Debug GPUIPC feature need more tests to validate (Sevin)
- 2019-03-04 New image structure (Guyon, Sevin)
- 2023-06-24 Change named shmim-external semaphores to unnamed shmim internal (Carcich)
- 2023-06-24 Update image structure (Carcich)
- 2023-06-24 Consolidate size calculations (Carcich)
- 2023-06-24 Change VERSION to "2.00" (Carcich)
- 2026-03-27 Python bindings ported from pybind11 to nanobind; Python package version 2.0.0 (Deo)
- 2026-03-27 Python: `IMAGE.update_atime()` takes a float timestamp; `write()` is a single overload (Deo)
- 2026-10-05 Python: GIL released during semaphore waits (Deo)
- 2026-03-26 `ImageStreamIO_set_printError()` callback takes `const char *` message (Deo)
- 2026-03-08 CMake: siblings must `#include "ImageStreamIO/..."`; Debug/Release build types (`-Ofast` only in Release) (Deo)
- 2026-04-12 Shared images are created via a temp file + `rename()` for atomic creation (Deo)
- 2026-05-03 Refactor `ImageStreamIO_closeIm` / `ImageStreamIO_destroyIm`; `openIm` now sets `used = 1` (Deo)
- 2026-07-21 License changed to MIT; REUSE/SPDX headers (Deo)
- 2026-11-16 Library version 2.01.00 (Deo)

### Removed
