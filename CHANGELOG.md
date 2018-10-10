# Change Log

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- 2017-12-14 Posting semlog sempaphore in functions ImageStreamIO_sempost, ImageStreamIO_sempost_loop (Sevin, Guyon)
- 2018-07-03 Add GPUIPC feature need to be validate with non-regression tests (Sevin)

### Changed

- 2017-12-14 Change return value in functions ImageStreamIO_sempost, ImageStreamIO_sempost_loop to be compiliant with standards (Sevin, Guyon)
- 2018-03-27 initialize shmfd and memsize to 0 when not shared, remove free semptr in ImageStreamIO_createsem which can produce segfault (Sevin)
- 2018-05-03 update CMakeLists.txt with 3.0 conventions
- 2018-07-04 Debug GPUIPC feature need more tests to validate (Sevin)

### Removed
