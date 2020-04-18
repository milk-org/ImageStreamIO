[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/dcfd9f03c69341f9bd71a5878c170881)](https://www.codacy.com/gh/milk-org/ImageStreamIO?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=milk-org/ImageStreamIO&amp;utm_campaign=Badge_Grade)


# Module ImageStreamIO {#page_module_ImageStreamIO}

Image stream definitions and tools.



## Overview

ImageStreamIO implements a shared memory image format, refered to as stream, for low-latency high throughput I/O. 

This is generally used for images, for example dumping the ouptut of a high speed camera onto shared memory, but can also be used for any data requiring fast I/O. The format includes semaphores for low-latency IPC.

## How to read/write to/from streams

The interface protocol can be either:

1. Slow interface by reading and writing FITS files. Command-line scripts are provided to apply FITS file to DM and also to read FITS files from any camera
2. Fast interface by direct access to shared memory. This option is done by loading shared library and calling it from C program. A Python version also exists; which is slightly slower.


### Approximate speed / timing accuracy:

- Option #1 is very slow (few 100 Hz max) and timing stability is probably no better than 50ms
- Option #2 from Python supports ~kHz and timing stability is probably at ms level (I haven't measured it ... just guessing)
- Option #2 from C program supports >10 kHz (- so it is limited by camera speed, not software) and timing stability is ~10us

For options (1) and (2) with C, you can install milk package:
<https://github.com/milk-org/milk>


### Command line interface (slow) details 

Download and install the [milk package](https://github.com/milk-org/milk).

The scripts to interface FITS files with our data streams are in:
<https://github.com/milk-org/milk/tree/master/scripts>

Look for scripts :
- shmim2fits : Converts stream to FITS file
- Fits2shm : Writes FITS file to stream


### C interface (fast) details 

Link ImageStreamIO to your program and call IO functions directly.

See example source code in ImageStremIO module: [ImCreate_test.c](https://github.com/milk-org/ImageStreamIO/blob/master/ImCreate_test.c).
	
