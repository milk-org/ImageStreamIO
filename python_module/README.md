# ImageStreamIOWrap

- [ImageStreamIOWrap](#imagestreamiowrap)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Creation of a new Image](#creation-of-a-new-image)
  - [Open of a existing Image](#open-of-a-existing-image)
  - [Use an Image](#use-an-image)

## Installation

In the root directory ie. ```ImageStreamIO```

```bash
pip install .
```

Note: You can install it in a custom path with the option ```-t $HOME/local/python```

Note2: If you update it, use the option ```-U```

## Usage

```python
import ImageStreamIOWrap as ISIO
img = ISIO.Image()
```

NOTE: ```img``` is empty at this time, you need to create or open a Image.

## Creation of a new Image

```python
In [2]: img.create?

Docstring:
create(*args, **kwargs)
Overloaded function.

1. create(self: ImageStreamIOWrap.Image, name: str, dims: numpy.ndarray[uint32], datatype: int=Type.FLOAT, shared: int=1, NBkw: int=1) -> int


            Create shared memory image stream
            Parameters:
                name     [in]:  the name of the shared memory file will be SHAREDMEMDIR/<name>_im.shm
                dims     [in]:  np.array of the image.
                datatype [in]:  data type code, pyImageStreamIO.Datatype
                shared   [in]:  if true then a shared memory buffer is allocated.  If false, only local storage is used.
                NBkw     [in]:  the number of keywords to allocate.
            Return:
                ret      [out]: error code


2. create(self: ImageStreamIOWrap.Image, name: str, dims: numpy.ndarray[uint32], datatype: int=Type.FLOAT, location: int=-1, shared: int=1, NBsem: int=10, NBkw: int=1, imagetype: int=2) -> int


            Create shared memory image stream
            Parameters:
                name      [in]:  the name of the shared memory file will be SHAREDMEMDIR/<name>_im.shm
                dims      [in]:  np.array of the image.
                datatype  [in]:  data type code, pyImageStreamIO.Datatype
                shared    [in]:  if true then a shared memory buffer is allocated.  If false, only local storage is used.
                NBsem     [in]:  the number of semaphores to allocate.
                NBkw      [in]:  the number of keywords to allocate.
                imagetype [in]:  type of the stream.
            Return:
                ret       [out]: error code

Type:      method
```

The first one is the legacy API where it uses CPU buffer.

The second one can create GPU IPC SHM (need CUDA).

```python
In [3]: img.create("cacaoTest", [128,128], ISIO.ImageStreamIODataType.FLOAT, 1, 8)
```

## Open of a existing Image

```python
In [15]: img.open?

Docstring:
open(self: ImageStreamIOWrap.Image, name: str) -> int


Open / connect to existing shared memory image stream
Parameters:
    name   [in]:  the name of the shared memory file to connect
Return:
    ret    [out]: error code
Type:      method


In [16]: img.open("cacaoTest")
```

## Use an Image

```python
In [5]: img.md.size
Out[5]: [128, 128]

In [6]: img.semReadPID
Out[6]: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

In [7]: img.md.lastaccesstime
Out[7]: datetime.datetime(2019, 3, 10, 13, 16, 59, 516486)

In [8]: import numpy as np

In [9]: np.array(img)
Out[9]:
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

In [12]: img.write(np.ones((128,128), dtype=np.float32))

In [14]: np.array(img)
Out[14]:
array([[1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       ...,
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.]], dtype=float32)

In [15]: img.md.lastaccesstime
Out[15]: datetime.datetime(2019, 3, 10, 13, 18, 20, 132500)
```
