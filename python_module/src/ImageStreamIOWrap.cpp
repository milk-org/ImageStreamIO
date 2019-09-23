#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ImageStreamIO.h"
#include "ImageStruct.h"

namespace py = pybind11;

struct ImageStreamIOType {
  enum Type : uint64_t {
    CIRCULAR_BUFFER_TYPE = CIRCULAR_BUFFER,
    MATH_DATA_TYPE = MATH_DATA,
    IMG_RECV_TYPE = IMG_RECV,
    IMG_SENT_TYPE = IMG_SENT,
    ZAXIS_UNDEF_TYPE = ZAXIS_UNDEF,
    ZAXIS_SPACIAL_TYPE = ZAXIS_SPACIAL,
    ZAXIS_TEMPORAL_TYPE = ZAXIS_TEMPORAL,
    ZAXIS_WAVELENGTH_TYPE = ZAXIS_WAVELENGTH,
    ZAXIS_MAPPING_TYPE = ZAXIS_MAPPING
  };

  Type type;

  ImageStreamIOType() : type(MATH_DATA_TYPE){};

  ImageStreamIOType(uint64_t type)
      : type(static_cast<ImageStreamIOType::Type>(type)){};

  enum Type get_type() {
    return static_cast<ImageStreamIOType::Type>(type & 0xF);
  }
  enum Type get_axis() {
    return static_cast<ImageStreamIOType::Type>(type & 0xF0000);
  }
};

struct ImageStreamIODataType {
  enum DataType : uint8_t {
    UINT8 = _DATATYPE_UINT8,
    INT8 = _DATATYPE_INT8,
    UINT16 = _DATATYPE_UINT16,
    INT16 = _DATATYPE_INT16,
    UINT32 = _DATATYPE_UINT32,
    INT32 = _DATATYPE_INT32,
    UINT64 = _DATATYPE_UINT64,
    INT64 = _DATATYPE_INT64,
    FLOAT = _DATATYPE_FLOAT,
    DOUBLE = _DATATYPE_DOUBLE,
    COMPLEX_FLOAT = _DATATYPE_COMPLEX_FLOAT,
    COMPLEX_DOUBLE = _DATATYPE_COMPLEX_DOUBLE,
    HALF = _DATATYPE_HALF
  };
  static const std::vector<uint8_t> Size;

  DataType datatype;
  uint8_t asize;

  ImageStreamIODataType() : datatype(FLOAT), asize(Size[FLOAT]){};

  ImageStreamIODataType(uint8_t datatype)
      : datatype(static_cast<ImageStreamIODataType::DataType>(datatype)),
        asize(Size[datatype]){};
};

const std::vector<uint8_t> ImageStreamIODataType::Size(
    {0, SIZEOF_DATATYPE_UINT8, SIZEOF_DATATYPE_INT8, SIZEOF_DATATYPE_UINT16,
     SIZEOF_DATATYPE_INT16, SIZEOF_DATATYPE_UINT32, SIZEOF_DATATYPE_INT32,
     SIZEOF_DATATYPE_UINT64, SIZEOF_DATATYPE_INT64, SIZEOF_DATATYPE_FLOAT,
     SIZEOF_DATATYPE_DOUBLE, SIZEOF_DATATYPE_COMPLEX_FLOAT,
     SIZEOF_DATATYPE_COMPLEX_DOUBLE, SIZEOF_DATATYPE_HALF});

std::string ImageStreamIODataTypeToPyFormat(ImageStreamIODataType dt) {
  switch (dt.datatype) {
    case ImageStreamIODataType::DataType::UINT8:
      return py::format_descriptor<uint8_t>::format();
    case ImageStreamIODataType::DataType::INT8:
      return py::format_descriptor<int8_t>::format();
    case ImageStreamIODataType::DataType::UINT16:
      return py::format_descriptor<uint16_t>::format();
    case ImageStreamIODataType::DataType::INT16:
      return py::format_descriptor<int16_t>::format();
    case ImageStreamIODataType::DataType::UINT32:
      return py::format_descriptor<uint32_t>::format();
    case ImageStreamIODataType::DataType::INT32:
      return py::format_descriptor<int32_t>::format();
    case ImageStreamIODataType::DataType::UINT64:
      return py::format_descriptor<uint64_t>::format();
    case ImageStreamIODataType::DataType::INT64:
      return py::format_descriptor<int64_t>::format();
    case ImageStreamIODataType::DataType::FLOAT:
      return py::format_descriptor<float>::format();
    case ImageStreamIODataType::DataType::DOUBLE:
      return py::format_descriptor<double>::format();
    // case ImageStreamIODataType::DataType::COMPLEX_FLOAT: return
    // py::format_descriptor<(std::complex<float>>::format(); case
    // ImageStreamIODataType::DataType::COMPLEX_DOUBLE: return
    // py::format_descriptor<(std::complex<double>>::format();
    default:
      throw std::runtime_error("Not implemented");
  }
}

ImageStreamIODataType PyFormatToImageStreamIODataType(const std::string &pf) {
  if (pf == py::format_descriptor<uint8_t>::format())
    return ImageStreamIODataType::DataType::UINT8;
  if (pf == py::format_descriptor<int8_t>::format())
    return ImageStreamIODataType::DataType::INT8;
  if (pf == py::format_descriptor<uint16_t>::format())
    return ImageStreamIODataType::DataType::UINT16;
  if (pf == py::format_descriptor<int16_t>::format())
    return ImageStreamIODataType::DataType::INT16;
  if (pf == py::format_descriptor<uint32_t>::format())
    return ImageStreamIODataType::DataType::UINT32;
  if (pf == py::format_descriptor<int32_t>::format())
    return ImageStreamIODataType::DataType::INT32;
  if (pf == py::format_descriptor<uint64_t>::format())
    return ImageStreamIODataType::DataType::UINT64;
  if (pf == py::format_descriptor<int64_t>::format())
    return ImageStreamIODataType::DataType::INT64;
  if (pf == py::format_descriptor<float>::format())
    return ImageStreamIODataType::DataType::FLOAT;
  if (pf == py::format_descriptor<double>::format())
    return ImageStreamIODataType::DataType::DOUBLE;
  // case ImageStreamIODataType::DataType::COMPLEX_FLOAT: return
  // py::format_descriptor<(std::complex<float>>::format(); case
  // ImageStreamIODataType::DataType::COMPLEX_DOUBLE: return
  // py::format_descriptor<(std::complex<double>>::format();
  throw std::runtime_error("Not implemented");
}

template <typename T>
py::array_t<T> convert_img(const IMAGE &img) {
  if (ImageStreamIO_typesize(img.md->datatype) != sizeof(T)) {
    throw std::runtime_error("IMAGE is not compatible with output format");
  }

  std::vector<ssize_t> shape(img.md->naxis);
  std::vector<ssize_t> strides(img.md->naxis);
  ssize_t stride = sizeof(T);

  // Row Major representation
  // for (int8_t axis(img.md->naxis-1); axis >= 0; --axis) {
  // Col Major representation
  for (int8_t axis(0); axis < img.md->naxis; ++axis) {
    shape[axis] = img.md->size[axis];
    strides[axis] = stride;
    stride *= shape[axis];
  }

  auto ret_buffer = py::array_t<T>(shape, strides);
  void *current_image = img.array.raw;
  #ifdef HAVE_CUDA
  if (img.md->location == -1) {
  #endif
    memcpy(ret_buffer.mutable_data(), current_image,
           img.md->nelement * sizeof(T));
  #ifdef HAVE_CUDA
  } else {
    cudaMemcpy(ret_buffer.mutable_data(), current_image,
               img.md->nelement * sizeof(T), cudaMemcpyDeviceToHost);
  }
  #endif
  return ret_buffer;
}

PYBIND11_MODULE(ImageStreamIOWrap, m) {
  m.doc() = "CACAO ImageStreamIO python module";

  auto imageDatatype =
      py::class_<ImageStreamIODataType>(m, "ImageStreamIODataType")
          .def(py::init([](uint8_t datatype) {
            return std::unique_ptr<ImageStreamIODataType>(
                new ImageStreamIODataType(datatype));
          }))
          .def_readonly("size", &ImageStreamIODataType::asize)
          .def_readonly("type", &ImageStreamIODataType::datatype);

  py::enum_<ImageStreamIODataType::DataType>(imageDatatype, "Type")
      .value("UINT8", ImageStreamIODataType::DataType::UINT8)
      .value("INT8", ImageStreamIODataType::DataType::INT8)
      .value("UINT16", ImageStreamIODataType::DataType::UINT16)
      .value("INT16", ImageStreamIODataType::DataType::INT16)
      .value("UINT32", ImageStreamIODataType::DataType::UINT32)
      .value("INT32", ImageStreamIODataType::DataType::INT32)
      .value("UINT64", ImageStreamIODataType::DataType::UINT64)
      .value("INT64", ImageStreamIODataType::DataType::INT64)
      .value("HALF", ImageStreamIODataType::DataType::HALF)
      .value("FLOAT", ImageStreamIODataType::DataType::FLOAT)
      .value("DOUBLE", ImageStreamIODataType::DataType::DOUBLE)
      .value("COMPLEX_FLOAT", ImageStreamIODataType::DataType::COMPLEX_FLOAT)
      .value("COMPLEX_DOUBLE", ImageStreamIODataType::DataType::COMPLEX_DOUBLE)
      .export_values();

  // IMAGE_KEYWORD interface
  py::class_<IMAGE_KEYWORD>(m, "Image_kw")
      .def(py::init(
          []() { return std::unique_ptr<IMAGE_KEYWORD>(new IMAGE_KEYWORD()); }))
      .def(py::init([](std::string name, int64_t numl, std::string comment) {
             if (name.size() > 16) throw std::invalid_argument("name too long");
             if (comment.size() > 80)
               throw std::invalid_argument("comment too long");
             auto kw = std::unique_ptr<IMAGE_KEYWORD>(new IMAGE_KEYWORD());
             std::copy(name.begin(), name.end(), kw->name);
             kw->type = 'L';
             kw->value.numl = numl;
             std::copy(comment.begin(), comment.end(), kw->comment);
             return kw;
           }),
           py::arg("name"), py::arg("numl"), py::arg("comment") = "")
      .def(py::init([](std::string name, double numf, std::string comment) {
             if (name.size() > 16) throw std::invalid_argument("name too long");
             if (comment.size() > 80)
               throw std::invalid_argument("comment too long");
             auto kw = std::unique_ptr<IMAGE_KEYWORD>(new IMAGE_KEYWORD());
             std::copy(name.begin(), name.end(), kw->name);
             kw->type = 'D';
             kw->value.numf = numf;
             std::copy(comment.begin(), comment.end(), kw->comment);
             return kw;
           }),
           py::arg("name"), py::arg("numf"), py::arg("comment") = "")
      .def(py::init(
               [](std::string name, std::string valstr, std::string comment) {
                 if (name.size() > 16)
                   throw std::invalid_argument("name too long");
                 if (valstr.size() > 16)
                   throw std::invalid_argument("valstr too long");
                 if (comment.size() > 80)
                   throw std::invalid_argument("comment too long");
                 auto kw = std::unique_ptr<IMAGE_KEYWORD>(new IMAGE_KEYWORD());
                 std::copy(name.begin(), name.end(), kw->name);
                 kw->type = 'S';
                 std::copy(valstr.begin(), valstr.end(), kw->value.valstr);
                 std::copy(comment.begin(), comment.end(), kw->comment);
                 return kw;
               }),
           py::arg("name"), py::arg("valstr"), py::arg("comment") = "")
      .def_readonly("name", &IMAGE_KEYWORD::name)
      .def_readonly("type", &IMAGE_KEYWORD::type)
      .def_property_readonly("value",
                             [](const IMAGE_KEYWORD &kw) -> py::object {
                               switch (kw.type) {
                                 case 'L':
                                   return py::int_(kw.value.numl);
                                 case 'D':
                                   return py::float_(kw.value.numf);
                                 case 'S':
                                   return py::str(kw.value.valstr);
                                 default:
                                   throw std::runtime_error("Unknown format");
                               }
                             })
      .def("__repr__",
           [](const IMAGE_KEYWORD &kw) {
             std::ostringstream tmp_str;
             //  tmp_str << kw.name << ": ";
             switch (kw.type) {
               case 'L':
                 tmp_str << kw.value.numl;
                 break;
               case 'D':
                 tmp_str << kw.value.numf;
                 break;
               case 'S':
                 tmp_str << kw.value.valstr;
                 break;
               default:
                 tmp_str << "Unknown format";
                 break;
             }
             tmp_str << " " << kw.comment;
             return tmp_str.str();
           })  // TODO handle union
      .def_readonly("comment", &IMAGE_KEYWORD::comment);

  // IMAGE_METADATA interface
  py::class_<IMAGE_METADATA>(m, "Image_md")
      // .def(py::init([]() {
      //     return std::unique_ptr<IMAGE_METADATA>(new IMAGE_METADATA());
      // }))
      .def_readonly("version", &IMAGE_METADATA::version)
      .def_readonly("name", &IMAGE_METADATA::name)
      .def_readonly("naxis", &IMAGE_METADATA::naxis)
      .def_property_readonly("size",
                             [](const IMAGE_METADATA &md) {
                               std::vector<uint32_t> dims(md.naxis);
                               const uint32_t *ptr = md.size;
                               for (auto &&dim : dims) {
                                 dim = *ptr;
                                 ++ptr;
                               }
                               return dims;
                             })
      .def_readonly("nelement", &IMAGE_METADATA::nelement)
      .def_property_readonly("datatype",
                             [](const IMAGE_METADATA &md) {
                               return ImageStreamIODataType(md.datatype);
                             })
      .def_readonly("imagetype", &IMAGE_METADATA::imagetype)
      .def_property_readonly(
          "creationtime",
          [](const IMAGE_METADATA &md) {
            auto creation_time =
                std::chrono::seconds{md.creationtime.tv_sec} +
                std::chrono::nanoseconds{md.creationtime.tv_nsec};
            std::chrono::system_clock::time_point tp{creation_time};
            return tp;
          })
      .def_property_readonly(
          "lastaccesstime",
          [](const IMAGE_METADATA &md) {
            auto creation_time =
                std::chrono::seconds{md.lastaccesstime.tv_sec} +
                std::chrono::nanoseconds{md.lastaccesstime.tv_nsec};
            std::chrono::system_clock::time_point tp{creation_time};
            return tp;
          })
      .def_property_readonly(
          "acqtime",
          [](const IMAGE_METADATA &md) {
            auto acqtime = std::chrono::seconds{md.atime.tv_sec} +
                           std::chrono::nanoseconds{md.atime.tv_nsec};
            std::chrono::system_clock::time_point tp{acqtime};
            return tp;
          })
      .def_property_readonly(
          "writetime",
          [](const IMAGE_METADATA &md) {
            auto writetime = std::chrono::seconds{md.writetime.tv_sec} +
                             std::chrono::nanoseconds{md.writetime.tv_nsec};
            std::chrono::system_clock::time_point tp{writetime};
            return tp;
          })
      .def_readonly("shared", &IMAGE_METADATA::shared)
      .def_readonly("location", &IMAGE_METADATA::location)
      .def_readonly("status", &IMAGE_METADATA::status)
      .def_readonly("logflag", &IMAGE_METADATA::logflag)
      .def_readonly("sem", &IMAGE_METADATA::sem)
      .def_readonly("cnt0", &IMAGE_METADATA::cnt0)
      .def_readonly("cnt1", &IMAGE_METADATA::cnt1)
      .def_readonly("cnt2", &IMAGE_METADATA::cnt2)
      .def_readonly("write", &IMAGE_METADATA::write)
      .def_readonly("flag", &IMAGE_METADATA::flag)
      .def_readonly("NBkw", &IMAGE_METADATA::NBkw);

  // IMAGE interface
  py::class_<IMAGE>(m, "Image", py::buffer_protocol())
      .def(py::init([]() { return std::unique_ptr<IMAGE>(new IMAGE()); }))
      .def_readonly("used", &IMAGE::used)
      .def_readonly("memsize", &IMAGE::memsize)
      .def_readonly("md", &IMAGE::md)
      .def_property_readonly("semReadPID",
                             [](const IMAGE &img) {
                               std::vector<pid_t> semReadPID(img.md->sem);
                               for (int i = 0; i < img.md->sem; ++i) {
                                 semReadPID[i] = img.semReadPID[i];
                               }
                               return semReadPID;
                             })
      .def_property_readonly(
          "acqtimearray",
          [](const IMAGE &img) {
            if (img.atimearray == NULL)
              throw std::runtime_error("acqtimearray not initialized");
            std::vector<std::chrono::system_clock::time_point> acqtimearray(
                img.md->size[2]);
            for (int i = 0; i < img.md->sem; ++i) {
              auto acqtime = std::chrono::seconds{img.atimearray[i].tv_sec} +
                             std::chrono::nanoseconds{img.atimearray[i].tv_nsec};
              std::chrono::system_clock::time_point tp{acqtime};
              acqtimearray[i] = tp;
            }
            return acqtimearray;
          })
      .def_property_readonly(
          "writetimearray",
          [](const IMAGE &img) {
            if (img.writetimearray == NULL)
              throw std::runtime_error("writetimearray not initialized");
            std::vector<std::chrono::system_clock::time_point> writetimearray(
                img.md->size[2]);
            for (int i = 0; i < img.md->sem; ++i) {
              auto writetime =
                  std::chrono::seconds{img.writetimearray[i].tv_sec} +
                  std::chrono::nanoseconds{img.writetimearray[i].tv_nsec};
              std::chrono::system_clock::time_point tp{writetime};
              writetimearray[i] = tp;
            }
            return writetimearray;
          })
      .def_property_readonly(
          "cntarray",
          [](const IMAGE &img) {
            if (img.cntarray == NULL)
              throw std::runtime_error("cntarray not initialized");
            std::vector<uint64_t> cntarray(img.md->size[2]);
            for (int i = 0; i < img.md->sem; ++i) {
              cntarray[i] = img.cntarray[i];
            }
            return cntarray;
          })
      .def_property_readonly(
          "flagarray",
          [](const IMAGE &img) {
            if (img.flagarray == NULL)
              throw std::runtime_error("flagarray not initialized");
            std::vector<uint64_t> flagarray(img.md->size[2]);
            for (int i = 0; i < img.md->sem; ++i) {
              flagarray[i] = img.flagarray[i];
            }
            return flagarray;
          })
      .def_property_readonly("semWritePID",
                             [](const IMAGE &img) {
                               std::vector<pid_t> semWritePID(img.md->sem);
                               for (int i = 0; i < img.md->sem; ++i) {
                                 semWritePID[i] = img.semWritePID[i];
                               }
                               return semWritePID;
                             })
      .def_property_readonly("kw",
                             [](const IMAGE &img) {
                               std::map<std::string, IMAGE_KEYWORD> keywords;
                               for (int i = 0; i < img.md->NBkw; ++i) {
                                 std::string key(img.kw[i].name);
                                 keywords[key] = img.kw[i];
                               }
                               return keywords;
                             })
      .def_buffer([](const IMAGE &img) -> py::buffer_info {
        if (img.md->location >= 0) {
          throw std::runtime_error("Can not use this with a GPU buffer");
        }

        ImageStreamIODataType dt(img.md->datatype);
        std::string format = ImageStreamIODataTypeToPyFormat(dt);
        std::vector<ssize_t> shape(img.md->naxis);
        std::vector<ssize_t> strides(img.md->naxis);
        ssize_t stride = dt.asize;

        // Row Major representation
        // for (int8_t axis(img.md->naxis-1); axis >= 0; --axis) {
        // Col Major representation
        for (int8_t axis(0); axis < img.md->naxis; ++axis) {
          shape[axis] = img.md->size[axis];
          strides[axis] = stride;
          stride *= shape[axis];
        }
        return py::buffer_info(
            img.array.UI8, /* Pointer to buffer */
            dt.asize,      /* Size of one scalar */
            format,        /* Python struct-style format descriptor */
            img.md->naxis, /* Number of dimensions */
            shape,         /* Buffer dimensions */
            strides        /* Strides (in bytes) for each index */
        );
      })

      .def("copy",
           [](const IMAGE &img) -> py::object {
             ImageStreamIODataType dt(img.md->datatype);
             switch (dt.datatype) {
               case ImageStreamIODataType::DataType::UINT8:
                 return convert_img<uint8_t>(img);
               case ImageStreamIODataType::DataType::INT8:
                 return convert_img<int8_t>(img);
               case ImageStreamIODataType::DataType::UINT16:
                 return convert_img<uint16_t>(img);
               case ImageStreamIODataType::DataType::INT16:
                 return convert_img<int16_t>(img);
               case ImageStreamIODataType::DataType::UINT32:
                 return convert_img<uint32_t>(img);
               case ImageStreamIODataType::DataType::INT32:
                 return convert_img<int32_t>(img);
               case ImageStreamIODataType::DataType::UINT64:
                 return convert_img<uint64_t>(img);
               case ImageStreamIODataType::DataType::INT64:
                 return convert_img<int64_t>(img);
               case ImageStreamIODataType::DataType::FLOAT:
                 return convert_img<float>(img);
               case ImageStreamIODataType::DataType::DOUBLE:
                 return convert_img<double>(img);
               // case ImageStreamIODataType::DataType::COMPLEX_FLOAT: return ;
               // case ImageStreamIODataType::DataType::COMPLEX_DOUBLE: return ;
               default:
                 throw std::runtime_error("Not implemented");
             }
           })

      .def(
          "write",
          [](IMAGE &img, py::buffer b) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();

            if (img.md->datatype !=
                PyFormatToImageStreamIODataType(info.format).datatype)
              throw std::invalid_argument("incompatible type");
            if (info.ndim != img.md->naxis)
              throw std::invalid_argument("incompatible number of axis");
            const uint32_t *size_ptr = img.md->size;
            for (auto &dim : info.shape) {
              if (*size_ptr != dim)
                throw std::invalid_argument("incompatible shape");
              ++size_ptr;
            }

            ImageStreamIODataType dt(img.md->datatype);
            uint8_t *buffer_ptr = (uint8_t *)info.ptr;
            uint64_t size = img.md->nelement * dt.asize;

            img.md->write = 1;  // set this flag to 1 when writing data
            std::copy(buffer_ptr, buffer_ptr + size, img.array.UI8);

            std::vector<uint32_t> ushape(info.ndim);
            std::copy(info.shape.begin(), info.shape.end(), ushape.begin());
            ImageStreamIO_sempost(&img, -1);
            clock_gettime(CLOCK_REALTIME, &img.md->lastaccesstime);
            img.md->write = 0;  // Done writing data
            img.md->cnt0++;
            img.md->cnt1++;
          },
          R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
          py::arg("buffer"))

      .def(
          "create",
          [](IMAGE &img, std::string name, py::array_t<uint32_t> dims,
             uint8_t datatype, uint8_t shared, uint16_t NBkw) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = dims.request();

            // uint8_t datatype =
            // PyFormatToImageStreamIODataType(info.format).datatype;
            // std::vector<uint32_t> ushape(info.ndim);
            // std::copy(info.shape.begin(), info.shape.end(), ushape.begin());

            return ImageStreamIO_createIm(&img, name.c_str(), info.size,
                                          (uint32_t *)info.ptr, datatype,
                                          shared, NBkw);
          },
          R"pbdoc(
            Create shared memory image stream
            Parameters:
                name     [in]:  the name of the shared memory file will be SHAREDMEMDIR/<name>_im.shm
                dims     [in]:  np.array of the image.
                datatype [in]:  data type code, pyImageStreamIO.Datatype
                shared   [in]:  if true then a shared memory buffer is allocated.  If false, only local storage is used.
                NBkw     [in]:  the number of keywords to allocate.
            Return:
                ret      [out]: error code
            )pbdoc",
          py::arg("name"), py::arg("dims"),
          py::arg("datatype") = ImageStreamIODataType::DataType::FLOAT,
          py::arg("shared") = 1, py::arg("NBkw") = 1)

      .def(
          "create",
          [](IMAGE &img, std::string name, py::array_t<uint32_t> dims,
             uint8_t datatype, int8_t location, uint8_t shared, int NBsem,
             int NBkw, uint64_t imagetype) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = dims.request();

            // uint8_t datatype =
            // PyFormatToImageStreamIODataType(info.format).datatype;
            // std::vector<uint32_t> ushape(info.ndim);
            // std::copy(info.shape.begin(), info.shape.end(), ushape.begin());

            return ImageStreamIO_createIm_gpu(
                &img, name.c_str(), info.size, (uint32_t *)info.ptr, datatype,
                location, shared, NBsem, NBkw, imagetype);
          },
          R"pbdoc(
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
            )pbdoc",
          py::arg("name"), py::arg("dims"),
          py::arg("datatype") = ImageStreamIODataType::DataType::FLOAT,
          py::arg("location") = -1, py::arg("shared") = 1,
          py::arg("NBsem") = IMAGE_NB_SEMAPHORE, py::arg("NBkw") = 1,
          py::arg("imagetype") = MATH_DATA)

      .def(
          "open",
          [](IMAGE &img, std::string name) {
            return ImageStreamIO_openIm(&img, name.c_str());
          },
          R"pbdoc(
            Open / connect to existing shared memory image stream
            Parameters:
                name   [in]:  the name of the shared memory file to connect
            Return:
                ret    [out]: error code
            )pbdoc",
          py::arg("name"))
      
      .def(         
          "destroy",
          [](IMAGE &img) {
            return ImageStreamIO_destroyIm(&img);
          },
          R"pbdoc(
            For a shared image:
            Closes all semaphores, deallcoates sem pointers,
            and removes associated files. Unmaps the shared memory
            segment, and finally removes the file. Sets the metadata and
            keyword pointers to NULL.

            For a non-shred image:
            Deallocates all arrays and sets pointers to NULL.
            )pbdoc")


      .def(
          "getsemwaitindex",
          [](IMAGE &img, long index) {
            return ImageStreamIO_getsemwaitindex(&img, index);
          },
          R"pbdoc(
            Get available shmim semaphore index

            Parameters:
                image	 [in]:  pointer to shmim (IMAGE)
                index  [in]:  preferred semaphore index, if available
            Return:
                ret    [out]: semaphore index available
            )pbdoc",
          py::arg("index"))

      .def(
          "semwait",
          [](IMAGE &img, long index) {
            return ImageStreamIO_semwait(&img, index);
          },
          R"pbdoc(
                Read / connect to existing shared memory image stream
                Parameters:
                    index  [in]:  index of semaphore to wait
                Return:
                    ret    [out]: error code
                )pbdoc",
          py::arg("index"))

      .def(
          "sempost",
          [](IMAGE &img, long index) {
            return ImageStreamIO_sempost(&img, index);
          },
          R"pbdoc(
                Read / connect to existing shared memory image stream
                Parameters:
                    index  [in]:  index of semaphore to be posted (-1 for all)
                Return:
                    ret    [out]: error code
                )pbdoc",
          py::arg("index")=-1)

      .def(
          "semflush",
          [](IMAGE &img, long index) {
            return ImageStreamIO_semflush(&img, index);
          },
          R"pbdoc(
                Flush shmim semaphore
                Parameters:
                    index  [in]:  index of semaphore to flush; flush all semaphores if index<0
                Return:
                    ret    [out]: error code
                )pbdoc",
          py::arg("index"));
}
