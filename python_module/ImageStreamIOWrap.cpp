#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctime>

#include "ImageStreamIO.h"
#include "ImageStruct.h"

namespace py = pybind11;

std::string toString(const IMAGE_KEYWORD &kw) {
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
}

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

  enum Type get_type() const {
    return static_cast<ImageStreamIOType::Type>(type & 0xF);
  }
  enum Type get_axis() const {
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
  if (pf == py::format_descriptor<uint8_t>::format()) {
    return ImageStreamIODataType::DataType::UINT8;
  }
  if (pf == py::format_descriptor<int8_t>::format()) {
    return ImageStreamIODataType::DataType::INT8;
  }
  if (pf == py::format_descriptor<uint16_t>::format()) {
    return ImageStreamIODataType::DataType::UINT16;
  }
  if (pf == py::format_descriptor<int16_t>::format()) {
    return ImageStreamIODataType::DataType::INT16;
  }
  if (pf == py::format_descriptor<uint32_t>::format()) {
    return ImageStreamIODataType::DataType::UINT32;
  }
  if (pf == py::format_descriptor<int32_t>::format()) {
    return ImageStreamIODataType::DataType::INT32;
  }
  if (pf == py::format_descriptor<uint64_t>::format()) {
    return ImageStreamIODataType::DataType::UINT64;
  }
  if (pf == py::format_descriptor<int64_t>::format()) {
    return ImageStreamIODataType::DataType::INT64;
  }
  if (pf == py::format_descriptor<float>::format()) {
    return ImageStreamIODataType::DataType::FLOAT;
  }
  if (pf == py::format_descriptor<double>::format()) {
    return ImageStreamIODataType::DataType::DOUBLE;
  }
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
  size_t size_data = img.md->nelement * sizeof(T);
  if (img.md->location == -1) {
    memcpy(ret_buffer.mutable_data(), current_image, size_data);
  } else {
#ifdef HAVE_CUDA
    cudaSetDevice(img.md->location);
    cudaMemcpy(ret_buffer.mutable_data(), current_image, size_data,
               cudaMemcpyDeviceToHost);
#else
    throw std::runtime_error(
        "unsupported location, CACAO needs to be compiled with -DUSE_CUDA=ON");
#endif
  }
  return ret_buffer;
}

template <typename T>
void write(IMAGE &img,
           py::array_t<T, py::array::f_style | py::array::forcecast> b) {
  if (img.array.raw == nullptr) {
    throw std::runtime_error("image not initialized");
  }
  /* Request a buffer descriptor from Python */
  py::buffer_info info = b.request();

  if (img.md->datatype !=
      PyFormatToImageStreamIODataType(info.format).datatype) {
    throw std::invalid_argument("incompatible type");
  }
  if (info.ndim != img.md->naxis) {
    throw std::invalid_argument("incompatible number of axis");
  }
  const uint32_t *size_ptr = img.md->size;
  for (auto &dim : info.shape) {
    if (*size_ptr != dim) {
      throw std::invalid_argument("incompatible shape");
    }
    ++size_ptr;
  }

  ImageStreamIODataType dt(img.md->datatype);
  uint8_t *buffer_ptr = (uint8_t *)info.ptr;
  uint64_t size = img.md->nelement * dt.asize;

  img.md->write = 1;  // set this flag to 1 when writing data

  void *current_image = img.array.raw;

  if (img.md->location == -1) {
    memcpy(current_image, buffer_ptr, size);
  } else {
#ifdef HAVE_CUDA
    cudaSetDevice(img.md->location);
    cudaMemcpy(current_image, buffer_ptr, size, cudaMemcpyHostToDevice);
#else
    throw std::runtime_error(
        "unsupported location, CACAO needs to be compiled with -DUSE_CUDA=ON");
#endif
  }
  ImageStreamIO_sempost(&img, -1);
  clock_gettime(CLOCK_REALTIME, &img.md->lastaccesstime);
  img.md->write = 0;  // Done writing data
  img.md->cnt0++;
  img.md->cnt1++;
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
          .def_readonly("type", &ImageStreamIODataType::datatype)
          .def("__repr__", [](const ImageStreamIODataType &img_datatype) {
            std::ostringstream tmp_str;
            tmp_str << "datatype: " << img_datatype.datatype << std::endl;
            tmp_str << "size: " << img_datatype.asize << std::endl;
            return tmp_str.str();
          });

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

  auto imagetype =
      py::class_<ImageStreamIOType>(m, "ImageStreamIOType")
          .def(py::init([](uint8_t type) {
            return std::unique_ptr<ImageStreamIOType>(
                new ImageStreamIOType(type));
          }))
          .def_property_readonly("axis", &ImageStreamIOType::get_axis)
          .def_property_readonly("type", &ImageStreamIOType::get_type)
          .def("__repr__", [](const ImageStreamIOType &image_type) {
            std::ostringstream tmp_str;
            tmp_str << "type: " << image_type.get_type() << std::endl;
            tmp_str << "axis: " << image_type.get_axis() << std::endl;
            return tmp_str.str();
          });

  py::enum_<ImageStreamIOType::Type>(imagetype, "Type")
      .value("CIRCULAR_BUFFER_TYPE",
             ImageStreamIOType::Type::CIRCULAR_BUFFER_TYPE)
      .value("MATH_DATA_TYPE", ImageStreamIOType::Type::MATH_DATA_TYPE)
      .value("IMG_RECV_TYPE", ImageStreamIOType::Type::IMG_RECV_TYPE)
      .value("IMG_SENT_TYPE", ImageStreamIOType::Type::IMG_SENT_TYPE)
      .value("ZAXIS_UNDEF_TYPE", ImageStreamIOType::Type::ZAXIS_UNDEF_TYPE)
      .value("ZAXIS_SPACIAL_TYPE", ImageStreamIOType::Type::ZAXIS_SPACIAL_TYPE)
      .value("ZAXIS_TEMPORAL_TYPE",
             ImageStreamIOType::Type::ZAXIS_TEMPORAL_TYPE)
      .value("ZAXIS_WAVELENGTH_TYPE",
             ImageStreamIOType::Type::ZAXIS_WAVELENGTH_TYPE)
      .value("ZAXIS_MAPPING_TYPE", ImageStreamIOType::Type::ZAXIS_MAPPING_TYPE)
      .export_values();

  // IMAGE_KEYWORD interface
  py::class_<IMAGE_KEYWORD>(m, "Image_kw")
      .def(py::init(
          []() { return std::unique_ptr<IMAGE_KEYWORD>(new IMAGE_KEYWORD()); }))
      .def(py::init([](std::string name, int64_t numl, std::string comment) {
             if (name.size() > KEYWORD_MAX_STRING) {
               throw std::invalid_argument("name too long");
             }
             if (comment.size() > KEYWORD_MAX_COMMENT) {
               throw std::invalid_argument("comment too long");
             }
             auto kw = std::unique_ptr<IMAGE_KEYWORD>(new IMAGE_KEYWORD());
             std::copy(name.begin(), name.end(), kw->name);
             kw->type = 'L';
             kw->value.numl = numl;
             std::copy(comment.begin(), comment.end(), kw->comment);
             return kw;
           }),
           py::arg("name"), py::arg("numl"), py::arg("comment") = "")
      .def(py::init([](std::string name, double numf, std::string comment) {
             if (name.size() > KEYWORD_MAX_STRING) {
               throw std::invalid_argument("name too long");
             }
             if (comment.size() > KEYWORD_MAX_COMMENT) {
               throw std::invalid_argument("comment too long");
             }
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
                 if (name.size() > KEYWORD_MAX_STRING) {
                   throw std::invalid_argument("name too long");
                 }
                 if (valstr.size() > KEYWORD_MAX_STRING) {
                   throw std::invalid_argument("valstr too long");
                 }
                 if (comment.size() > KEYWORD_MAX_COMMENT) {
                   throw std::invalid_argument("comment too long");
                 }
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
      .def("__str__", [](const IMAGE_KEYWORD &kw) { return toString(kw); })
      .def("__repr__", [](const IMAGE_KEYWORD &kw) { return toString(kw); })
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
      .def_property_readonly(
          "datatype",
          [](const IMAGE_METADATA &md) {
            return ImageStreamIODataType(md.datatype).datatype;
          })
      .def_property_readonly(
          "imagetype",
          [](const IMAGE_METADATA &md) {
            return ImageStreamIOType(md.imagetype).get_type();
          })
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
      .def_property_readonly("location_str",
                             [](const IMAGE_METADATA &md) {
                               if (md.location < 0) {
                                 return std::string("CPU RAM");
                               }

                               std::ostringstream tmp_str;
                               tmp_str << "GPU" << int(md.location) << " RAM";
                               return tmp_str.str();
                             })
      .def_readonly("status", &IMAGE_METADATA::status)
      .def_readonly("logflag", &IMAGE_METADATA::logflag)
      .def_readonly("sem", &IMAGE_METADATA::sem)
      .def_readonly("cnt0", &IMAGE_METADATA::cnt0)
      .def_readonly("cnt1", &IMAGE_METADATA::cnt1)
      .def_readonly("cnt2", &IMAGE_METADATA::cnt2)
      .def_readonly("write", &IMAGE_METADATA::write)
      .def_readonly("flag", &IMAGE_METADATA::flag)
      .def_readonly("NBkw", &IMAGE_METADATA::NBkw)
      .def("__repr__", [](const IMAGE_METADATA &md) {
        std::ostringstream tmp_str;
        tmp_str << "Name: " << md.name << std::endl;
        tmp_str << "Version: " << md.version << std::endl;
        tmp_str << "Size: [" << md.size[0];
        for (int i = 1; i < md.naxis; ++i) {
          tmp_str << ", " << md.size[i];
        }
        tmp_str << "]" << std::endl;
        tmp_str << "nelement: " << md.nelement << std::endl;
        // tmp_str << "datatype: " << md.datatype << std::endl;
        // tmp_str << "imagetype: " << md.imagetype << std::endl;
        {
          auto creationtime = std::chrono::seconds{md.creationtime.tv_sec} +
                              std::chrono::nanoseconds{md.creationtime.tv_nsec};
          std::chrono::system_clock::time_point tp{creationtime};
          std::time_t t = std::chrono::system_clock::to_time_t(tp);
          tmp_str << "creationtime: " << std::ctime(&t);
        }
        {
          auto lastaccesstime =
              std::chrono::seconds{md.lastaccesstime.tv_sec} +
              std::chrono::nanoseconds{md.lastaccesstime.tv_nsec};
          std::chrono::system_clock::time_point tp{lastaccesstime};
          std::time_t t = std::chrono::system_clock::to_time_t(tp);
          tmp_str << "lastaccesstime: " << std::ctime(&t);
        }
        {
          auto acqtime = std::chrono::seconds{md.atime.tv_sec} +
                         std::chrono::nanoseconds{md.atime.tv_nsec};
          std::chrono::system_clock::time_point tp{acqtime};
          std::time_t t = std::chrono::system_clock::to_time_t(tp);
          tmp_str << "acqtime: " << std::ctime(&t);
        }
        tmp_str << "shared: " << int(md.shared) << std::endl;
        tmp_str << "location: ";
        if (md.location < 0) {
          tmp_str << "CPU RAM" << std::endl;
        } else {
          tmp_str << "GPU" << int(md.location) << " RAM" << std::endl;
        }
        tmp_str << "flag: " << md.flag << std::endl;
        tmp_str << "logflag: " << int(md.logflag) << std::endl;
        tmp_str << "sem: " << md.sem << std::endl;
        tmp_str << "cnt0: " << md.cnt0 << std::endl;
        tmp_str << "cnt1: " << md.cnt1 << std::endl;
        tmp_str << "cnt2: " << md.cnt2;

        return tmp_str.str();
      });

  // IMAGE interface
  py::class_<IMAGE>(m, "Image", py::buffer_protocol())
      .def(py::init([]() { return std::unique_ptr<IMAGE>(new IMAGE()); }))
      .def_readonly("used", &IMAGE::used)
      .def_readonly("memsize", &IMAGE::memsize)
      .def_readonly("md", &IMAGE::md)
      .def_property_readonly(
          "shape",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            py::tuple dims(img.md->naxis);
            const uint32_t *ptr = img.md->size;
            // std::copy(ptr, ptr + img.md->naxis, dims);
            for (int i{}; i < img.md->naxis; ++i) {
              dims[i] = ptr[i];
            }
            return dims;
          })

      .def_property_readonly(
          "semReadPID",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::vector<pid_t> semReadPID(img.md->sem);
            for (int i = 0; i < img.md->sem; ++i) {
              semReadPID[i] = img.semReadPID[i];
            }
            return semReadPID;
          })
      .def_property_readonly(
          "acqtimearray",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img.atimearray == NULL) {
              throw std::runtime_error("acqtimearray not initialized");
            }
            std::vector<std::chrono::system_clock::time_point> acqtimearray(
                img.md->size[2]);
            for (int i = 0; i < img.md->sem; ++i) {
              auto acqtime =
                  std::chrono::seconds{img.atimearray[i].tv_sec} +
                  std::chrono::nanoseconds{img.atimearray[i].tv_nsec};
              std::chrono::system_clock::time_point tp{acqtime};
              acqtimearray[i] = tp;
            }
            return acqtimearray;
          })
      .def_property_readonly(
          "writetimearray",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img.writetimearray == NULL) {
              throw std::runtime_error("writetimearray not initialized");
            }
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
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img.cntarray == NULL) {
              throw std::runtime_error("cntarray not initialized");
            }
            std::vector<uint64_t> cntarray(img.md->size[2]);
            for (int i = 0; i < img.md->sem; ++i) {
              cntarray[i] = img.cntarray[i];
            }
            return cntarray;
          })
      .def_property_readonly(
          "flagarray",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img.flagarray == NULL) {
              throw std::runtime_error("flagarray not initialized");
            }
            std::vector<uint64_t> flagarray(img.md->size[2]);
            for (int i = 0; i < img.md->sem; ++i) {
              flagarray[i] = img.flagarray[i];
            }
            return flagarray;
          })
      .def_property_readonly(
          "semWritePID",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::vector<pid_t> semWritePID(img.md->sem);
            for (int i = 0; i < img.md->sem; ++i) {
              semWritePID[i] = img.semWritePID[i];
            }
            return semWritePID;
          })
      .def("get_kws",
           [](const IMAGE &img) {
             if (img.array.raw == nullptr) {
               throw std::runtime_error("image not initialized");
             }
             std::map<std::string, IMAGE_KEYWORD> keywords;
             for (int i = 0; i < img.md->NBkw; ++i) {
               if (strcmp(img.kw[i].name, "") == 0) {
                 break;
               }
               std::string key(img.kw[i].name);
               keywords[key] = img.kw[i];
             }
             return keywords;
           })
      .def(
          "set_kws",
          [](const IMAGE &img, std::map<std::string, IMAGE_KEYWORD> &keywords) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::map<std::string, IMAGE_KEYWORD>::iterator it =
                keywords.begin();
            int cnt = 0;
            while (it != keywords.end()) {
              if (cnt >= img.md->NBkw)
                throw std::runtime_error("Too many keywords provided");
              img.kw[cnt] = it->second;
              it++;
              cnt++;
            }
            // Pad with empty keywords
            if (cnt < img.md->NBkw) {
              img.kw[cnt] = IMAGE_KEYWORD();
            }
          })
      .def_buffer([](const IMAGE &img) -> py::buffer_info {
        if (img.array.raw == nullptr) {
          py::print("image not initialized");
          return py::buffer_info();
        }
        if (img.md->location >= 0) {
          py::print("Can not use this with a GPU buffer");
          return py::buffer_info();
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
            img.array.raw, /* Pointer to buffer */
            dt.asize,      /* Size of one scalar */
            format,        /* Python struct-style format descriptor */
            img.md->naxis, /* Number of dimensions */
            shape,         /* Buffer dimensions */
            strides        /* Strides (in bytes) for each index */
        );
      })

      .def("copy",
           [](const IMAGE &img) -> py::object {
             if (img.array.raw == nullptr)
               throw std::runtime_error("image not initialized");
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

      .def("write", &write<uint8_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<uint16_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<uint32_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<uint64_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<int8_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<int16_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<int32_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<int64_t>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<float>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def("write", &write<double>,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           py::arg("buffer"))

      .def(
          "create",
          [](IMAGE &img, const std::string &name, const py::buffer &buffer,
             int8_t location, uint8_t shared, int NBsem, int NBkw,
             uint64_t imagetype) {
            py::buffer_info info = buffer.request();

            auto buf = pybind11::array::ensure(buffer);

            if (!buf) {
              throw std::invalid_argument("input buffer is not an np.array");
            }

            uint8_t datatype =
                PyFormatToImageStreamIODataType(info.format).datatype;

            uint32_t dims[buf.ndim()];
            for (int i = 0; i < buf.ndim(); ++i) {
              dims[i] = buf.shape()[i];
            }

            int res = ImageStreamIO_createIm_gpu(
                &img, name.c_str(), buf.ndim(), dims, datatype, location,
                shared, NBsem, NBkw, imagetype);
            if (res == 0) {
              if (buf.dtype() == pybind11::dtype::of<uint8_t>()) {
                write<uint8_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int8_t>()) {
                write<int8_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<uint16_t>()) {
                write<uint16_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int16_t>()) {
                write<int16_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<uint32_t>()) {
                write<uint32_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int32_t>()) {
                write<int32_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<uint64_t>()) {
                write<uint64_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int64_t>()) {
                write<int64_t>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<float>()) {
                write<float>(img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<double>()) {
                write<double>(img, buffer);
              } else {
                throw std::invalid_argument("unsupported array datatype");
              }
            }
            return res;
          },
          R"pbdoc(
            Create shared memory image stream
            Parameters:
                name     [in]:  the name of the shared memory file will be SHAREDMEMDIR/<name>_im.shm
                buffer   [in]:  np.array of the image.
                location [in]:  location of allocate the image (-1 for CPU or GPU number)
                shared   [in]:  if true then a shared memory buffer is allocated.  If false, only local storage is used.
                NBkw     [in]:  the number of keywords to allocate.
                NBsem    [in]:  the number of semaphore to attach.
                imagetype[in]:  the type of the image to create (ImageStreamIOType).
            Return:
                ret      [out]: error code
            )pbdoc",
          py::arg("name"), py::arg("buffer"), py::arg("location") = -1,
          py::arg("shared") = 1, py::arg("NBsem") = IMAGE_NB_SEMAPHORE,
          py::arg("NBkw") = 1, py::arg("imagetype") = MATH_DATA)

      // .def(
      //     "create",
      //     [](IMAGE &img, std::string name, py::array_t<uint32_t> dims,
      //        uint8_t datatype, uint8_t shared, uint16_t NBkw) {
      //       /* Request a buffer descriptor from Python */
      //       py::buffer_info info = dims.request();

      //       // uint8_t datatype =
      //       // PyFormatToImageStreamIODataType(info.format).datatype;
      //       // std::vector<uint32_t> ushape(info.ndim);
      //       // std::copy(info.shape.begin(), info.shape.end(),
      //       ushape.begin());

      //       return ImageStreamIO_createIm(&img, name.c_str(), info.size,
      //                                     (uint32_t *)info.ptr, datatype,
      //                                     shared, NBkw);
      //     },
      //     R"pbdoc(
      //       Create shared memory image stream
      //       Parameters:
      //           name     [in]:  the name of the shared memory file will be
      //           SHAREDMEMDIR/<name>_im.shm dims     [in]:  np.array of the
      //           image. datatype [in]:  data type code,
      //           pyImageStreamIO.Datatype shared   [in]:  if true then a
      //           shared memory buffer is allocated.  If false, only local
      //           storage is used. NBkw     [in]:  the number of keywords to
      //           allocate.
      //       Return:
      //           ret      [out]: error code
      //       )pbdoc",
      //     py::arg("name"), py::arg("dims"),
      //     py::arg("datatype") = ImageStreamIODataType::DataType::FLOAT,
      //     py::arg("shared") = 1, py::arg("NBkw") = 1)

      // .def(
      //     "create",
      //     [](IMAGE &img, std::string name, py::array_t<uint32_t> dims,
      //        uint8_t datatype, int8_t location, uint8_t shared, int NBsem,
      //        int NBkw, uint64_t imagetype) {
      //       /* Request a buffer descriptor from Python */
      //       py::buffer_info info = dims.request();

      //       // uint8_t datatype =
      //       // PyFormatToImageStreamIODataType(info.format).datatype;
      //       // std::vector<uint32_t> ushape(info.ndim);
      //       // std::copy(info.shape.begin(), info.shape.end(),
      //       ushape.begin());

      //       return ImageStreamIO_createIm_gpu(
      //           &img, name.c_str(), info.size, (uint32_t *)info.ptr,
      //           datatype, location, shared, NBsem, NBkw, imagetype);
      //     },
      //     R"pbdoc(
      //       Create shared memory image stream
      //       Parameters:
      //           name      [in]:  the name of the shared memory file will be
      //           SHAREDMEMDIR/<name>_im.shm dims      [in]:  np.array of the
      //           image. datatype  [in]:  data type code,
      //           pyImageStreamIO.Datatype shared    [in]:  if true then a
      //           shared memory buffer is allocated.  If false, only local
      //           storage is used. NBsem     [in]:  the number of semaphores to
      //           allocate. NBkw      [in]:  the number of keywords to
      //           allocate. imagetype [in]:  type of the stream.
      //       Return:
      //           ret       [out]: error code
      //       )pbdoc",
      //     py::arg("name"), py::arg("dims"),
      //     py::arg("datatype") = ImageStreamIODataType::DataType::FLOAT,
      //     py::arg("location") = -1, py::arg("shared") = 1,
      //     py::arg("NBsem") = IMAGE_NB_SEMAPHORE, py::arg("NBkw") = 1,
      //     py::arg("imagetype") = MATH_DATA)

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
          "close",
          [](IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_closeIm(&img);
          },
          R"pbdoc(
            Close a shared memory image stream
            Parameters:
                image  [in]:  pointer to shmim (IMAGE)
            Return:
                ret    [out]: error code
            )pbdoc")

      .def(
          "destroy",
          [](IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
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
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
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
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
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
          "semtimedwait",
          [](IMAGE &img, long index, float timeoutsec) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            struct timespec timeout;
            clock_gettime(CLOCK_REALTIME, &timeout);
            timeout.tv_nsec += (long)(timeoutsec * 1000000000L);
            timeout.tv_sec += timeout.tv_nsec / 1000000000L;
            timeout.tv_nsec = timeout.tv_nsec % 1000000000L;
            return ImageStreamIO_semtimedwait(&img, index, &timeout);
          },
          R"pbdoc(
                Read / connect to existing shared memory image stream
                Parameters:
                    index  [in]:  index of semaphore to wait
                Return:
                    ret    [out]: error code
                )pbdoc",
          py::arg("index"), py::arg("timeoutsec"))

      .def(
          "sempost",
          [](IMAGE &img, long index) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_sempost(&img, index);
          },
          R"pbdoc(
                Read / connect to existing shared memory image stream
                Parameters:
                    index  [in]:  index of semaphore to be posted (-1 for all)
                Return:
                    ret    [out]: error code
                )pbdoc",
          py::arg("index") = -1)

      .def(
          "semflush",
          [](IMAGE &img, long index) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
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
