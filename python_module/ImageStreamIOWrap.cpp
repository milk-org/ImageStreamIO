#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctime>

#include "ImageStreamIO.h"
#include "ImageStruct.h"

namespace py = pybind11;


// Some crazy trivial wraps
/*
Objective: binding into an interactive terminal
TWO imagestreamIOwraps compiled at different versions.
Construct such as py::class<IMAGE_KEYWORD> directly templating a C struct
create a conflict in the interpreter namespace.
This wrapping avoids any direct presence of objects from libImageStreamIO in the pybind namespace.
*/
struct IMAGE_KEYWORD_B {
    IMAGE_KEYWORD kw;

    IMAGE_KEYWORD_B() : kw(){};
    IMAGE_KEYWORD_B(IMAGE_KEYWORD kw) : kw(kw){};
};

struct IMAGE_METADATA_B {
    IMAGE_METADATA md;

    IMAGE_METADATA_B() : md(){};
    IMAGE_METADATA_B(IMAGE_METADATA md) : md(md){};
};

struct IMAGE_B {
    IMAGE img;

    IMAGE_B() : img(){};
    IMAGE_B(IMAGE img) : img(img){};
};

std::string toString(const IMAGE_KEYWORD_B &kw) {
  std::ostringstream tmp_str;
  //  tmp_str << kw.name << ": ";
  switch (kw.kw.type) {
    case 'L':
      tmp_str << kw.kw.value.numl;
      break;
    case 'D':
      tmp_str << kw.kw.value.numf;
      break;
    case 'S':
      tmp_str << kw.kw.value.valstr;
      break;
    default:
      tmp_str << "Unknown format";
      break;
  }
  tmp_str << " " << kw.kw.comment;
  return tmp_str.str();
}


struct ImageStreamIOType_b {
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

  ImageStreamIOType_b() : type(MATH_DATA_TYPE){};

  ImageStreamIOType_b(uint64_t type)
      : type(static_cast<ImageStreamIOType_b::Type>(type)){};

  enum Type get_type() const {
    return static_cast<ImageStreamIOType_b::Type>(type & 0xF);
  }
  enum Type get_axis() const {
    return static_cast<ImageStreamIOType_b::Type>(type & 0xF0000);
  }
};

struct ImageStreamIODataType_b {
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

  ImageStreamIODataType_b() : datatype(FLOAT), asize(Size[FLOAT]){};

  ImageStreamIODataType_b(uint8_t datatype)
      : datatype(static_cast<ImageStreamIODataType_b::DataType>(datatype)),
        asize(Size[datatype]){};
};

const std::vector<uint8_t> ImageStreamIODataType_b::Size(
    {0, SIZEOF_DATATYPE_UINT8, SIZEOF_DATATYPE_INT8, SIZEOF_DATATYPE_UINT16,
     SIZEOF_DATATYPE_INT16, SIZEOF_DATATYPE_UINT32, SIZEOF_DATATYPE_INT32,
     SIZEOF_DATATYPE_UINT64, SIZEOF_DATATYPE_INT64, SIZEOF_DATATYPE_FLOAT,
     SIZEOF_DATATYPE_DOUBLE, SIZEOF_DATATYPE_COMPLEX_FLOAT,
     SIZEOF_DATATYPE_COMPLEX_DOUBLE, SIZEOF_DATATYPE_HALF});

std::string ImageStreamIODataTypeToPyFormat(ImageStreamIODataType_b dt) {
  switch (dt.datatype) {
    case ImageStreamIODataType_b::DataType::UINT8:
      return py::format_descriptor<uint8_t>::format();
    case ImageStreamIODataType_b::DataType::INT8:
      return py::format_descriptor<int8_t>::format();
    case ImageStreamIODataType_b::DataType::UINT16:
      return py::format_descriptor<uint16_t>::format();
    case ImageStreamIODataType_b::DataType::INT16:
      return py::format_descriptor<int16_t>::format();
    case ImageStreamIODataType_b::DataType::UINT32:
      return py::format_descriptor<uint32_t>::format();
    case ImageStreamIODataType_b::DataType::INT32:
      return py::format_descriptor<int32_t>::format();
    case ImageStreamIODataType_b::DataType::UINT64:
      return py::format_descriptor<uint64_t>::format();
    case ImageStreamIODataType_b::DataType::INT64:
      return py::format_descriptor<int64_t>::format();
    case ImageStreamIODataType_b::DataType::FLOAT:
      return py::format_descriptor<float>::format();
    case ImageStreamIODataType_b::DataType::DOUBLE:
      return py::format_descriptor<double>::format();
    // case ImageStreamIODataType_b::DataType::COMPLEX_FLOAT: return
    // py::format_descriptor<(std::complex<float>>::format(); case
    // ImageStreamIODataType_b::DataType::COMPLEX_DOUBLE: return
    // py::format_descriptor<(std::complex<double>>::format();
    default:
      throw std::runtime_error("Not implemented");
  }
}

ImageStreamIODataType_b PyFormatToImageStreamIODataType(const std::string &pf) {
  if (pf == py::format_descriptor<uint8_t>::format()) {
    return ImageStreamIODataType_b::DataType::UINT8;
  }
  if (pf == py::format_descriptor<int8_t>::format()) {
    return ImageStreamIODataType_b::DataType::INT8;
  }
  if (pf == py::format_descriptor<uint16_t>::format()) {
    return ImageStreamIODataType_b::DataType::UINT16;
  }
  if (pf == py::format_descriptor<int16_t>::format()) {
    return ImageStreamIODataType_b::DataType::INT16;
  }
  if (pf == py::format_descriptor<uint32_t>::format()) {
    return ImageStreamIODataType_b::DataType::UINT32;
  }
  if (pf == py::format_descriptor<int32_t>::format()) {
    return ImageStreamIODataType_b::DataType::INT32;
  }
  if (pf == py::format_descriptor<uint64_t>::format()) {
    return ImageStreamIODataType_b::DataType::UINT64;
  }
  if (pf == py::format_descriptor<int64_t>::format()) {
    return ImageStreamIODataType_b::DataType::INT64;
  }
  if (pf == py::format_descriptor<float>::format()) {
    return ImageStreamIODataType_b::DataType::FLOAT;
  }
  if (pf == py::format_descriptor<double>::format()) {
    return ImageStreamIODataType_b::DataType::DOUBLE;
  }
  // case ImageStreamIODataType_b::DataType::COMPLEX_FLOAT: return
  // py::format_descriptor<(std::complex<float>>::format(); case
  // ImageStreamIODataType_b::DataType::COMPLEX_DOUBLE: return
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

  ImageStreamIODataType_b dt(img.md->datatype);
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

PYBIND11_MODULE(ImageStreamIOWrap_backport, m) {
  m.doc() = "CACAO ImageStreamIO python module";

  auto imageDatatype =
      py::class_<ImageStreamIODataType_b>(m, "ImageStreamIODataType_b")
          .def(py::init([](uint8_t datatype) {
            return std::unique_ptr<ImageStreamIODataType_b>(
                new ImageStreamIODataType_b(datatype));
          }))
          .def_readonly("size", &ImageStreamIODataType_b::asize)
          .def_readonly("type", &ImageStreamIODataType_b::datatype)
          .def("__repr__", [](const ImageStreamIODataType_b &img_datatype) {
            std::ostringstream tmp_str;
            tmp_str << "datatype: " << img_datatype.datatype << std::endl;
            tmp_str << "size: " << img_datatype.asize << std::endl;
            return tmp_str.str();
          });

  py::enum_<ImageStreamIODataType_b::DataType>(imageDatatype, "Type")
      .value("UINT8", ImageStreamIODataType_b::DataType::UINT8)
      .value("INT8", ImageStreamIODataType_b::DataType::INT8)
      .value("UINT16", ImageStreamIODataType_b::DataType::UINT16)
      .value("INT16", ImageStreamIODataType_b::DataType::INT16)
      .value("UINT32", ImageStreamIODataType_b::DataType::UINT32)
      .value("INT32", ImageStreamIODataType_b::DataType::INT32)
      .value("UINT64", ImageStreamIODataType_b::DataType::UINT64)
      .value("INT64", ImageStreamIODataType_b::DataType::INT64)
      .value("HALF", ImageStreamIODataType_b::DataType::HALF)
      .value("FLOAT", ImageStreamIODataType_b::DataType::FLOAT)
      .value("DOUBLE", ImageStreamIODataType_b::DataType::DOUBLE)
      .value("COMPLEX_FLOAT", ImageStreamIODataType_b::DataType::COMPLEX_FLOAT)
      .value("COMPLEX_DOUBLE", ImageStreamIODataType_b::DataType::COMPLEX_DOUBLE)
      .export_values();

  auto imagetype =
      py::class_<ImageStreamIOType_b>(m, "ImageStreamIOType_b")
          .def(py::init([](uint8_t type) {
            return std::unique_ptr<ImageStreamIOType_b>(
                new ImageStreamIOType_b(type));
          }))
          .def_property_readonly("axis", &ImageStreamIOType_b::get_axis)
          .def_property_readonly("type", &ImageStreamIOType_b::get_type)
          .def("__repr__", [](const ImageStreamIOType_b &image_type) {
            std::ostringstream tmp_str;
            tmp_str << "type: " << image_type.get_type() << std::endl;
            tmp_str << "axis: " << image_type.get_axis() << std::endl;
            return tmp_str.str();
          });

  py::enum_<ImageStreamIOType_b::Type>(imagetype, "Type")
      .value("CIRCULAR_BUFFER_TYPE",
             ImageStreamIOType_b::Type::CIRCULAR_BUFFER_TYPE)
      .value("MATH_DATA_TYPE", ImageStreamIOType_b::Type::MATH_DATA_TYPE)
      .value("IMG_RECV_TYPE", ImageStreamIOType_b::Type::IMG_RECV_TYPE)
      .value("IMG_SENT_TYPE", ImageStreamIOType_b::Type::IMG_SENT_TYPE)
      .value("ZAXIS_UNDEF_TYPE", ImageStreamIOType_b::Type::ZAXIS_UNDEF_TYPE)
      .value("ZAXIS_SPACIAL_TYPE", ImageStreamIOType_b::Type::ZAXIS_SPACIAL_TYPE)
      .value("ZAXIS_TEMPORAL_TYPE",
             ImageStreamIOType_b::Type::ZAXIS_TEMPORAL_TYPE)
      .value("ZAXIS_WAVELENGTH_TYPE",
             ImageStreamIOType_b::Type::ZAXIS_WAVELENGTH_TYPE)
      .value("ZAXIS_MAPPING_TYPE", ImageStreamIOType_b::Type::ZAXIS_MAPPING_TYPE)
      .export_values();

  // IMAGE_KEYWORD_B interface
  py::class_<IMAGE_KEYWORD_B>(m, "Image_kw")
      .def(py::init(
          []() { return std::unique_ptr<IMAGE_KEYWORD_B>(new IMAGE_KEYWORD_B()); }))
      .def(py::init([](std::string name, int64_t numl, std::string comment) {
             if (name.size() > KEYWORD_MAX_STRING) {
               throw std::invalid_argument("name too long");
             }
             if (comment.size() > KEYWORD_MAX_COMMENT) {
               throw std::invalid_argument("comment too long");
             }
             auto kw = std::unique_ptr<IMAGE_KEYWORD_B>(new IMAGE_KEYWORD_B());
             std::copy(name.begin(), name.end(), kw->kw.name);
             kw->kw.type = 'L';
             kw->kw.value.numl = numl;
             std::copy(comment.begin(), comment.end(), kw->kw.comment);
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
             auto kw = std::unique_ptr<IMAGE_KEYWORD_B>(new IMAGE_KEYWORD_B());
             std::copy(name.begin(), name.end(), kw->kw.name);
             kw->kw.type = 'D';
             kw->kw.value.numf = numf;
             std::copy(comment.begin(), comment.end(), kw->kw.comment);
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
                 auto kw = std::unique_ptr<IMAGE_KEYWORD_B>(new IMAGE_KEYWORD_B());
                 std::copy(name.begin(), name.end(), kw->kw.name);
                 kw->kw.type = 'S';
                 std::copy(valstr.begin(), valstr.end(), kw->kw.value.valstr);
                 std::copy(comment.begin(), comment.end(), kw->kw.comment);
                 return kw;
               }),
           py::arg("name"), py::arg("valstr"), py::arg("comment") = "")
      .def_property_readonly("name", [](const IMAGE_KEYWORD_B &kw) { return kw.kw.name; })
      .def_property_readonly("type", [](const IMAGE_KEYWORD_B &kw) { return kw.kw.type; })
      .def_property_readonly("value",
                             [](const IMAGE_KEYWORD_B &kw) -> py::object {
                               switch (kw.kw.type) {
                                 case 'L':
                                   return py::int_(kw.kw.value.numl);
                                 case 'D':
                                   return py::float_(kw.kw.value.numf);
                                 case 'S':
                                   return py::str(kw.kw.value.valstr);
                                 default:
                                   throw std::runtime_error("Unknown format");
                               }
                             })
      .def("__str__", [](const IMAGE_KEYWORD_B &kw) { return toString(kw); })
      .def("__repr__", [](const IMAGE_KEYWORD_B &kw) { return toString(kw); })
      .def_property_readonly("comment", [](const IMAGE_KEYWORD_B &kw) { return kw.kw.comment; });
      //.def_readonly("comment", &IMAGE_KEYWORD_B::comment);

  // IMAGE_METADATA interface
  py::class_<IMAGE_METADATA_B>(m, "Image_md")
      // .def(py::init([]() {
      //     return std::unique_ptr<IMAGE_METADATA>(new IMAGE_METADATA());
      // }))
      .def_property_readonly("version", [](const IMAGE_METADATA_B &md) { return md.md.version; })
      .def_property_readonly("name", [](const IMAGE_METADATA_B &md) { return md.md.name; })
      .def_property_readonly("naxis", [](const IMAGE_METADATA_B &md) { return md.md.naxis; })
      .def_property_readonly("size",
                             [](const IMAGE_METADATA_B &md) {
                               std::vector<uint32_t> dims(md.md.naxis);
                               const uint32_t *ptr = md.md.size;
                               for (auto &&dim : dims) {
                                 dim = *ptr;
                                 ++ptr;
                               }
                               return dims;
                             })
      .def_property_readonly("nelement", [](const IMAGE_METADATA_B &md) { return md.md.nelement; })
      .def_property_readonly(
          "datatype",
          [](const IMAGE_METADATA_B &md) {
            return ImageStreamIODataType_b(md.md.datatype).datatype;
          })
      .def_property_readonly(
          "imagetype",
          [](const IMAGE_METADATA_B &md) {
            return ImageStreamIOType_b(md.md.imagetype).get_type();
          })
      .def_property_readonly(
          "creationtime",
          [](const IMAGE_METADATA_B &md) {
            auto creation_time =
                std::chrono::seconds{md.md.creationtime.tv_sec} +
                std::chrono::nanoseconds{md.md.creationtime.tv_nsec};
            std::chrono::system_clock::time_point tp{creation_time};
            return tp;
          })
      .def_property_readonly(
          "lastaccesstime",
          [](const IMAGE_METADATA_B &md) {
            auto creation_time =
                std::chrono::seconds{md.md.lastaccesstime.tv_sec} +
                std::chrono::nanoseconds{md.md.lastaccesstime.tv_nsec};
            std::chrono::system_clock::time_point tp{creation_time};
            return tp;
          })
      .def_property_readonly(
          "acqtime",
          [](const IMAGE_METADATA_B &md) {
            auto acqtime = std::chrono::seconds{md.md.atime.tv_sec} +
                           std::chrono::nanoseconds{md.md.atime.tv_nsec};
            std::chrono::system_clock::time_point tp{acqtime};
            return tp;
          })
      .def_property_readonly(
          "writetime",
          [](const IMAGE_METADATA_B &md) {
            auto writetime = std::chrono::seconds{md.md.writetime.tv_sec} +
                             std::chrono::nanoseconds{md.md.writetime.tv_nsec};
            std::chrono::system_clock::time_point tp{writetime};
            return tp;
          })
      .def_property_readonly("shared", [](const IMAGE_METADATA_B &md) { return md.md.shared; })
      .def_property_readonly("location", [](const IMAGE_METADATA_B &md) { return md.md.location; })
      .def_property_readonly("location_str",
                             [](const IMAGE_METADATA_B &md) {
                               if (md.md.location < 0) {
                                 return std::string("CPU RAM");
                               }

                               std::ostringstream tmp_str;
                               tmp_str << "GPU" << int(md.md.location) << " RAM";
                               return tmp_str.str();
                             })
      .def_property_readonly("status", [](const IMAGE_METADATA_B &md) { return md.md.status; })
      .def_property_readonly("logflag", [](const IMAGE_METADATA_B &md) { return md.md.logflag; })
      .def_property_readonly("sem", [](const IMAGE_METADATA_B &md) { return md.md.sem; })
      .def_property_readonly("cnt0", [](const IMAGE_METADATA_B &md) { return md.md.cnt0; })
      .def_property_readonly("cnt1", [](const IMAGE_METADATA_B &md) { return md.md.cnt1; })
      .def_property_readonly("cnt2", [](const IMAGE_METADATA_B &md) { return md.md.cnt2; })
      .def_property_readonly("write", [](const IMAGE_METADATA_B &md) { return md.md.write; })
      .def_property_readonly("flag", [](const IMAGE_METADATA_B &md) { return md.md.flag; })
      .def_property_readonly("NBkw", [](const IMAGE_METADATA_B &md) { return md.md.NBkw; })
      .def("__repr__", [](const IMAGE_METADATA_B &md) {
        std::ostringstream tmp_str;
        tmp_str << "Name: " << md.md.name << std::endl;
        tmp_str << "Version: " << md.md.version << std::endl;
        tmp_str << "Size: [" << md.md.size[0];
        for (int i = 1; i < md.md.naxis; ++i) {
          tmp_str << ", " << md.md.size[i];
        }
        tmp_str << "]" << std::endl;
        tmp_str << "nelement: " << md.md.nelement << std::endl;
        // tmp_str << "datatype: " << md.datatype << std::endl;
        // tmp_str << "imagetype: " << md.imagetype << std::endl;
        {
          auto creationtime = std::chrono::seconds{md.md.creationtime.tv_sec} +
                              std::chrono::nanoseconds{md.md.creationtime.tv_nsec};
          std::chrono::system_clock::time_point tp{creationtime};
          std::time_t t = std::chrono::system_clock::to_time_t(tp);
          tmp_str << "creationtime: " << std::ctime(&t);
        }
        {
          auto lastaccesstime =
              std::chrono::seconds{md.md.lastaccesstime.tv_sec} +
              std::chrono::nanoseconds{md.md.lastaccesstime.tv_nsec};
          std::chrono::system_clock::time_point tp{lastaccesstime};
          std::time_t t = std::chrono::system_clock::to_time_t(tp);
          tmp_str << "lastaccesstime: " << std::ctime(&t);
        }
        {
          auto acqtime = std::chrono::seconds{md.md.atime.tv_sec} +
                         std::chrono::nanoseconds{md.md.atime.tv_nsec};
          std::chrono::system_clock::time_point tp{acqtime};
          std::time_t t = std::chrono::system_clock::to_time_t(tp);
          tmp_str << "acqtime: " << std::ctime(&t);
        }
        tmp_str << "shared: " << int(md.md.shared) << std::endl;
        tmp_str << "location: ";
        if (md.md.location < 0) {
          tmp_str << "CPU RAM" << std::endl;
        } else {
          tmp_str << "GPU" << int(md.md.location) << " RAM" << std::endl;
        }
        tmp_str << "flag: " << md.md.flag << std::endl;
        tmp_str << "logflag: " << int(md.md.logflag) << std::endl;
        tmp_str << "sem: " << md.md.sem << std::endl;
        tmp_str << "cnt0: " << md.md.cnt0 << std::endl;
        tmp_str << "cnt1: " << md.md.cnt1 << std::endl;
        tmp_str << "cnt2: " << md.md.cnt2;

        return tmp_str.str();
      });


  // IMAGE interface
  py::class_<IMAGE_B>(m, "Image", py::buffer_protocol())
      .def(py::init([]() { return std::unique_ptr<IMAGE_B>(new IMAGE_B()); }))
      .def_property_readonly("used", [](const IMAGE_B &img_b) { return img_b.img.used; })
      .def_property_readonly("memsize", [](const IMAGE_B &img_b) { return img_b.img.memsize; })
      .def_property_readonly("md", [](const IMAGE_B &img_b) { return img_b.img.md; })
      .def_property_readonly(
          "shape",
          [](const IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            py::tuple dims(img_b.img.md->naxis);
            const uint32_t *ptr = img_b.img.md->size;
            // std::copy(ptr, ptr + img.md->naxis, dims);
            for (int i{}; i < img_b.img.md->naxis; ++i) {
              dims[i] = ptr[i];
            }
            return dims;
          })

      .def_property_readonly(
          "semReadPID",
          [](const IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::vector<pid_t> semReadPID(img_b.img.md->sem);
            for (int i = 0; i < img_b.img.md->sem; ++i) {
              semReadPID[i] = img_b.img.semReadPID[i];
            }
            return semReadPID;
          })
      .def_property_readonly(
          "acqtimearray",
          [](const IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img_b.img.atimearray == NULL) {
              throw std::runtime_error("acqtimearray not initialized");
            }
            std::vector<std::chrono::system_clock::time_point> acqtimearray(
                img_b.img.md->size[2]);
            for (int i = 0; i < img_b.img.md->size[2]; ++i) {
              auto acqtime =
                  std::chrono::seconds{img_b.img.atimearray[i].tv_sec} +
                  std::chrono::nanoseconds{img_b.img.atimearray[i].tv_nsec};
              std::chrono::system_clock::time_point tp{acqtime};
              acqtimearray[i] = tp;
            }
            return acqtimearray;
          })
      .def_property_readonly(
          "writetimearray",
          [](const IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img_b.img.writetimearray == NULL) {
              throw std::runtime_error("writetimearray not initialized");
            }
            std::vector<std::chrono::system_clock::time_point> writetimearray(
                img_b.img.md->size[2]);
            for (int i = 0; i < img_b.img.md->size[2]; ++i) {
              auto writetime =
                  std::chrono::seconds{img_b.img.writetimearray[i].tv_sec} +
                  std::chrono::nanoseconds{img_b.img.writetimearray[i].tv_nsec};
              std::chrono::system_clock::time_point tp{writetime};
              writetimearray[i] = tp;
            }
            return writetimearray;
          })
      .def_property_readonly(
          "cntarray",
          [](const IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img_b.img.cntarray == NULL) {
              throw std::runtime_error("cntarray not initialized");
            }
            std::vector<uint64_t> cntarray(img_b.img.md->size[2]);
            for (int i = 0; i < img_b.img.md->size[2]; ++i) {
              cntarray[i] = img_b.img.cntarray[i];
            }
            return cntarray;
          })
      // TODO: fix flagarray never allocated and cause segfaults
      // .def_property_readonly(
      //     "flagarray",
      //     [](const IMAGE_B &img) {
      //       if (img.array.raw == nullptr) {
      //         throw std::runtime_error("image not initialized");
      //       }
      //       if (img.flagarray == NULL) {
      //         throw std::runtime_error("flagarray not initialized");
      //       }
      //       std::vector<uint64_t> flagarray(img.md->size[2]);
      //       for (int i = 0; i < img.md->size[2]; ++i) {
      //         flagarray[i] = img.flagarray[i];
      //       }
      //       return flagarray;
      //     })
      .def_property_readonly(
          "semWritePID",
          [](const IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::vector<pid_t> semWritePID(img_b.img.md->sem);
            for (int i = 0; i < img_b.img.md->sem; ++i) {
              semWritePID[i] = img_b.img.semWritePID[i];
            }
            return semWritePID;
          })
      .def("get_kws",
           [](const IMAGE_B &img_b) {
             if (img_b.img.array.raw == nullptr) {
               throw std::runtime_error("image not initialized");
             }
             std::map<std::string, IMAGE_KEYWORD_B> keywords;
             for (int i = 0; i < img_b.img.md->NBkw; ++i) {
               if (strcmp(img_b.img.kw[i].name, "") == 0) {
                 break;
               }
               std::string key(img_b.img.kw[i].name);
               keywords[key] = img_b.img.kw[i];
             }
             return keywords;
           })

      .def("get_kws_list",
           [](const IMAGE_B &img_b) {
             if (img_b.img.array.raw == nullptr) {
               throw std::runtime_error("image not initialized");
             }
             std::list<IMAGE_KEYWORD_B> keywords;
             for (int i = 0; i < img_b.img.md->NBkw; ++i) {
               if (strcmp(img_b.img.kw[i].name, "") == 0) {
                 break;
               }
               keywords.push_back(IMAGE_KEYWORD_B(img_b.img.kw[i]));
             }
             return keywords;
           })

      .def("set_kws",
          [](const IMAGE_B &img_b, std::map<std::string, IMAGE_KEYWORD_B> &keywords) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::map<std::string, IMAGE_KEYWORD_B>::iterator it =
                keywords.begin();
            int cnt = 0;
            while (it != keywords.end()) {
              if (cnt >= img_b.img.md->NBkw)
                throw std::runtime_error("Too many keywords provided");
              img_b.img.kw[cnt] = (it->second).kw;
              it++;
              cnt++;
            }
            // Pad with empty keywords
            if (cnt < img_b.img.md->NBkw) {
              img_b.img.kw[cnt] = IMAGE_KEYWORD_B().kw;
            }
          })

      .def("set_kws_list",
          [](const IMAGE_B &img_b, std::list<IMAGE_KEYWORD_B> &keywords) {
            IMAGE img = img_b.img;
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::list<IMAGE_KEYWORD_B>::iterator it =
                keywords.begin();
            int cnt = 0;
            while (it != keywords.end()) {
              if (cnt >= img_b.img.md->NBkw)
                throw std::runtime_error("Too many keywords provided");
              img_b.img.kw[cnt] = (*it).kw;
              it++;
              cnt++;
            }
            // Pad with empty keywords
            if (cnt < img_b.img.md->NBkw) {
              img_b.img.kw[cnt] = IMAGE_KEYWORD_B().kw;
            }
          })


      .def_buffer([](const IMAGE_B &img_b) -> py::buffer_info {
        if (img_b.img.array.raw == nullptr) {
          py::print("image not initialized");
          return py::buffer_info();
        }
        if (img_b.img.md->location >= 0) {
          py::print("Can not use this with a GPU buffer");
          return py::buffer_info();
        }

        ImageStreamIODataType_b dt(img_b.img.md->datatype);
        std::string format = ImageStreamIODataTypeToPyFormat(dt);
        std::vector<ssize_t> shape(img_b.img.md->naxis);
        std::vector<ssize_t> strides(img_b.img.md->naxis);
        ssize_t stride = dt.asize;

        // Row Major representation
        // for (int8_t axis(img.md->naxis-1); axis >= 0; --axis) {
        // Col Major representation
        for (int8_t axis(0); axis < img_b.img.md->naxis; ++axis) {
          shape[axis] = img_b.img.md->size[axis];
          strides[axis] = stride;
          stride *= shape[axis];
        }
        return py::buffer_info(
            img_b.img.array.raw, // Pointer to buffer
            dt.asize,      // Size of one scalar
            format,        // Python struct-style format descriptor
            img_b.img.md->naxis, // Number of dimensions
            shape,         // Buffer dimensions
            strides        // Strides (in bytes) for each index
        );
      })

      .def("copy",
           [](const IMAGE_B &img_b) -> py::object {
             if (img_b.img.array.raw == nullptr)
               throw std::runtime_error("image not initialized");
             ImageStreamIODataType_b dt(img_b.img.md->datatype);
             switch (dt.datatype) {
               case ImageStreamIODataType_b::DataType::UINT8:
                 return convert_img<uint8_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::INT8:
                 return convert_img<int8_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::UINT16:
                 return convert_img<uint16_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::INT16:
                 return convert_img<int16_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::UINT32:
                 return convert_img<uint32_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::INT32:
                 return convert_img<int32_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::UINT64:
                 return convert_img<uint64_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::INT64:
                 return convert_img<int64_t>(img_b.img);
               case ImageStreamIODataType_b::DataType::FLOAT:
                 return convert_img<float>(img_b.img);
               case ImageStreamIODataType_b::DataType::DOUBLE:
                 return convert_img<double>(img_b.img);
               // case ImageStreamIODataType_b::DataType::COMPLEX_FLOAT: return ;
               // case ImageStreamIODataType_b::DataType::COMPLEX_DOUBLE: return ;
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
          [](IMAGE_B &img_b, const std::string &name, const py::buffer &buffer,
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
                &(img_b.img), name.c_str(), buf.ndim(), dims, datatype, location,
                shared, NBsem, NBkw, imagetype);
            if (res == 0) {
              if (buf.dtype() == pybind11::dtype::of<uint8_t>()) {
                write<uint8_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int8_t>()) {
                write<int8_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<uint16_t>()) {
                write<uint16_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int16_t>()) {
                write<int16_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<uint32_t>()) {
                write<uint32_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int32_t>()) {
                write<int32_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<uint64_t>()) {
                write<uint64_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<int64_t>()) {
                write<int64_t>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<float>()) {
                write<float>(img_b.img, buffer);
              } else if (buf.dtype() == pybind11::dtype::of<double>()) {
                write<double>(img_b.img, buffer);
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
                imagetype[in]:  the type of the image to create (ImageStreamIOType_b).
            Return:
                ret      [out]: error code
            )pbdoc",
          py::arg("name"), py::arg("buffer"), py::arg("location") = -1,
          py::arg("shared") = 1, py::arg("NBsem") = IMAGE_NB_SEMAPHORE,
          py::arg("NBkw") = 1, py::arg("imagetype") = MATH_DATA)

      // .def(
      //     "create",
      //     [](IMAGE_B &img, std::string name, py::array_t<uint32_t> dims,
      //        uint8_t datatype, uint8_t shared, uint16_t NBkw) {
      //       // Request a buffer descriptor from Python
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
      //     py::arg("datatype") = ImageStreamIODataType_b::DataType::FLOAT,
      //     py::arg("shared") = 1, py::arg("NBkw") = 1)

      // .def(
      //     "create",
      //     [](IMAGE_B &img, std::string name, py::array_t<uint32_t> dims,
      //        uint8_t datatype, int8_t location, uint8_t shared, int NBsem,
      //        int NBkw, uint64_t imagetype) {
      //       // Request a buffer descriptor from Python
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
      //     py::arg("datatype") = ImageStreamIODataType_b::DataType::FLOAT,
      //     py::arg("location") = -1, py::arg("shared") = 1,
      //     py::arg("NBsem") = IMAGE_NB_SEMAPHORE, py::arg("NBkw") = 1,
      //     py::arg("imagetype") = MATH_DATA)

      .def(
          "open",
          [](IMAGE_B &img_b, std::string name) {
            return ImageStreamIO_openIm(&(img_b.img), name.c_str());
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
          [](IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_closeIm(&(img_b.img));
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
          [](IMAGE_B &img_b) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_destroyIm(&(img_b.img));
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
          [](IMAGE_B &img_b, long index) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_getsemwaitindex(&(img_b.img), index);
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
          [](IMAGE_B &img_b, long index) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_semwait(&(img_b.img), index);
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
          [](IMAGE_B &img_b, long index, float timeoutsec) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            struct timespec timeout;
            clock_gettime(CLOCK_REALTIME, &timeout);
            timeout.tv_nsec += (long)(timeoutsec * 1000000000L);
            timeout.tv_sec += timeout.tv_nsec / 1000000000L;
            timeout.tv_nsec = timeout.tv_nsec % 1000000000L;
            return ImageStreamIO_semtimedwait(&(img_b.img), index, &timeout);
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
          [](IMAGE_B &img_b, long index) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_sempost(&(img_b.img), index);
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
          [](IMAGE_B &img_b, long index) {
            if (img_b.img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_semflush(&(img_b.img), index);
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