#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/list.h>

#include <ctime>
#include <iostream>
#include <sstream>

#include "../ImageStreamIO.h"
#include "../ImageStruct.h"

namespace nb = nanobind;

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

  operator uint8_t() const { return datatype; }
};

const std::vector<uint8_t> ImageStreamIODataType::Size(
    {0, SIZEOF_DATATYPE_UINT8, SIZEOF_DATATYPE_INT8, SIZEOF_DATATYPE_UINT16,
     SIZEOF_DATATYPE_INT16, SIZEOF_DATATYPE_UINT32, SIZEOF_DATATYPE_INT32,
     SIZEOF_DATATYPE_UINT64, SIZEOF_DATATYPE_INT64, SIZEOF_DATATYPE_FLOAT,
     SIZEOF_DATATYPE_DOUBLE, SIZEOF_DATATYPE_COMPLEX_FLOAT,
     SIZEOF_DATATYPE_COMPLEX_DOUBLE, SIZEOF_DATATYPE_HALF});

// Helper: map ndarray dtype to ImageStreamIO datatype
template <typename... Args>
uint8_t NdarrayDtypeToImageStreamIODataType(const nb::ndarray<Args...> &arr) {
  auto dt = arr.dtype();
  if (dt == nb::dtype<uint8_t>()) return _DATATYPE_UINT8;
  if (dt == nb::dtype<int8_t>()) return _DATATYPE_INT8;
  if (dt == nb::dtype<uint16_t>()) return _DATATYPE_UINT16;
  if (dt == nb::dtype<int16_t>()) return _DATATYPE_INT16;
  if (dt == nb::dtype<uint32_t>()) return _DATATYPE_UINT32;
  if (dt == nb::dtype<int32_t>()) return _DATATYPE_INT32;
  if (dt == nb::dtype<uint64_t>()) return _DATATYPE_UINT64;
  if (dt == nb::dtype<int64_t>()) return _DATATYPE_INT64;
  if (dt == nb::dtype<float>()) return _DATATYPE_FLOAT;
  if (dt == nb::dtype<double>()) return _DATATYPE_DOUBLE;
  throw std::runtime_error(
      "NdarrayDtypeToImageStreamIODataType -- Not implemented datatype");
}

template <typename T>
nb::object convert_img(const IMAGE &img) {
  if (ImageStreamIO_typesize(img.md->datatype) != sizeof(T)) {
    throw std::runtime_error("IMAGE is not compatible with output format");
  }

  size_t nelement = img.md->nelement;
  T *data = new T[nelement];

  if (img.md->location == -1) {
    memcpy(data, img.array.raw, nelement * sizeof(T));
  } else {
#ifdef HAVE_CUDA
    cudaSetDevice(img.md->location);
    cudaMemcpy(data, img.array.raw, nelement * sizeof(T),
               cudaMemcpyDeviceToHost);
#else
    delete[] data;
    throw std::runtime_error(
        "unsupported location, CACAO needs to be compiled with -DUSE_CUDA=ON");
#endif
  }

  nb::capsule owner(data, [](void *p) noexcept { delete[] (T *)p; });

  std::vector<size_t> shape(img.md->naxis);
  for (int8_t axis = 0; axis < img.md->naxis; ++axis) {
    shape[axis] = img.md->size[axis];
  }

  return nb::cast(nb::ndarray<nb::numpy, T>(
      data, img.md->naxis, shape.data(), owner, nullptr, nb::dtype<T>(),
      nb::device::cpu::value, 0, 'F'));
}

template <typename T>
nb::object view_img(const IMAGE &img) {
  if (ImageStreamIO_typesize(img.md->datatype) != sizeof(T)) {
    throw std::runtime_error("IMAGE is not compatible with output format");
  }

  std::vector<size_t> shape(img.md->naxis);
  for (int8_t axis = 0; axis < img.md->naxis; ++axis) {
    shape[axis] = img.md->size[axis];
  }

  // No-op capsule: shared memory is externally managed
  nb::capsule owner((void *)img.array.raw, [](void *) noexcept {});

  return nb::cast(nb::ndarray<nb::numpy, T>(
      (T *)img.array.raw, img.md->naxis, shape.data(), owner, nullptr,
      nb::dtype<T>(), nb::device::cpu::value, 0, 'F'));
}

void write_img(IMAGE &img, nb::ndarray<nb::f_contig, nb::device::cpu> b) {
  if (img.array.raw == nullptr) {
    throw std::runtime_error("image not initialized");
  }

  uint8_t datatype = NdarrayDtypeToImageStreamIODataType(b);
  if (img.md->datatype != datatype) {
    throw std::invalid_argument("incompatible type");
  }
  if ((size_t)b.ndim() != (size_t)img.md->naxis) {
    throw std::invalid_argument("incompatible number of axis");
  }
  for (size_t i = 0; i < b.ndim(); ++i) {
    if (b.shape(i) != img.md->size[i]) {
      throw std::invalid_argument("incompatible shape");
    }
  }

  uint64_t size = img.md->nelement * ImageStreamIO_typesize(datatype);

  img.md->write = 1;  // set this flag to 1 when writing data

  void *current_image = img.array.raw;

  if (img.md->location == -1) {
    memcpy(current_image, b.data(), size);
  } else {
#ifdef HAVE_CUDA
    cudaSetDevice(img.md->location);
    cudaMemcpy(current_image, b.data(), size, cudaMemcpyHostToDevice);
#else
    throw std::runtime_error(
        "unsupported location, CACAO needs to be compiled with -DUSE_CUDA=ON");
#endif
  }
  ImageStreamIO_sempost(&img, -1);
  clock_gettime(CLOCK_ISIO, &img.md->lastaccesstime);
  img.md->write = 0;  // Done writing data
  img.md->cnt0++;
  img.md->cnt1++;
}

NB_MODULE(ImageStreamIOWrap, m) {
  m.doc() = "CACAO ImageStreamIO python module";

#ifdef COVERAGE_BUILD
  m.def("_gcov_dump", &_gcov_dump);
#endif

  auto imageDatatype =
      nb::class_<ImageStreamIODataType>(m, "ImageStreamIODataType")
          .def(nb::init<uint8_t>())
          .def_ro("size", &ImageStreamIODataType::asize)
          .def_ro("type", &ImageStreamIODataType::datatype)
          .def("__repr__", [](const ImageStreamIODataType &img_datatype) {
            std::ostringstream tmp_str;
            tmp_str << "datatype: " << img_datatype.datatype << std::endl;
            tmp_str << "size: " << img_datatype.asize << std::endl;
            return tmp_str.str();
          });

  nb::enum_<ImageStreamIODataType::DataType>(imageDatatype, "Type")
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
      nb::class_<ImageStreamIOType>(m, "ImageStreamIOType")
          .def(nb::init<uint8_t>())
          .def_prop_ro("axis", &ImageStreamIOType::get_axis)
          .def_prop_ro("type", &ImageStreamIOType::get_type)
          .def("__repr__", [](const ImageStreamIOType &image_type) {
            std::ostringstream tmp_str;
            tmp_str << "type: " << image_type.get_type() << std::endl;
            tmp_str << "axis: " << image_type.get_axis() << std::endl;
            return tmp_str.str();
          });

  nb::enum_<ImageStreamIOType::Type>(imagetype, "Type")
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
  nb::class_<IMAGE_KEYWORD>(m, "Image_kw")
      .def(nb::init<>())
      .def("__init__",
           [](IMAGE_KEYWORD *kw, std::string name, int64_t numl,
              std::string comment) {
             if (name.size() > KEYWORD_MAX_STRING) {
               throw std::invalid_argument("name too long");
             }
             if (comment.size() > KEYWORD_MAX_COMMENT) {
               throw std::invalid_argument("comment too long");
             }
             new (kw) IMAGE_KEYWORD();
             std::copy(name.begin(), name.end(), kw->name);
             kw->type = 'L';
             kw->value.numl = numl;
             std::copy(comment.begin(), comment.end(), kw->comment);
           },
           nb::arg("name"), nb::arg("numl"), nb::arg("comment") = "")
      .def("__init__",
           [](IMAGE_KEYWORD *kw, std::string name, double numf,
              std::string comment) {
             if (name.size() > KEYWORD_MAX_STRING) {
               throw std::invalid_argument("name too long");
             }
             if (comment.size() > KEYWORD_MAX_COMMENT) {
               throw std::invalid_argument("comment too long");
             }
             new (kw) IMAGE_KEYWORD();
             std::copy(name.begin(), name.end(), kw->name);
             kw->type = 'D';
             kw->value.numf = numf;
             std::copy(comment.begin(), comment.end(), kw->comment);
           },
           nb::arg("name"), nb::arg("numf"), nb::arg("comment") = "")
      .def("__init__",
           [](IMAGE_KEYWORD *kw, std::string name, std::string valstr,
              std::string comment) {
             if (name.size() > KEYWORD_MAX_STRING) {
               throw std::invalid_argument("name too long");
             }
             if (valstr.size() > KEYWORD_MAX_STRING) {
               throw std::invalid_argument("valstr too long");
             }
             if (comment.size() > KEYWORD_MAX_COMMENT) {
               throw std::invalid_argument("comment too long");
             }
             new (kw) IMAGE_KEYWORD();
             std::copy(name.begin(), name.end(), kw->name);
             kw->type = 'S';
             std::copy(valstr.begin(), valstr.end(), kw->value.valstr);
             std::copy(comment.begin(), comment.end(), kw->comment);
           },
           nb::arg("name"), nb::arg("valstr"), nb::arg("comment") = "")
      .def_ro("name", &IMAGE_KEYWORD::name)
      .def_ro("type", &IMAGE_KEYWORD::type)
      .def_prop_ro("value",
                             [](const IMAGE_KEYWORD &kw) -> nb::object {
                               switch (kw.type) {
                                 case 'L':
                                   return nb::int_(kw.value.numl);
                                 case 'D':
                                   return nb::float_(kw.value.numf);
                                 case 'S':
                                   return nb::str(kw.value.valstr);
                                 default:
                                   throw std::runtime_error("Unknown format");
                               }
                             })
      .def("__str__", [](const IMAGE_KEYWORD &kw) { return toString(kw); })
      .def("__repr__", [](const IMAGE_KEYWORD &kw) { return toString(kw); })
      .def_ro("comment", &IMAGE_KEYWORD::comment);

  // STREAM_PROC_TRACE interface
  nb::class_<STREAM_PROC_TRACE>(m, "Proc_trace")
      .def_ro("triggermode", &STREAM_PROC_TRACE::triggermode)
      .def_ro("pid_write", &STREAM_PROC_TRACE::procwrite_PID)
      .def_ro("trigger_inode", &STREAM_PROC_TRACE::trigger_inode)
      .def_ro("ts_procstart", &STREAM_PROC_TRACE::ts_procstart)
      .def_ro("ts_streamupdate", &STREAM_PROC_TRACE::ts_streamupdate)
      .def_ro("trigger_semindex", &STREAM_PROC_TRACE::trigsemindex)
      .def_ro("trigger_status", &STREAM_PROC_TRACE::triggerstatus)
      .def_ro("cnt0", &STREAM_PROC_TRACE::cnt0);

  // IMAGE_METADATA interface
  nb::class_<IMAGE_METADATA>(m, "Image_md")
      // .def(nb::init([]() {
      //     return std::unique_ptr<IMAGE_METADATA>(new IMAGE_METADATA());
      // }))
      .def_ro("version", &IMAGE_METADATA::version)
      .def_ro("name", &IMAGE_METADATA::name)
      .def_ro("naxis", &IMAGE_METADATA::naxis)
      .def_prop_ro("size",
                             [](const IMAGE_METADATA &md) {
                               std::vector<uint32_t> dims(md.naxis);
                               const uint32_t *ptr = md.size;
                               for (auto &&dim : dims) {
                                 dim = *ptr;
                                 ++ptr;
                               }
                               return dims;
                             })
      .def_ro("nelement", &IMAGE_METADATA::nelement)
      .def_prop_ro(
          "datatype",
          [](const IMAGE_METADATA &md) {
            return ImageStreamIODataType(md.datatype).datatype;
          })
      .def_prop_ro(
          "imagetype",
          [](const IMAGE_METADATA &md) {
            return ImageStreamIOType(md.imagetype).get_type();
          })
      .def_prop_ro(
          "creationtime",
          [](const IMAGE_METADATA &md) {
            return (double)md.creationtime.tv_sec +
                   (double)md.creationtime.tv_nsec * 1e-9;
          })
      .def_prop_ro(
          "lastaccesstime",
          [](const IMAGE_METADATA &md) {
            return (double)md.lastaccesstime.tv_sec +
                   (double)md.lastaccesstime.tv_nsec * 1e-9;
          })
      .def_prop_ro(
          "acqtime",
          [](const IMAGE_METADATA &md) {
            return (double)md.atime.tv_sec +
                   (double)md.atime.tv_nsec * 1e-9;
          })
      .def_prop_ro(
          "writetime",
          [](const IMAGE_METADATA &md) {
            return (double)md.writetime.tv_sec +
                   (double)md.writetime.tv_nsec * 1e-9;
          })
      .def_ro("shared", &IMAGE_METADATA::shared)
      .def_ro("location", &IMAGE_METADATA::location)
      .def_prop_ro("location_str",
                             [](const IMAGE_METADATA &md) {
                               if (md.location < 0) {
                                 return std::string("CPU RAM");
                               }

                               std::ostringstream tmp_str;
                               tmp_str << "GPU" << int(md.location) << " RAM";
                               return tmp_str.str();
                             })
      .def_ro("status", &IMAGE_METADATA::status)
      .def_ro("inode", &IMAGE_METADATA::inode)
      .def_ro("logflag", &IMAGE_METADATA::logflag)
      .def_ro("sem", &IMAGE_METADATA::sem)
      .def_ro("cnt0", &IMAGE_METADATA::cnt0)
      .def_ro("cnt1", &IMAGE_METADATA::cnt1)
      .def_ro("cnt2", &IMAGE_METADATA::cnt2)
      .def_ro("write", &IMAGE_METADATA::write)
      .def_ro("flag", &IMAGE_METADATA::flag)
      .def_ro("NBkw", &IMAGE_METADATA::NBkw)
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
          std::time_t t = (std::time_t)md.creationtime.tv_sec;
          tmp_str << "creationtime: " << std::ctime(&t);
        }
        {
          std::time_t t = (std::time_t)md.lastaccesstime.tv_sec;
          tmp_str << "lastaccesstime: " << std::ctime(&t);
        }
        {
          std::time_t t = (std::time_t)md.atime.tv_sec;
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
  nb::class_<IMAGE>(m, "Image")
      .def(nb::init<>())
      .def_ro("used", &IMAGE::used)
      .def_ro("memsize", &IMAGE::memsize)
      .def_ro("md", &IMAGE::md)
      .def_ro("streamproctrace0", &IMAGE::streamproctrace)
      .def_prop_ro(
          "shape",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::vector<uint32_t> dims(img.md->naxis);
            const uint32_t *ptr = img.md->size;
            for (int i{}; i < img.md->naxis; ++i) {
              dims[i] = ptr[i];
            }
            return dims;
          })

      .def_prop_ro(
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
      .def_prop_ro(
          "acqtimearray",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img.atimearray == NULL) {
              throw std::runtime_error("acqtimearray not initialized");
            }
            std::vector<double> acqtimearray(img.md->size[2]);
            for (int i = 0; i < img.md->size[2]; ++i) {
              acqtimearray[i] = (double)img.atimearray[i].tv_sec +
                                (double)img.atimearray[i].tv_nsec * 1e-9;
            }
            return acqtimearray;
          })
      .def_prop_ro(
          "writetimearray",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img.writetimearray == NULL) {
              throw std::runtime_error("writetimearray not initialized");
            }
            std::vector<double> writetimearray(img.md->size[2]);
            for (int i = 0; i < img.md->size[2]; ++i) {
              writetimearray[i] = (double)img.writetimearray[i].tv_sec +
                                  (double)img.writetimearray[i].tv_nsec * 1e-9;
            }
            return writetimearray;
          })
      .def_prop_ro(
          "cntarray",
          [](const IMAGE &img) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            if (img.cntarray == NULL) {
              throw std::runtime_error("cntarray not initialized");
            }
            std::vector<uint64_t> cntarray(img.md->size[2]);
            for (int i = 0; i < img.md->size[2]; ++i) {
              cntarray[i] = img.cntarray[i];
            }
            return cntarray;
          })
      // TODO: fix flagarray never allocated and cause segfaults
      // .def_prop_ro(
      //     "flagarray",
      //     [](const IMAGE &img) {
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
      .def_prop_ro(
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
      .def("get_kws_list",
           [](const IMAGE &img) {
             if (img.array.raw == nullptr) {
               throw std::runtime_error("image not initialized");
             }
             std::list<IMAGE_KEYWORD> keywords;
             for (int i = 0; i < img.md->NBkw; ++i) {
               if (strcmp(img.kw[i].name, "") == 0) {
                 break;
               }
               keywords.push_back(img.kw[i]);
             }
             return keywords;
           })

      .def("set_kws",
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

      .def("set_kws_list",
          [](const IMAGE &img, std::list<IMAGE_KEYWORD> &keywords) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            std::list<IMAGE_KEYWORD>::iterator it =
                keywords.begin();
            int cnt = 0;
            while (it != keywords.end()) {
              if (cnt >= img.md->NBkw)
                throw std::runtime_error("Too many keywords provided");
              img.kw[cnt] = *it;
              it++;
              cnt++;
            }
            // Pad with empty keywords
            if (cnt < img.md->NBkw) {
              img.kw[cnt] = IMAGE_KEYWORD();
            }
          })

      .def("copy",
           [](const IMAGE &img) -> nb::object {
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

      .def("view",
           [](const IMAGE &img) -> nb::object {
             if (img.array.raw == nullptr)
               throw std::runtime_error("image not initialized");
             if (img.md->location >= 0)
               throw std::runtime_error(
                   "Cannot create a zero-copy view of a GPU buffer");
             ImageStreamIODataType dt(img.md->datatype);
             switch (dt.datatype) {
               case ImageStreamIODataType::DataType::UINT8:
                 return view_img<uint8_t>(img);
               case ImageStreamIODataType::DataType::INT8:
                 return view_img<int8_t>(img);
               case ImageStreamIODataType::DataType::UINT16:
                 return view_img<uint16_t>(img);
               case ImageStreamIODataType::DataType::INT16:
                 return view_img<int16_t>(img);
               case ImageStreamIODataType::DataType::UINT32:
                 return view_img<uint32_t>(img);
               case ImageStreamIODataType::DataType::INT32:
                 return view_img<int32_t>(img);
               case ImageStreamIODataType::DataType::UINT64:
                 return view_img<uint64_t>(img);
               case ImageStreamIODataType::DataType::INT64:
                 return view_img<int64_t>(img);
               case ImageStreamIODataType::DataType::FLOAT:
                 return view_img<float>(img);
               case ImageStreamIODataType::DataType::DOUBLE:
                 return view_img<double>(img);
               default:
                 throw std::runtime_error("Not implemented");
             }
           })

      .def("update", [](IMAGE &img) {
        if (img.array.raw == nullptr)
            throw std::runtime_error("image not initialized");
        return ImageStreamIO_UpdateIm(&img);
      })

      .def("update_atime", [](IMAGE &img, double timestamp) {
        if (img.array.raw == nullptr)
            throw std::runtime_error("image not initialized");
        struct timespec atime;
        atime.tv_sec = (time_t)timestamp;
        atime.tv_nsec = (long)((timestamp - (double)atime.tv_sec) * 1e9);
        return ImageStreamIO_UpdateIm_atime(&img, &atime);
      })

      .def("write", &write_img,
           R"pbdoc(
          Write into memory image stream
          Parameters:
            buffer [in]:  buffer to put into memory image stream
          )pbdoc",
           nb::arg("buffer"))

      .def(
          "create",
          [](IMAGE &img, const std::string &name,
             nb::ndarray<nb::f_contig, nb::device::cpu> buffer,
             int8_t location, uint8_t shared, int NBsem, int NBkw,
             uint64_t imagetype, uint32_t CBsize) {
            uint8_t datatype = NdarrayDtypeToImageStreamIODataType(buffer);

            uint32_t dims[buffer.ndim()];
            for (size_t i = 0; i < buffer.ndim(); ++i) {
              dims[i] = buffer.shape(i);
            }

            int res = ImageStreamIO_createIm_gpu(
                &img, name.c_str(), buffer.ndim(), dims, datatype, location,
                shared, NBsem, NBkw, imagetype, CBsize);
            if (res == 0) {
              write_img(img, buffer);
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
                CBsize   [in]:  fast circular buffer size
            Return:
                ret      [out]: error code
            )pbdoc",
          nb::arg("name"), nb::arg("buffer"), nb::arg("location") = -1,
          nb::arg("shared") = 1, nb::arg("NBsem") = IMAGE_NB_SEMAPHORE,
          nb::arg("NBkw") = 1, nb::arg("imagetype") = MATH_DATA,
          nb::arg("CBsize") = 0)

      // .def(
      //     "create",
      //     [](IMAGE &img, std::string name, py::array_t<uint32_t> dims,
      //        uint8_t datatype, uint8_t shared, uint16_t NBkw) {
      //       /* Request a buffer descriptor from Python */
      //       py::buffer_info info = dims.request();

      //       // uint8_t datatype =
      //       // PyFormatToImageStreamIODataType(info);
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
      //     nb::arg("name"), nb::arg("dims"),
      //     nb::arg("datatype") = ImageStreamIODataType::DataType::FLOAT,
      //     nb::arg("shared") = 1, nb::arg("NBkw") = 1)

      // .def(
      //     "create",
      //     [](IMAGE &img, std::string name, py::array_t<uint32_t> dims,
      //        uint8_t datatype, int8_t location, uint8_t shared, int NBsem,
      //        int NBkw, uint64_t imagetype) {
      //       /* Request a buffer descriptor from Python */
      //       py::buffer_info info = dims.request();

      //       // uint8_t datatype =
      //       // PyFormatToImageStreamIODataType(info);
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
      //     nb::arg("name"), nb::arg("dims"),
      //     nb::arg("datatype") = ImageStreamIODataType::DataType::FLOAT,
      //     nb::arg("location") = -1, nb::arg("shared") = 1,
      //     nb::arg("NBsem") = IMAGE_NB_SEMAPHORE, nb::arg("NBkw") = 1,
      //     nb::arg("imagetype") = MATH_DATA)

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
          nb::arg("name"))

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
          nb::arg("index"))

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
          nb::arg("index"))

      .def(
          "semtimedwait",
          [](IMAGE &img, long index, float timeoutsec) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            struct timespec timeout;
            clock_gettime(CLOCK_ISIO, &timeout);
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
          nb::arg("index"), nb::arg("timeoutsec"))

      .def(
          "semtrywait",
          [](IMAGE &img, long index) {
            if (img.array.raw == nullptr) {
              throw std::runtime_error("image not initialized");
            }
            return ImageStreamIO_semtrywait(&img, index);
          },
          R"pbdoc(
                Check the semaphore value in non-blocking mode
                Parameters:
                    index  [in]:  index of semaphore to wait
                Return:
                    ret    [out]: error code
                )pbdoc",
          nb::arg("index"))

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
          nb::arg("index") = -1)

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
          nb::arg("index"))

      .def(
        "semvalue",
        [](IMAGE &img, long index) {
          if (img.array.raw == nullptr) {
            throw std::runtime_error("image not initialized");
          }
          return ImageStreamIO_semvalue(&img, index);
        },
        R"pbdoc(
              Flush shmim semaphore
              Parameters:
                  index  [in]:  index of semaphore to flush; flush all semaphores if index<0
              Return:
                  ret    [out]: error code
              )pbdoc",
        nb::arg("index"));
}
