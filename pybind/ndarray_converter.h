#ifndef __NDARRAY_CONVERTER_H__
#define __NDARRAY_CONVERTER_H__

#include <Python.h>

#include <opencv2/core/core.hpp>

class NDArrayConverter
{
public:
  // must call this first, or the other routines don't work!
  static bool init_numpy();

  static bool toMat(PyObject *o, cv::Mat &m);
  static PyObject *toNDArray(const cv::Mat &mat);
};

//
// Define the type converter
//

#include <pybind11/pybind11.h>

namespace pybind11
{
  namespace detail
  {
    template <>
    struct type_caster<cv::Mat>
    {
    public:
      PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

      bool load(handle src, bool /* convert */)
      {
        return NDArrayConverter::toMat(src.ptr(), value);
      }

      static handle cast(const cv::Mat &m, return_value_policy, handle defval)
      {
        return handle(NDArrayConverter::toNDArray(m));
      }
    };

  } // namespace detail
} // namespace pybind11

#endif