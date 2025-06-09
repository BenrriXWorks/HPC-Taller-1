#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../lib/vergesort.h"
#include "../lib/ska_sort.hpp"
#include <string>
#include <array>
#include <functional>

namespace py = pybind11;

// Template para ordenamiento directo con vergesort
template<typename T>
void sort_numeric_array(py::array_t<T> input) {
    py::buffer_info buf = input.request();
    T* ptr = static_cast<T*>(buf.ptr);
    size_t size = buf.size;
    
    if (size <= 1) return;
    vergesort::vergesort(ptr, ptr + size);
}

// Ordenamiento especializado para strings
void sort_string_array(py::array input) {
    py::buffer_info buf = input.request();
    
    if (buf.size <= 1) return;
    
    if (input.dtype().kind() != 'S') {
        throw std::runtime_error("Only byte string arrays (dtype='S') are supported");
    }
    
    char* ptr = static_cast<char*>(buf.ptr);
    size_t item_size = buf.itemsize;
    size_t num_items = static_cast<size_t>(buf.size);
    
    std::vector<std::pair<char*, size_t>> string_pairs;
    string_pairs.reserve(num_items);
    
    for (size_t i = 0; i < num_items; ++i) {
        string_pairs.emplace_back(ptr + i * item_size, i);
    }
    
    auto string_compare = [](const std::pair<char*, size_t>& a, const std::pair<char*, size_t>& b) {
        return std::strcmp(a.first, b.first) < 0;
    };
    
    vergesort::vergesort(string_pairs.begin(), string_pairs.end(), string_compare);
    
    std::vector<char> temp_data(item_size * num_items);
    for (size_t i = 0; i < num_items; ++i) {
        std::memcpy(temp_data.data() + i * item_size, string_pairs[i].first, item_size);
    }
    std::memcpy(ptr, temp_data.data(), item_size * num_items);
}

// Función principal simplificada con switch directo
void sort(py::array input) {
    if (input.size() <= 1) return;
    
    if (input.ndim() != 1) {
        throw std::runtime_error("Only 1D arrays are supported");
    }
    
    // Switch directo sobre el tipo NumPy - más eficiente y compacto
    switch (input.dtype().num()) {
        case py::detail::npy_api::NPY_BOOL_:
            return sort_numeric_array<bool>(input.cast<py::array_t<bool>>());
        case py::detail::npy_api::NPY_BYTE_:
            return sort_numeric_array<int8_t>(input.cast<py::array_t<int8_t>>());
        case py::detail::npy_api::NPY_UBYTE_:
            return sort_numeric_array<uint8_t>(input.cast<py::array_t<uint8_t>>());
        case py::detail::npy_api::NPY_SHORT_:
            return sort_numeric_array<int16_t>(input.cast<py::array_t<int16_t>>());
        case py::detail::npy_api::NPY_USHORT_:
            return sort_numeric_array<uint16_t>(input.cast<py::array_t<uint16_t>>());
        case py::detail::npy_api::NPY_INT_:
            return sort_numeric_array<int32_t>(input.cast<py::array_t<int32_t>>());
        case py::detail::npy_api::NPY_UINT_:
            return sort_numeric_array<uint32_t>(input.cast<py::array_t<uint32_t>>());
        case py::detail::npy_api::NPY_LONG_:
        case py::detail::npy_api::NPY_LONGLONG_:
            return sort_numeric_array<int64_t>(input.cast<py::array_t<int64_t>>());
        case py::detail::npy_api::NPY_ULONG_:
        case py::detail::npy_api::NPY_ULONGLONG_:
            return sort_numeric_array<uint64_t>(input.cast<py::array_t<uint64_t>>());
        case py::detail::npy_api::NPY_FLOAT_:
            return sort_numeric_array<float>(input.cast<py::array_t<float>>());
        case py::detail::npy_api::NPY_DOUBLE_:
            return sort_numeric_array<double>(input.cast<py::array_t<double>>());
        case py::detail::npy_api::NPY_STRING_:
            return sort_string_array(input);
        default: {
            std::string error_msg = "Unsupported data type for vergesort: ";
            error_msg += py::str(input.dtype()).cast<std::string>();
            throw std::runtime_error(error_msg);
        }
    }
}

// Información del módulo
std::vector<std::string> get_supported_types() {
    return {
        "bool", "int8", "uint8", "int16", "uint16", 
        "int32", "uint32", "int64", "uint64", 
        "float32", "float64", "string"
    };
}

std::string get_optimization_info() {
    return "vergesort_pure";
}

std::string get_version() {
    return "1.0.0";
}

// Definición del módulo Python
PYBIND11_MODULE(vergesort_py, m) {
    m.doc() = "Vergesort Python Binding - Pure vergesort for adaptive high-performance sorting";
    
    m.def("sort", &sort, "Sort numpy array in-place using pure vergesort",
          py::arg("array"));
    
    m.def("supported_types", &get_supported_types, 
          "Get list of supported numpy data types");
    
    m.attr("__version__") = get_version();
    m.attr("optimization") = get_optimization_info();
    m.attr("supported_types") = get_supported_types();
    
    // Información adicional
    m.attr("algorithm") = "vergesort";
    m.attr("fallback") = "none";
    m.attr("description") = "Pure vergesort algorithm optimized for data with existing sorted runs";
}