#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../lib/ska_sort.hpp"
#include <string>

namespace py = pybind11;

// Template para ordenamiento directo con ska_sort
template<typename T>
void sort_numeric_array(py::array_t<T> input) {
    py::buffer_info buf = input.request();
    T* ptr = static_cast<T*>(buf.ptr);
    size_t size = buf.size;
    
    if (size <= 1) return;
    ska_sort(ptr, ptr + size);
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
    
    std::vector<std::string> strings;
    strings.reserve(num_items);
    
    // Convertir datos a strings
    for (size_t i = 0; i < num_items; ++i) {
        char* str_ptr = ptr + i * item_size;
        size_t str_len = strnlen(str_ptr, item_size);
        strings.emplace_back(str_ptr, str_len);
    }
    
    // Usar ska_sort directamente en el vector de strings
    ska_sort(strings.begin(), strings.end());
    
    // Copiar los strings ordenados de vuelta al array original
    for (size_t i = 0; i < num_items; ++i) {
        char* dest = ptr + i * item_size;
        std::memset(dest, 0, item_size);
        size_t copy_len = std::min(strings[i].size(), item_size - 1);
        std::memcpy(dest, strings[i].c_str(), copy_len);
        dest[copy_len] = '\0';
    }
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
            std::string error_msg = "Unsupported data type for ska_sort: ";
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
    return "ska_sort_pure";
}

std::string get_version() {
    return "2.5.0";
}

// Definición del módulo Python
PYBIND11_MODULE(ska_sort_py, m) {
    m.doc() = "SKA Sort Python Binding - Pure ska_sort for ultra-fast radix sorting";
    
    m.def("sort", &sort, "Sort numpy array in-place using pure ska_sort",
          py::arg("array"));
    
    m.def("supported_types", &get_supported_types, 
          "Get list of supported numpy data types");
    
    m.attr("__version__") = get_version();
    m.attr("optimization") = get_optimization_info();
    m.attr("supported_types") = get_supported_types();
    
    // Información adicional
    m.attr("algorithm") = "ska_sort";
    m.attr("fallback") = "none";
    m.attr("description") = "Pure ska_sort algorithm optimized for radix-sortable data types";
}
