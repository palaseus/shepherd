/**
 * @file memory_allocator.cpp
 * @brief Memory allocator implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "memory/memory_allocator.h"
#include <stdexcept>

namespace edge_ai {

MemoryAllocator::MemoryAllocator(std::shared_ptr<Device> device)
    : device_(device) {
}

} // namespace edge_ai
