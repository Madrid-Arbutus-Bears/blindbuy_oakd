#pragma once

#include <nlohmann/json.hpp>

namespace dai {
/**
 * On which processor the node will be placed
 *
 * Enum specifying processor
 */
enum class ProcessorType : int32_t { LOS, LRT };

}  // namespace dai