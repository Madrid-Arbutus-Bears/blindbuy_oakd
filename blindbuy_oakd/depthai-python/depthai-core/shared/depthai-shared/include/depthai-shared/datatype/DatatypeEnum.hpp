#pragma once

#include <cstdint>

namespace dai {

enum class DatatypeEnum : std::int32_t {
    Buffer,
    ImgFrame,
    NNData,
    ImageManipConfig,
    CameraControl,
    ImgDetections,
    SpatialImgDetections,
    SystemInformation,
    SpatialLocationCalculatorConfig,
    SpatialLocationCalculatorData
};
bool isDatatypeSubclassOf(DatatypeEnum parent, DatatypeEnum children);

}  // namespace dai
