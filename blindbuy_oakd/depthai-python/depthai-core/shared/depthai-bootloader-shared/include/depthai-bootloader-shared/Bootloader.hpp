#pragma once

#include <cstdint>

namespace dai
{
namespace bootloader
{

namespace request {

    enum Command {
        USB_ROM_BOOT = 0,
        BOOT_APPLICATION, 
        UPDATE_FLASH,
        GET_BOOTLOADER_VERSION
    };

    // Specific request PODs
    struct UsbRomBoot {
        // Common
        Command cmd;
        UsbRomBoot() : cmd(USB_ROM_BOOT) {}
    };
    struct BootApplication {
        // Common
        Command cmd;
        BootApplication() : cmd(BOOT_APPLICATION) {}
    
        // Data
    };

    struct UpdateFlash {
        // Common
        Command cmd;
        UpdateFlash() : cmd(UPDATE_FLASH) {}

        // Data
        enum Storage : uint32_t { SBR, BOOTLOADER };
        Storage storage;
        uint32_t totalSize;
        uint32_t numPackets;
    };

    struct GetBootloaderVersion {
        // Common
        Command cmd;
        GetBootloaderVersion() : cmd(GET_BOOTLOADER_VERSION) {}
    
        // Data
    };

}


namespace response {

    enum Command : uint32_t {
        FLASH_COMPLETE = 0, 
        FLASH_STATUS_UPDATE,
        BOOTLOADER_VERSION
    };

    // Specific request PODs
    struct FlashComplete{
        // Common
        Command cmd;
        FlashComplete() : cmd(FLASH_COMPLETE) {}

        // Data
        uint32_t success;
        char errorMsg[64];
    };
    struct FlashStatusUpdate {
        // Common
        Command cmd;
        FlashStatusUpdate() : cmd(FLASH_STATUS_UPDATE) {}

        // Data
        float progress;
    };
    struct BootloaderVersion {
        // Common
        Command cmd;
        BootloaderVersion() : cmd(BOOTLOADER_VERSION) {}
        
        // Data
        uint32_t major, minor, patch;
    };

}

} // namespace bootloader
} // namespace dai


