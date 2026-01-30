# GRITSBot Hardware Specifications

Complete technical reference for the GRITSBot robots used in Robotarium experiments.

## Source Repository

Full hardware designs available at: https://github.com/robotarium/GRITSBot_hardware_design

## Architecture

The GRITSBot consists of two stacked circuit boards:
1. **Main Board** - Processing, WiFi, power management
2. **Motor Board** - Locomotion, sensors, motor control

## Main Board

### Processor
- **Chip:** ESP8266 12-E
- **Clock:** 80/160 MHz
- **Memory:** ~80kB DRAM, ~35kB IRAM
- **Flash:** 4MB (32 MBit) - supports OTA updates

### Power Management
- MCP73831 LiPo charging IC
- AP2112K-3.3V regulator (600mA)
- MCP1640 step-up converter (150mA for motors)
- INA219 I2C current/voltage sensor

### Communication
- IEEE 802.11 B/G/N WiFi (built into ESP8266)
- ~150mA average current consumption

### Security
- ATECC108 encryption and authentication chip

## Motor Board

### Processor
- **Chip:** Atmega168/328
- **Clock:** 8MHz
- **Flash:** 16/32 KB
- **RAM:** 2 KB

### Locomotion
- Two LB1836M motor drivers
- Miniature stepper motors
- High-resolution positioning (up to 8 rotations/second)

### Sensors
- QRE1113 downward-facing IR line sensors (x2)
- STLM20 temperature sensors for motor monitoring (x2)
- ZXCT1009 current sensors (x2)
- 24AA025UIDT-I/OT 2KBIT I2C EEPROM (unique ID)

### Safety
- Reverse polarity protection on charger inputs
- LEDs for visual debugging

## Charging System

- Qi wireless charging receiver
- Autonomous docking with overhead camera guidance
- ~1 hour battery life at full activity

## Programming

### Main Board
- Serial bootloader via USB-to-TTL adapter (3.3V only!)
- OTA firmware updates supported

### Motor Board  
- SPI programming via ISP header
- Requires avr-gcc toolchain

## References

- Firmware: https://github.com/robotarium/GRITSBot_firmware
- Hardware: https://github.com/robotarium/GRITSBot_hardware_design
