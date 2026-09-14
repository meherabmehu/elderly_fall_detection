# ESP32 firmware

`live/main.cpp` reads MPU6050 data. `replay/main.cpp` replays a selected KFall trial. Copy the relevant `main.cpp`, `model_data.h`, `model_config.h`, and (for replay) `replay_data.h` into a PlatformIO project.

Pins: SDA 21, SCL 22, buzzer 4, LED 2. The MPU6050 address is 0x68.
