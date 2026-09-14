# hardware/ — wiring, BOM, assembly documents

The text guide with the bring-up sequence is `docs/HARDWARE.md`; this folder
holds the visual and tabular source material.

| File | Content |
|---|---|
| `esp32_fall_detector_pin_diagram.svg` | schematic pin diagram (ESP32 ↔ MPU6050, buzzer, LED) |
| `wiring_diagram.png` | wiring diagram used in the technical package |
| `esp32_mpu6050_breadboard_setup.png` | photo-style breadboard overview |
| `realistic_esp32_wiring_annotated.png` | annotated realistic wiring figure (used in reports/presentation) |
| `realistic_schematic_wiring.png` | schematic-style wiring figure |
| `esp32_fall_detector_minimum_bom.docx` | budget sheet with Dhaka estimated prices |
| `complete_pin_connections.docx` | full 15-row connection sheet + breadboard rails + power options + final checklist |
| `usb_direct_pin_connections.docx` | USB-powered-build connection sheet (the verified build) |

Single source of truth for pins: SDA 21, SCL 22, buzzer 4, LED 2 (220 Ω),
AD0 → GND (0x68), MPU6050 on 3V3 — **never 5 V**, all grounds common.
