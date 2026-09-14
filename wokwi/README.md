# wokwi/ — simulation projects

Simulation verifies firmware+model **integration** (I²C, scheduling, INT8
inference, alarm logic). It is not physical validation — see
`docs/WOKWI.md` for the full guide and honest caveats.

| Project | Use |
|---|---|
| `final_model/` | **Recommended.** Final INT8 model (23,080 B) + per-window instance norm. `src/main.cpp` supports `REPLAY_MODE 1` (KFall S06T20R01, default) and `0` (live MPU6050). Builds verified with PlatformIO for both modes. Simulator execution of the final-model replay is pending — record the serial log when you run it |
| `legacy_synthetic_model/` | The August 2026 verified replay project, running the legacy synthetic-trained model. Provenance only |

Both projects target the ESP32 DevKit V1 with MPU6050 at 0x68, buzzer on
GPIO 4, LED on GPIO 2 (220 Ω) — identical to the physical build so demo
behaviour transfers.
