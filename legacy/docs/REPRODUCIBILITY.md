# Reproducibility checklist

- Working rate: 50 Hz
- Input: 100 samples × 6 channels
- Stride: 25 samples
- Channels: ax, ay, az, gx, gy, gz
- Classes: non-fall, alert, fall
- Model: separable CNN, 8,219 parameters
- INT8 model: `models/model.tflite`
- Frozen constants: `models/frozen_normalization.json`, `firmware/model_config.h`
- Hardware model: ESP32 DevKit V1 + MPU6050
- Wokwi replay: KFall S06T20R01

The supplied SisFall archive is pre-windowed and lacks original subject IDs; strict SisFall LOSO is therefore not claimed. Wokwi verifies firmware/model integration, not physical battery life or measured hardware latency.
