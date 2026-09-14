# legacy/wokwi/ — the August 2026 verified replay project (LEGACY)

This Wokwi/PlatformIO project runs the **legacy synthetic-trained model**
(`model/model_data.h`, 19,928 B) with frozen global normalisation and replays
KFall `S06T20R01` (`src/replay_data.h`) through the INT8 model in the
simulator. *This is the project that produced the verified Wokwi
buzzer/LED demo in August 2026.*

It is kept as the historical working reference. For anything current use
`wokwi/final_model/` (final model, instance normalisation).

Note the include layout in this snapshot (`src/main.cpp` including headers
from a project-root `model/` folder) may require adding the project root to
the compiler's include path in modern PlatformIO versions; the current
`wokwi/final_model/` keeps all headers under `src/` instead.
