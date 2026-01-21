# Dataset Script Generation Summary

| Dataset | Seq Len | Variates | Type | Batch | d_model | Trigger Config (d_bd, ff_bd, heads) |
|---------|---------|----------|------|-------|---------|-------------------------------------|
| AbnormalHeartbeat | 18530 | 1 | long | 16 | 128 | d_bd=64, ff_bd=128, heads=2 |
| BasicMotions | 100 | 6 | short | 8 | 32 | d_bd=16, ff_bd=32, heads=2 |
| BirdChicken | 512 | 1 | long | 8 | 128 | d_bd=64, ff_bd=128, heads=2 |
| CharacterTrajectories | 119 | 3 | medium | 32 | 64 | d_bd=32, ff_bd=64, heads=2 |
| Cricket | 1197 | 6 | long | 16 | 128 | d_bd=64, ff_bd=128, heads=2 |
| ECG5000 | 140 | 1 | medium | 32 | 64 | d_bd=32, ff_bd=64, heads=2 |
| ElectricDevices | 96 | 1 | short | 32 | 32 | d_bd=16, ff_bd=32, heads=2 |
| Epilepsy | 206 | 3 | medium | 16 | 64 | d_bd=32, ff_bd=64, heads=2 |
| FaceDetection | 62 | 144 | short | 32 | 64 | d_bd=64, ff_bd=128, heads=8 |
| FingerMovements | 50 | 28 | short | 16 | 32 | d_bd=24, ff_bd=48, heads=4 |
| HandMovementDirection | 400 | 10 | medium | 16 | 64 | d_bd=32, ff_bd=64, heads=2 |
| Handwriting | 152 | 3 | medium | 16 | 64 | d_bd=32, ff_bd=64, heads=2 |
| Haptics | 1092 | 1 | long | 16 | 128 | d_bd=64, ff_bd=128, heads=2 |
| Heartbeat | 405 | 61 | medium | 16 | 128 | d_bd=64, ff_bd=128, heads=4 |
| JapaneseVowels | 26 | 12 | short | 16 | 32 | d_bd=24, ff_bd=48, heads=4 |
| LSST | 36 | 6 | short | 32 | 32 | d_bd=16, ff_bd=32, heads=2 |
| MotorImagery | 3000 | 64 | long | 16 | 256 | d_bd=128, ff_bd=256, heads=4 |
| NATOPS | 51 | 24 | short | 16 | 32 | d_bd=24, ff_bd=48, heads=4 |
| PEMS-SF | 144 | 963 | medium | 16 | 128 | d_bd=128, ff_bd=256, heads=8 |
| PenDigits | 8 | 2 | short | 32 | 32 | d_bd=16, ff_bd=32, heads=2 |
| PhonemeSpectra | 217 | 11 | medium | 32 | 64 | d_bd=48, ff_bd=96, heads=4 |
| SelfRegulationSCP1 | 896 | 6 | long | 16 | 128 | d_bd=64, ff_bd=128, heads=2 |
| SelfRegulationSCP2 | 1152 | 7 | long | 16 | 128 | d_bd=64, ff_bd=128, heads=2 |
| SharePriceIncrease | 60 | 1 | short | 32 | 32 | d_bd=16, ff_bd=32, heads=2 |
| Sleep | 178 | 1 | medium | 32 | 64 | d_bd=32, ff_bd=64, heads=2 |
| SpokenArabicDigits | 93 | 13 | short | 32 | 32 | d_bd=24, ff_bd=48, heads=4 |
| Strawberry | 235 | 1 | medium | 32 | 64 | d_bd=32, ff_bd=64, heads=2 |
| UWaveGestureLibrary | 315 | 3 | medium | 16 | 64 | d_bd=32, ff_bd=64, heads=2 |
| WalkingSittingStanding | 206 | 3 | medium | 32 | 64 | d_bd=32, ff_bd=64, heads=2 |
| Wine | 234 | 1 | medium | 8 | 64 | d_bd=32, ff_bd=64, heads=2 |

## Trigger Network Forecast Types
- **short**: seq_len <= 100 -> pred_len = seq_len/8
- **medium**: 100 < seq_len <= 500 -> pred_len = seq_len/4
- **long**: seq_len > 500 -> pred_len = min(seq_len/2, 256)
