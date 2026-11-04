## Журнал запусков

| Дата | Коммит | Режим | Seed | Эпохи | Лучшая val accuracy, % | Время обучения, мин | Процент компенсации |
|---|---|---|---:|---:|---:|---:|---:|
| 03.11.2026 11:59 | — | `without_replacement` | 43 | 20 | 59,40 | 31,11 | — |
| 03.11.2026 11:59 | — | `adaptive` | 43 | 20 | 63,90 | 25,21 | 25% |
| 03.11.2026 14:22 | — | `without_replacement` | 44 | 20 | 61,65 | — | — |
| 03.11.2026 14:22 | — | `adaptive` | 44 | 20 | 56,85 | — | 25% |
| 03.11.2026 15:23 | — | `without_replacement` | 42 | 20 | 59,80 | 34,02 | — |
| 03.11.2026 15:23 | — | `adaptive` | 42 | 20 | 60,45 | 26,92 | 25% |
| 03.11.2026 16:09 | — | `without_replacement` | 42 | 30 | 63,85 | 48,10 | — |
| 03.11.2026 16:09 | — | `adaptive` | 42 | 30 | 64,55 | 38,68 | 25% |
| 04.11.2026 06:10 | — | `without_replacement` | — | 30 | 61,05 | 46,95 | — |
| 04.11.2026 06:10 | — | `adaptive` | — | 30 | 59,90 | 40,30 | 25% |


These ALSO runs used Learning Rate decay. That may have been an issue. I will be implementing LR Decay into my own algo and sampler but now will be switching ALSO back into constant LR.

| Дата | Коммит | Режим | Seed | Эпохи | Лучшая val accuracy, % | Время обучения, мин | Процент компенсации |
|---|---|---|---:|---:|---:|---:|---:|
