# Essentia off-the-shelf genre classification — loops & one-shots

4 genres (house / techno / dnb / garage_dubstep), 50 loops + 50 one-shots each = 400 clips. Ground truth = source sample pack. **Restricted 4-way argmax** (over only the labels mapped to the 4 genres); random = **25%**.

## Restricted accuracy — overall, loop vs one-shot

| model | family | backbone | overall | loop | one-shot | Δ(loop−1shot) |
|---|---|---|--:|--:|--:|--:|
| discogs400-effnet | Discogs400 | effnet | **44.0** | 51.5 | 36.5 | 15.0 |
| discogs400-maest-5s-pw | Discogs400 | maest | **43.8** | 56.0 | 31.5 | 24.5 |
| discogs400-maest-10s-pw | Discogs400 | maest | **43.2** | 58.0 | 28.5 | 29.5 |
| discogs400-maest-10s-fs | Discogs400 | maest | **37.5** | 50.0 | 25.0 | 25.0 |
| discogs400-maest-10s-dw | Discogs400 | maest | **45.5** | 59.0 | 32.0 | 27.0 |
| discogs400-maest-20s-pw | Discogs400 | maest | **39.8** | 54.5 | 25.0 | 29.5 |
| discogs400-maest-30s-pw | Discogs400 | maest | **34.2** | 45.0 | 23.5 | 21.5 |
| discogs400-maest-30s-pw-ts | Discogs400 | maest | **37.8** | 50.5 | 25.0 | 25.5 |
| discogs519-maest-30s-pw | Discogs519 | maest | **41.5** | 50.0 | 33.0 | 17.0 |
| jamendo-effnet | MTG-Jamendo | effnet | **47.0** | 56.0 | 38.0 | 18.0 |
| jamendo-artist | MTG-Jamendo | effnet | **56.0** | 68.0 | 44.0 | 24.0 |
| jamendo-label | MTG-Jamendo | effnet | **54.8** | 67.0 | 42.5 | 24.5 |
| jamendo-multi | MTG-Jamendo | effnet | **53.2** | 67.0 | 39.5 | 27.5 |
| jamendo-release | MTG-Jamendo | effnet | **54.2** | 65.5 | 43.0 | 22.5 |
| jamendo-track | MTG-Jamendo | effnet | **56.0** | 69.5 | 42.5 | 27.0 |

## Restricted accuracy per genre (recall)

| model | house | techno | dnb | garage_dubstep |
|---|--:|--:|--:|--:|
| discogs400-effnet | 61.0 | 66.0 | 32.0 | 17.0 |
| discogs400-maest-5s-pw | 51.0 | 77.0 | 25.0 | 22.0 |
| discogs400-maest-10s-pw | 51.0 | 77.0 | 21.0 | 24.0 |
| discogs400-maest-10s-fs | 31.0 | 79.0 | 18.0 | 22.0 |
| discogs400-maest-10s-dw | 59.0 | 68.0 | 29.0 | 26.0 |
| discogs400-maest-20s-pw | 60.0 | 55.0 | 17.0 | 27.0 |
| discogs400-maest-30s-pw | 16.0 | 91.0 | 11.0 | 19.0 |
| discogs400-maest-30s-pw-ts | 52.0 | 65.0 | 13.0 | 21.0 |
| discogs519-maest-30s-pw | 63.0 | 70.0 | 13.0 | 20.0 |
| jamendo-effnet | 84.0 | 79.0 | 11.0 | 14.0 |
| jamendo-artist | 75.0 | 81.0 | 34.0 | 34.0 |
| jamendo-label | 78.0 | 84.0 | 27.0 | 30.0 |
| jamendo-multi | 78.0 | 81.0 | 25.0 | 29.0 |
| jamendo-release | 91.0 | 81.0 | 28.0 | 17.0 |
| jamendo-track | 91.0 | 82.0 | 25.0 | 26.0 |

## Unrestricted top-1 (full taxonomy)

*top-1 acc = model's single highest label maps to the right genre; on-genre% = how often the #1 label is any of the 4 genres at all.*

| model | top-1 overall | loop | one-shot | on-genre % |
|---|--:|--:|--:|--:|
| discogs400-effnet | 24.2 | 29.5 | 19.0 | 38.5 |
| discogs400-maest-5s-pw | 32.5 | 41.0 | 24.0 | 59.2 |
| discogs400-maest-10s-pw | 28.7 | 41.5 | 16.0 | 48.5 |
| discogs400-maest-10s-fs | 20.0 | 31.0 | 9.0 | 40.5 |
| discogs400-maest-10s-dw | 32.8 | 44.5 | 21.0 | 56.0 |
| discogs400-maest-20s-pw | 26.5 | 41.0 | 12.0 | 46.0 |
| discogs400-maest-30s-pw | 18.5 | 29.5 | 7.5 | 38.5 |
| discogs400-maest-30s-pw-ts | 15.2 | 28.5 | 2.0 | 30.5 |
| discogs519-maest-30s-pw | 22.8 | 33.0 | 12.5 | 37.0 |
| jamendo-effnet | 2.0 | 3.5 | 0.5 | 2.0 |
| jamendo-artist | 2.2 | 2.0 | 2.5 | 2.8 |
| jamendo-label | 3.5 | 6.5 | 0.5 | 4.0 |
| jamendo-multi | 1.8 | 2.5 | 1.0 | 1.8 |
| jamendo-release | 5.0 | 8.0 | 2.0 | 5.0 |
| jamendo-track | 7.8 | 10.0 | 5.5 | 7.8 |
