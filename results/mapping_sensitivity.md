# Mapping robustness — strict vs expanded label→genre mapping

Re-scored from cached score vectors (no re-inference). **Expanded** adds `Halftime`→dnb and `Grime`→garage_dubstep (Discogs taxonomies only; MTG-Jamendo's 87 tags contain no equivalent, so it is unchanged).

| model | overall (strict→expanded) | dnb (strict→expanded) | garage_dubstep (strict→expanded) |
|---|--:|--:|--:|
| discogs400-effnet | 44.0 → **44.5** | 32.0 → **33.0** | 17.0 → **19.0** |
| discogs400-maest-5s-pw | 43.8 → **43.5** | 25.0 | 22.0 → **23.0** |
| discogs400-maest-10s-pw | 43.2 → **42.8** | 21.0 → **18.0** | 24.0 → **25.0** |
| discogs400-maest-10s-fs | 37.5 | 18.0 | 22.0 |
| discogs400-maest-10s-dw | 45.5 → **46.0** | 29.0 | 26.0 → **28.0** |
| discogs400-maest-20s-pw | 39.8 → **40.0** | 17.0 | 27.0 → **29.0** |
| discogs400-maest-30s-pw | 34.2 → **34.5** | 11.0 | 19.0 → **20.0** |
| discogs400-maest-30s-pw-ts | 37.8 → **38.0** | 13.0 | 21.0 → **22.0** |
| discogs519-maest-30s-pw | 41.5 → **41.8** | 13.0 | 20.0 → **21.0** |
| jamendo-effnet | 47.0 | 11.0 | 14.0 |
| jamendo-artist | 56.0 | 34.0 | 34.0 |
| jamendo-label | 54.8 | 27.0 | 30.0 |
| jamendo-multi | 53.2 | 25.0 | 29.0 |
| jamendo-release | 54.2 | 28.0 | 17.0 |
| jamendo-track | 56.0 | 25.0 | 26.0 |

## Labels added by the expanded mapping

- **discogs400**: {'dnb': ['Electronic---Halftime'], 'garage_dubstep': ['Electronic---Grime', 'Hip Hop---Grime']}
- **discogs519**: {'dnb': ['Electronic---Halftime'], 'garage_dubstep': ['Electronic---Grime', 'Hip Hop---Grime']}
- **mtg_jamendo**: nothing added
