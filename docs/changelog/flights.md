# Ball-flight detection: changelog

`possession/features/flight.py`, run by `scripts/detect_flights.py`.

## Nov 9: fixes, and the kinematic part made a labeller input

The Nov 7 run on BAR-ATH gave 799 flights with peak ground speeds up to 341 km/h (131 above
130 km/h) and flights up to 11 s long. Causes and fixes:

| # | problem | fix |
|---|---|---|
| 1 | read the raw ATD ball: one-frame false blobs (1,071 on BAR-ATH) look like 50 m/s kicks | source = the prepared ball (`outputs/ball/ball_reconstructed.npz` from `scripts/reconstruct_ball.py`, or `prepare_ball` in memory) |
| 2 | linear bridge of 12 frames on top, across re-detections | removed for a prepared ball; the prepared ball fills only gaps a 35 m/s ball could cross |
| 3 | speed = one-frame difference of a 5-frame smoothed track | Savitzky-Golay derivative over 7 frames |
| 4 | start = first frame above 12 m/s, a few frames after the kick | walked back to the speed minimum (<= 5 frames) |
| 5 | nothing ended a flight received without a sharp deceleration | reception exit (ball within 1.5 m of a player, slower than 8 m/s) and a 4 s cap |
| 6 | a jump faster than any kick between two frames could start / end a flight | runs split at steps > 40 m/s |
| 7 | height fit padded the last observation's camera and pixel, not the end frame's | end frame's own homography and position |
| 8 | one focal length for the match; the camera zooms (IQR 0.18), P vs H off by ~100 px | focal per flight from its own homographies; `camera_gap_px` stored; airborne only if <= 10 px |

Result on BAR-ATH: 810 flights (35,380 frames), median 1.48 s, max 4.04 s; 5 airborne. Height is
still mostly unobservable (the camera derived from the homography matches it within 10 px on only
about a third of flights), so `in_air` stays a weak hint. The kinematic part, `flight_mask`, needs
no homography: it runs on the bench and is what labeller v5 uses (docs/changelog/labeller.md).

Outputs: `outputs/flights/ball_flights.csv` (+ `focal_px`, `camera_gap_px`), `in_air.npy`,
`in_flight.npy`.
