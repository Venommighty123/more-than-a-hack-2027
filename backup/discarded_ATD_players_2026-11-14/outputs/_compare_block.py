# -- five-panel comparison: ATD vs our YOLO detections -------------------------------------------

CELL_W, CELL_H = (2 * PANEL_W) // 3, (2 * PANEL_H) // 3      # 640 x 360: three cells = timeline width
REF_BGR, GK_BGR = (0, 230, 255), (255, 255, 255)


def load_yolo_window(path, a, b, team_mode, window=12):
    """Our detections for frames a..b: the slot layout v6 reads (top view) and the raw boxes (video).

    Slots come from possession/io/detections.py (same packing and team rule as
    `scripts/run.py --players detected`), so the top view shows exactly what the labeller was given.
    Boxes are read from the parquet with the same team rule computed on the window (+/- window).
    """
    import pandas as pd
    from possession.io.detections import _local_team, load_detected

    det = load_detected(path, n_frames=N_FRAMES, team_mode=team_mode, window=window)
    L, W = vt.PITCH_L, vt.PITCH_W
    slots = np.stack([det["px"][a:b + 1] / L, det["py"][a:b + 1] / W], axis=2)    # normalised
    rows = pd.read_parquet(path, filters=[("frame", ">=", a - window), ("frame", "<=", b + window)],
                           columns=["frame", "track_id", "role", "team", "x1", "y1", "x2", "y2",
                                    "x_m", "y_m", "team_p_home"])
    rows["vote"] = rows["team"].map({"H": 0, "A": 1})
    if team_mode == "local":
        rows, p = _local_team(rows, window)
        rows = rows.assign(tm=np.where(np.isfinite(p), (p < 0.5).astype(float), rows["vote"].to_numpy()))
    else:
        rows = rows.assign(tm=rows["vote"])
    rows = rows[(rows.frame >= a) & (rows.frame <= b)].sort_values("frame")
    starts = np.searchsorted(rows.frame.to_numpy(), np.arange(a, b + 2))
    return slots, det["team"], rows.reset_index(drop=True), starts, det["stats"]


def draw_atd_video(img, Hk, players_k, team, ball_k, scale, project):
    ok = ~np.isnan(players_k[:, 0]) & ~np.isnan(players_k[:, 1])
    if ok.any():
        u, v = project(Hk, players_k[ok, 0], players_k[ok, 1])
        for uu, vv, tm in zip(np.atleast_1d(u), np.atleast_1d(v), team[ok]):
            if np.isfinite(uu) and np.isfinite(vv):
                c = (int(uu * scale), int(vv * scale))
                cv2.circle(img, c, 6, vt.HOME_BGR if tm == 0 else vt.AWAY_BGR, -1, cv2.LINE_AA)
                cv2.circle(img, c, 6, (0, 0, 0), 1, cv2.LINE_AA)
    if not np.isnan(ball_k).any():
        u, v = project(Hk, ball_k[0], ball_k[1])
        if np.isfinite(u) and np.isfinite(v):
            cv2.circle(img, (int(u * scale), int(v * scale)), 8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_yolo_video(img, rows, scale, show_ids):
    for r in rows.itertuples():
        if r.role == "referee":
            col = REF_BGR
        elif np.isfinite(r.tm):
            col = vt.HOME_BGR if r.tm == 0 else vt.AWAY_BGR
        else:
            col = (160, 160, 160)
        x1, y1, x2, y2 = r.x1 * scale, r.y1 * scale, r.x2 * scale, r.y2 * scale
        cx, w = int((x1 + x2) / 2), max(int((x2 - x1) / 2), 5)
        cv2.ellipse(img, (cx, int(y2)), (w, max(w // 3, 2)), 0, -45, 235, col, 2, cv2.LINE_AA)
        if r.role == "goalkeeper":
            cv2.ellipse(img, (cx, int(y2)), (w + 4, max(w // 3, 2) + 3), 0, -45, 235, GK_BGR, 1, cv2.LINE_AA)
        if show_ids:
            vt.put(img, str(int(r.track_id) % 1_000_000), (int(x1), int(y1) - 3), 0.35, col)


def label_frame(img, lab, names, title):
    col = LABEL_BGR[lab]
    cv2.rectangle(img, (BORDER // 2, BORDER // 2), (img.shape[1] - 1 - BORDER // 2, img.shape[0] - 1 - BORDER // 2),
                  col, BORDER)
    badge = "{}   {}  {}".format(title, lab, team_word(lab, names))
    (tw, th), _ = cv2.getTextSize(badge, vt.FONT, 0.8, 2)
    cv2.rectangle(img, (BORDER + 6, BORDER + 6), (BORDER + 24 + tw, BORDER + 20 + th), col, -1)
    vt.put(img, badge, (BORDER + 14, BORDER + 13 + th), 0.8, (255, 255, 255), 2)


def caption(img, text):
    cv2.rectangle(img, (0, img.shape[0] - 34), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    vt.put(img, text, (14, img.shape[0] - 10), 0.75, (255, 255, 255), 2)
    return img


def cell(img):
    return cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)


def render_compare(args, labels, labels2, a, b, names, team, raw_path, out_path):
    """One mp4, five synced panels + an info cell + timeline:

        original video | video + ATD players | video + YOLO players
        top view ATD   | top view YOLO        | info (labels, counts, nearest-to-ball)
    """
    n = b - a + 1
    print("\nCOMPARE window {}..{} ({} frames, {} to {}) -> {}".format(
        a, b, n, vt.fmt_time(a), vt.fmt_time(b + 1), out_path))
    players, ball = vt.read_tracking(raw_path, a, n, len(team))
    slots, yteam, rows, starts, st = load_yolo_window(args.players_file, a, b, args.team_mode)
    print("  YOLO: {} slots (H {}, A {}), team mode {}, {} boxes in the window".format(
        st["slots"], st["slots_home"], st["slots_away"], st["team_mode"], len(rows)))
    from starter.load_homography import project
    H, H_valid = vt.load_homography_window(a, n)
    reader, src_w = vt.open_video(args, a, n)
    scale = PANEL_W / float(src_w)

    blocks = half_video_blocks(GAME)
    second = labels2[a:b + 1] if labels2 is not None else None
    timeline, x_of, mx = build_timeline(labels, second, a, b, blocks[0][1], blocks[-1][2])
    if second is not None:                                   # the GT band holds labels2 here
        timeline[32:46, :90] = (25, 25, 25)
        vt.put(timeline, "LABELS 2", (6, 44), 0.4)
    panel = vt.PitchPanel(PANEL_W, PANEL_H, flip_y=args.flip_y)
    size = (3 * CELL_W, 2 * CELL_H + TL_H)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        sys.exit("could not open the video writer for {}".format(out_path))
    name1 = os.path.splitext(os.path.basename(args.labels))[0]
    name2 = os.path.splitext(os.path.basename(args.labels2))[0] if labels2 is not None else None
    drawn, t = [], time.time()
    try:
        for k in range(n):
            f = a + k
            img = reader.read()
            if img is None:
                print("\n  video ended at frame {}".format(f))
                break
            lab = str(labels[f])
            lab2 = str(labels2[f]) if labels2 is not None else None
            hv = k < len(H_valid) and H_valid[k]

            raw = img.copy()
            atd = img.copy()
            if hv:
                draw_atd_video(atd, H[k], players[k], team, ball[k], scale, project)
            yolo = img.copy()
            yr = rows.iloc[starts[k]:starts[k + 1]]
            draw_yolo_video(yolo, yr, scale, args.ids)
            label_frame(atd, lab, names, "ATD run" if labels2 is None else name1)
            if lab2 is not None:
                label_frame(yolo, lab2, names, name2)

            trail = ball[max(0, k - args.trail):k + 1] if args.trail > 0 else ball[k:k + 1]
            top_atd, ah, aa, anear = panel.render(players[k], ball[k], team, trail, names[0], names[1], args.ids)
            top_yolo, yh, ya, ynear = panel.render(slots[k], ball[k], yteam, trail, names[0], names[1], args.ids)
            for r in yr.itertuples():                           # referees: not in the slots v6 reads
                if r.role == "referee" and np.isfinite(r.x_m):
                    px_, py_ = panel.norm(r.x_m / vt.PITCH_L, r.y_m / vt.PITCH_W)
                    cv2.rectangle(top_yolo, (int(px_) - 5, int(py_) - 5), (int(px_) + 5, int(py_) + 5), REF_BGR, -1)
            for top, lb in ((top_atd, lab), (top_yolo, lab2 if lab2 is not None else lab)):
                if lb == "D":
                    top[:] = cv2.addWeighted(top, 0.6, np.full_like(top, 110), 0.4, 0)

            n_ref = int((yr.role == "referee").sum())
            n_gk = int((yr.role == "goalkeeper").sum())
            info = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            info[:] = (30, 30, 30)
            half = half_of_video_frame(GAME, f)
            lines = [
                ("frame {}   t {}   half {}".format(f, vt.fmt_time(f), half or "-"), 0.8, (255, 255, 255)),
                ("{}: {}  {}".format(name1, lab, team_word(lab, names)), 0.85, LABEL_BGR[lab]),
            ]
            if lab2 is not None:
                lines.append(("{}: {}  {}".format(name2, lab2, team_word(lab2, names)), 0.85, LABEL_BGR[lab2]))
                lines.append(("labels AGREE" if lab == lab2 else "labels DIFFER", 0.85,
                              (120, 255, 120) if lab == lab2 else (110, 110, 255)))
            lines += [
                ("ATD   players H {}  A {}   nearest-to-ball {}".format(ah, aa, {0: "H", 1: "A", -1: "-"}[anear]),
                 0.7, (230, 230, 230)),
                ("YOLO  players H {}  A {}   nearest-to-ball {}".format(yh, ya, {0: "H", 1: "A", -1: "-"}[ynear]),
                 0.7, (230, 230, 230)),
                ("YOLO  referees {}   goalkeepers {}".format(n_ref, n_gk), 0.7, (230, 230, 230)),
                ("ball {}  (ATD ball on both top views)".format(
                    "tracked" if not np.isnan(ball[k, 0]) else "MISSING"), 0.7,
                 (255, 255, 255) if not np.isnan(ball[k, 0]) else (120, 120, 255)),
                ("over 11 on a team:  ATD {}   YOLO {}".format(
                    "YES" if max(ah, aa) > 11 else "no", "YES" if max(yh, ya) > 11 else "no"), 0.7, (230, 230, 230)),
                ("ellipse = YOLO box, yellow = referee, white ring = GK", 0.6, (170, 170, 170)),
            ]
            for j, (text, sc, col) in enumerate(lines):
                vt.put(info, text[:70], (24, 50 + 52 * j), sc, col, 2 if sc >= 0.8 else 1)

            canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            row1 = [caption(raw, "1  original"), caption(atd, "2  ATD players"), caption(yolo, "3  YOLO players")]
            row2 = [caption(top_atd, "4  top view ATD"), caption(top_yolo, "5  top view YOLO (what v6 reads)"), info]
            for i, im in enumerate(row1):
                canvas[:CELL_H, i * CELL_W:(i + 1) * CELL_W] = cell(im)
            for i, im in enumerate(row2):
                canvas[CELL_H:2 * CELL_H, i * CELL_W:(i + 1) * CELL_W] = cell(im)
            tl = timeline.copy()
            x = x_of(f)
            cv2.line(tl, (x, 0), (x, TL_H - 26), (255, 255, 255), 2)
            xm = mx(f)
            cv2.line(tl, (xm, TL_H - 22), (xm, TL_H - 4), (255, 255, 255), 1)
            canvas[2 * CELL_H:, :] = tl[:, :size[0]]
            writer.write(canvas)
            drawn.append((f, lab))
            if len(drawn) % 25 == 0 or k == n - 1:
                progress("rendering", len(drawn) - 1, n, t)
    finally:
        writer.release()
        reader.close()
    print()
    return drawn


