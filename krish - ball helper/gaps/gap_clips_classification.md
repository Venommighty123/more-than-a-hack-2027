# Gap Clips Classification: Set-Piece / Restart Analysis

This document categorizes all **135 rendered video clips** in `gap_clips/` based on whether a **Set Piece**, **Restart**, **Goal**, **Match Boundary**, or **Possession Handover** occurred after the ball tracking gap, as the ball came back into play.

- **Total Gap Clips Analyzed**: 135
- **Gaps WITH Set-Piece / Restart / State Change**: 93
- **Gaps WITHOUT Set-Piece / State Change (In-Play Occlusions)**: 42

---

## 1. Gap Clips WITH Set Piece, Restart, or State Change (93 Clips)

| Gap Clip Filename | ATD Frame Range | Gap Length (Frames / Sec) | Restart Event / Change Details |
|---|---|---|---|
| `gap_0010985_00152.mp4` | 11036..11187 | 152 (6.1s) | Set Piece: Kick Off (Start), Match Boundary: Pre-kickoff |
| `gap_0017410_00366.mp4` | 17461..17826 | 366 (14.6s) | Set Piece: Free Kick [Atletic Club], Possession Handover |
| `gap_0018979_00743.mp4` | 19030..19772 | 743 (29.7s) | Set Piece: Throw-in, Goal Kick [Barcelona], Possession Handover |
| `gap_0019933_00135.mp4` | 19984..20118 | 135 (5.4s) | Set Piece: Throw-in |
| `gap_0023607_00152.mp4` | 23658..23809 | 152 (6.1s) | Possession Handover (Atletic Club/Barcelona -> Atletic Club) |
| `gap_0023781_00266.mp4` | 23832..24097 | 266 (10.6s) | Possession Handover (Atletic Club -> Atletic Club/Barcelona) |
| `gap_0027590_00163.mp4` | 27641..27803 | 163 (6.5s) | Set Piece: Throw-in |
| `gap_0028980_00751.mp4` | 29031..29781 | 751 (30.0s) | Set Piece: Corner Kick [Barcelona] |
| `gap_0029879_00319.mp4` | 29930..30248 | 319 (12.8s) | Set Piece: Corner Kick [Barcelona] |
| `gap_0034150_00627.mp4` | 34201..34827 | 627 (25.1s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0035138_00220.mp4` | 35189..35408 | 220 (8.8s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0036214_00788.mp4` | 36265..37052 | 788 (31.5s) | Set Piece: Corner Kick [Atletic Club] |
| `gap_0042400_00298.mp4` | 42451..42748 | 298 (11.9s) | Set Piece: Corner Kick [Atletic Club] |
| `gap_0042991_00136.mp4` | 43042..43177 | 136 (5.4s) | Set Piece: Corner Kick [Atletic Club], Possession Handover |
| `gap_0044488_00201.mp4` | 44539..44739 | 201 (8.0s) | Set Piece: Throw-in |
| `gap_0045404_00743.mp4` | 45455..46197 | 743 (29.7s) | Set Piece: Free Kick [Barcelona], Possession Handover |
| `gap_0046628_00640.mp4` | 46679..47318 | 640 (25.6s) | Set Piece: Free Kick [Barcelona], Possession Handover |
| `gap_0047722_00259.mp4` | 47773..48031 | 259 (10.4s) | Set Piece: Free Kick [Barcelona] |
| `gap_0048028_00522.mp4` | 48079..48600 | 522 (20.9s) | Set Piece: Free Kick [Barcelona] |
| `gap_0048590_00293.mp4` | 48641..48933 | 293 (11.7s) | Set Piece: Free Kick [Barcelona] |
| `gap_0048948_00288.mp4` | 48999..49286 | 288 (11.5s) | Set Piece: Free Kick [Barcelona] |
| `gap_0049448_00215.mp4` | 49499..49713 | 215 (8.6s) | Set Piece: Kick Off (After Goal) |
| `gap_0050224_00722.mp4` | 50275..50996 | 722 (28.9s) | Set Piece: Goal Kick [Barcelona], Possession Handover |
| `gap_0051459_00137.mp4` | 51510..51646 | 137 (5.5s) | Set Piece: Goal Kick [Barcelona], Possession Handover |
| `gap_0052170_00611.mp4` | 52221..52831 | 611 (24.4s) | Set Piece: Free Kick [Barcelona] |
| `gap_0055464_00149.mp4` | 55515..55663 | 149 (6.0s) | Possession Handover (Atletic Club/Barcelona -> Barcelona) |
| `gap_0056971_00512.mp4` | 57022..57533 | 512 (20.5s) | Set Piece: Corner Kick [Atletic Club], Possession Handover |
| `gap_0058031_03466.mp4` | 58082..61547 | 3466 (138.6s) | Set Piece: Corner Kick [Atletic Club] |
| `gap_0061711_00573.mp4` | 61762..62334 | 573 (22.9s) | Set Piece: Kick Off (After Goal) |
| `gap_0065233_00443.mp4` | 65284..65726 | 443 (17.7s) | Set Piece: Throw-in, Possession Handover |
| `gap_0065685_00094.mp4` | 65736..65829 | 94 (3.8s) | Set Piece: Throw-in |
| `gap_0068295_00311.mp4` | 68346..68656 | 311 (12.4s) | Set Piece: Throw-in, Possession Handover |
| `gap_0068748_00183.mp4` | 68799..68981 | 183 (7.3s) | Set Piece: Throw-in |
| `gap_0069066_00289.mp4` | 69117..69405 | 289 (11.6s) | Set Piece: Throw-in |
| `gap_0072510_01391.mp4` | 72561..73951 | 1391 (55.6s) | Set Piece: Free Kick [Atletic Club] |
| `gap_0073947_01535.mp4` | 73998..75532 | 1535 (61.4s) | Set Piece: Kick Off (After Goal), Free Kick [Atletic Club] |
| `gap_0079841_00132.mp4` | 79892..80023 | 132 (5.3s) | Possession Handover |
| `gap_0080001_01124.mp4` | 80052..81175 | 1124 (45.0s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0082198_00558.mp4` | 82249..82806 | 558 (22.3s) | Possession Handover (Atletic Club/Barcelona -> Atletic Club) |
| `gap_0083478_00164.mp4` | 83529..83692 | 164 (6.6s) | Possession Handover |
| `gap_0084067_00729.mp4` | 84118..84846 | 729 (29.2s) | Set Piece: Goal Kick [Atletic Club], Possession Handover |
| `gap_0085151_00544.mp4` | 85202..85745 | 544 (21.8s) | Set Piece: Throw-in, Goal Kick [Atletic Club] |
| `gap_0086137_00207.mp4` | 86188..86394 | 207 (8.3s) | Set Piece: Throw-in |
| `gap_0090172_00787.mp4` | 90223..91009 | 787 (31.5s) | Match Boundary: Halftime Break |
| `gap_0113261_00275.mp4` | 113312..113586 | 275 (11.0s) | Set Piece: Kick Off (Start 2nd Half) |
| `gap_0115584_00350.mp4` | 115635..115984 | 350 (14.0s) | Set Piece: Throw-in |
| `gap_0120415_00242.mp4` | 120466..120707 | 242 (9.7s) | Set Piece: Throw-in |
| `gap_0121855_00190.mp4` | 121906..122095 | 190 (7.6s) | Set Piece: Throw-in |
| `gap_0123319_00380.mp4` | 123370..123749 | 380 (15.2s) | Set Piece: Throw-in |
| `gap_0124406_00135.mp4` | 124457..124591 | 135 (5.4s) | Set Piece: Throw-in |
| `gap_0126823_00509.mp4` | 126874..127382 | 509 (20.4s) | Set Piece: Throw-in |
| `gap_0129643_00779.mp4` | 129694..130472 | 779 (31.2s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0130548_00529.mp4` | 130599..131127 | 529 (21.2s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0133882_00521.mp4` | 133933..134453 | 521 (20.8s) | Possession Handover |
| `gap_0135574_01055.mp4` | 135625..136679 | 1055 (42.2s) | Set Piece: Corner Kick [Atletic Club] |
| `gap_0137275_00203.mp4` | 137326..137528 | 203 (8.1s) | Set Piece: Corner Kick [Atletic Club] |
| `gap_0137599_00606.mp4` | 137650..138255 | 606 (24.2s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0138378_00485.mp4` | 138429..138913 | 485 (19.4s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0138918_00287.mp4` | 138969..139255 | 287 (11.5s) | Set Piece: Corner Kick [Barcelona] |
| `gap_0140037_00129.mp4` | 140088..140216 | 129 (5.2s) | Set Piece: Goal Kick [Atletic Club] |
| `gap_0142565_00462.mp4` | 142616..143077 | 462 (18.5s) | Possession Handover |
| `gap_0144346_00176.mp4` | 144397..144572 | 176 (7.0s) | Set Piece: Goal Kick [Barcelona] |
| `gap_0146560_00264.mp4` | 146611..146874 | 264 (10.6s) | Set Piece: Corner Kick [Barcelona], Possession Handover |
| `gap_0147166_00146.mp4` | 147217..147362 | 146 (5.8s) | Set Piece: Corner Kick [Barcelona] |
| `gap_0155217_00504.mp4` | 155268..155771 | 504 (20.2s) | Set Piece: Corner Kick [Atletic Club], Free Kick [Atletic Club] |
| `gap_0156717_00240.mp4` | 156768..157007 | 240 (9.6s) | Set Piece: Throw-in, Possession Handover |
| `gap_0157496_01104.mp4` | 157547..158650 | 1104 (44.2s) | Set Piece: Throw-in, Possession Handover |
| `gap_0158691_00780.mp4` | 158742..159521 | 780 (31.2s) | Set Piece: Set Piece Phase [Barcelona] |
| `gap_0159476_01082.mp4` | 159527..160608 | 1082 (43.3s) | Set Piece: Set Piece Phase [Barcelona] |
| `gap_0160797_00087.mp4` | 160848..160934 | 87 (3.5s) | Set Piece: Set Piece Phase [Barcelona] |
| `gap_0161275_01758.mp4` | 161326..163083 | 1758 (70.3s) | Set Piece: Kick Off (After Goal) |
| `gap_0163814_00101.mp4` | 163865..163965 | 101 (4.0s) | Set Piece: Goal Kick [Barcelona] |
| `gap_0164489_00289.mp4` | 164540..164828 | 289 (11.6s) | Set Piece: Throw-in, Goal Kick [Barcelona] |
| `gap_0165393_00122.mp4` | 165444..165565 | 122 (4.9s) | Set Piece: Throw-in |
| `gap_0168868_01260.mp4` | 168919..170178 | 1260 (50.4s) | Set Piece: Throw-in |
| `gap_0170154_00200.mp4` | 170205..170404 | 200 (8.0s) | Set Piece: Throw-in |
| `gap_0170401_00170.mp4` | 170452..170621 | 170 (6.8s) | Set Piece: Throw-in |
| `gap_0171309_00604.mp4` | 171360..171963 | 604 (24.2s) | Possession Handover |
| `gap_0173321_00511.mp4` | 173372..173882 | 511 (20.4s) | Possession Handover |
| `gap_0177198_00625.mp4` | 177249..177873 | 625 (25.0s) | Possession Handover |
| `gap_0177934_00194.mp4` | 177985..178178 | 194 (7.8s) | Possession Handover |
| `gap_0179168_00239.mp4` | 179219..179457 | 239 (9.6s) | Set Piece: Throw-in, Possession Handover |
| `gap_0179649_00205.mp4` | 179700..179904 | 205 (8.2s) | Set Piece: Throw-in |
| `gap_0180029_00177.mp4` | 180080..180256 | 177 (7.1s) | Set Piece: Throw-in |
| `gap_0181927_00639.mp4` | 181978..182616 | 639 (25.6s) | Set Piece: Goal Kick [Atletic Club] |
| `gap_0182673_00292.mp4` | 182724..183015 | 292 (11.7s) | Set Piece: Throw-in, Goal Kick [Atletic Club] |
| `gap_0183723_00260.mp4` | 183774..184033 | 260 (10.4s) | Set Piece: Throw-in |
| `gap_0184112_00153.mp4` | 184163..184315 | 153 (6.1s) | Set Piece: Throw-in |
| `gap_0184968_00407.mp4` | 185019..185425 | 407 (16.3s) | Set Piece: Throw-in, Goal Kick [Barcelona] |
| `gap_0187077_00137.mp4` | 187128..187264 | 137 (5.5s) | Possession Handover |
| `gap_0188382_00177.mp4` | 188433..188609 | 177 (7.1s) | Set Piece: Throw-in |
| `gap_0189293_00577.mp4` | 189344..189920 | 577 (23.1s) | Set Piece: Throw-in |
| `gap_0190026_00417.mp4` | 190077..190493 | 417 (16.7s) | Match Boundary: Post-match (Full Time) |

---

## 2. Gap Files WITHOUT Set Piece or State Change (42 Clips)

These 42 clips represent **in-play occlusions** where the ball tracking was temporarily lost (e.g., ball high in the air, camera cutaway mid-action, player occlusion in crowded area) but **no set piece, restart, goal, or possession change occurred** when tracking resumed:

1. `gap_0012781_00148.mp4`
2. `gap_0016596_00132.mp4`
3. `gap_0016997_00232.mp4`
4. `gap_0026624_00297.mp4`
5. `gap_0033413_00432.mp4`
6. `gap_0040394_00234.mp4`
7. `gap_0056680_00169.mp4`
8. `gap_0063620_00144.mp4`
9. `gap_0063985_00148.mp4`
10. `gap_0064475_00146.mp4`
11. `gap_0069576_00269.mp4`
12. `gap_0070146_01324.mp4`
13. `gap_0071511_00603.mp4`
14. `gap_0072218_00268.mp4`
15. `gap_0076402_00406.mp4`
16. `gap_0077767_00166.mp4`
17. `gap_0079243_00222.mp4`
18. `gap_0086565_01737.mp4`
19. `gap_0088647_00273.mp4`
20. `gap_0089059_00929.mp4`
21. `gap_0112140_00265.mp4`
22. `gap_0118505_00146.mp4`
23. `gap_0118709_00414.mp4`
24. `gap_0119291_00140.mp4`
25. `gap_0119539_00223.mp4`
26. `gap_0122622_00376.mp4`
27. `gap_0124763_00187.mp4`
28. `gap_0129362_00169.mp4`
29. `gap_0135229_00169.mp4`
30. `gap_0139314_00404.mp4`
31. `gap_0143820_00183.mp4`
32. `gap_0146126_00385.mp4`
33. `gap_0163148_00289.mp4`
34. `gap_0167649_00134.mp4`
35. `gap_0168065_00134.mp4`
36. `gap_0168270_00268.mp4`
37. `gap_0172212_00502.mp4`
38. `gap_0175334_00194.mp4`
39. `gap_0175665_01278.mp4`
40. `gap_0178823_00159.mp4`
41. `gap_0181133_00320.mp4`
42. `gap_0181675_00152.mp4`
