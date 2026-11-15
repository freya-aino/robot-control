# cmu-mocap

The CMU Graphics Lab Motion Capture Database as a `data-source/1`
record, `code_inspected`: the terms were audited from the database's own
front page on 2026-11-15, and one boxing trial (subject 13, trial 17 —
4,840 frames at 120 Hz, 330 channels, 99.93 % valid) was fetched and
round-tripped through `robot data import`'s C3D path, stored
content-addressed as the record's pinned `content_digest`.

**Terms, quoted from mocap.cs.cmu.edu (2026-11-15):** "This dataset of
motions is free for all uses." / "This data is free for use in research
projects. You may include this data in commercially-sold products, but
you may not resell this data directly, even in converted form." The
requested acknowledgment is: "The data used in this project was obtained
from mocap.cs.cmu.edu. The database was created with funding from NSF
EIA-0196217." The site asks not to be crawled; files are fetched
individually, never in bulk, and never vendored into this repository.

Boxing-adjacent trials found by the database's own search: subjects 13
(13_17, 13_18), 14 (14_01..14_03), 15 (15_04, 15_05, 15_13), 17 (17_10),
79 (79_08) and 80 (80_10).
