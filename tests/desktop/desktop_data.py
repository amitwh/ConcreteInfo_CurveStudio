"""Shared data snippets for desktop tests (a plain module: test dirs have no packages)."""

GOOD_CSV = "x,y\n1,2.0\n2,4.1\n3,6.2\n4,7.9\n5,10.3\n"
"""Five clean rows: enough for a fit, no issues."""

TEXT_COLUMN_CSV = "x,y,z\n1,2.0,foo\n2,abc,bar\n3,5.9,baz\n4,8.1,qux\n5,9.9,quux\n"
"""Text where numbers belong in ``y`` (error) plus a text column ``z``."""

NONPOSITIVE_CSV = "x,y\n1,2.0\n2,-4.1\n3,0.0\n4,7.9\n5,10.3\n"
"""Contains non-positive y values: a log y-axis must be unavailable."""

MANY_ROWS = 60
MANY_CSV = "a,b\n" + "".join(f"{i},{i * 1.5:.2f}\n" for i in range(MANY_ROWS))
"""More rows than the import preview shows (the preview is capped at 50)."""

CLIPBOARD_TAB_TEXT = "time\tload\n0\t0.5\n1\t2.4\n2\t4.6\n3\t6.1\n4\t8.5\n"
CLIPBOARD_COMMA_TEXT = "time,load\n0,0.5\n1,2.4\n2,4.6\n3,6.1\n4,8.5\n"
