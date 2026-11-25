"""A module-global the pollution worker writes and the probe worker reads: proof that a
cancelled job's globals do not reach the next job in the same warm worker."""

MARK = "clean"
