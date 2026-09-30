"""Support Integrity Auditor (SIA): detect priority mismatch in support tickets."""

PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
LEVEL_OF = {name: i for i, name in enumerate(PRIORITY_LEVELS)}

# Levels at or above this index are treated as "urgent" (High/Critical).
# A mismatch is a ticket whose assigned and inferred severity fall on
# opposite sides of this boundary.
URGENT_THRESHOLD = LEVEL_OF["High"]

REQUIRED_COLUMNS = ["Ticket_ID", "Ticket_Description", "Priority_Level"]
