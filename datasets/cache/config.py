"""
Default configuration for the dataset cache.
"""

# The maximum size of the cache in gigabytes.
# If the cache exceeds this size, the oldest files will be deleted.
MAX_SIZE_GB = 50.0

# The time-to-live for cached items in days.
# Files older than this will be removed when the cache is cleaned.
# Set to 0 or a negative number to disable time-based expiration.
TTL_DAYS = 30
