def get_unique_segments(road_ids):
    """Return the sequence of road segments with consecutive duplicates collapsed."""
    if not road_ids:
        return []
    unique = [road_ids[0]]
    for i in range(1, len(road_ids)):
        if road_ids[i] != road_ids[i-1]:
            unique.append(road_ids[i])
    return unique
