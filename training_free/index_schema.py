import re


NODE_TABLES = ("entity", "episode", "semantic", "visual_clip")
SEARCH_TABLES = {"entity": "text", "episode": "text", "semantic": "text", "visual_clip": "text", "topic_facts": "fact", "event_steps": "description"}
EDGE_TABLES = {"mentioned_in": ("entity", "episode"), "appeared_in": ("entity", "visual_clip"), "has_property": ("entity", "semantic"), "temporal_next": ("episode", "episode"), "contains": ("episode", "episode"), "co_clip_visual_episode": ("visual_clip", "episode"), "co_clip_episode_visual": ("episode", "visual_clip")}
DDL = {table: f"CREATE TABLE {table} (id VARCHAR PRIMARY KEY, node_type VARCHAR, text VARCHAR, start_ts BIGINT, end_ts BIGINT, metadata JSON)" for table in NODE_TABLES}
DDL.update({"topic_chains": "CREATE TABLE topic_chains (id VARCHAR PRIMARY KEY, entity VARCHAR, has_lifecycle BOOLEAN)", "event_chains": "CREATE TABLE event_chains (id VARCHAR PRIMARY KEY, name VARCHAR, key_entities VARCHAR[])", "topic_facts": "CREATE TABLE topic_facts (id VARCHAR PRIMARY KEY, chain_id VARCHAR REFERENCES topic_chains(id), time VARCHAR, fact VARCHAR, location VARCHAR)", "event_steps": "CREATE TABLE event_steps (id VARCHAR PRIMARY KEY, chain_id VARCHAR REFERENCES event_chains(id), step_index INTEGER, day VARCHAR, start_time VARCHAR, end_time VARCHAR, description VARCHAR)", "topic_entities": "CREATE TABLE topic_entities (chain_id VARCHAR REFERENCES topic_chains(id), entity_id VARCHAR REFERENCES entity(id))", "event_entities": "CREATE TABLE event_entities (chain_id VARCHAR REFERENCES event_chains(id), entity_id VARCHAR REFERENCES entity(id))"})
DDL.update({table: f"CREATE TABLE {table} (src_id VARCHAR REFERENCES {src}(id), dst_id VARCHAR REFERENCES {dst}(id), edge_type VARCHAR, weight DOUBLE)" for table, (src, dst) in EDGE_TABLES.items()})
SCHEMA = "\n".join(DDL.values()) + "\nmetadata preserves native MAGIC fields. Timestamps are EgoLife DHHMMSSFF, not seconds. topic_facts.time and event_steps.day/start_time/end_time retain their source strings. All rows in this database have already been filtered for the question's cutoff."


def fact_interval(time: str) -> tuple[int, int] | None:
    if time in ("Various", "GENERAL", "Overall habit"):
        return None
    match = re.fullmatch(r"DAY(\d) (\d\d):(\d\d)(?:-(\d\d):(\d\d))?(?: \((?:\d+|first entry|second entry)\))?", time)
    if match is None:
        raise ValueError(f"Unrecognized topic fact time: {time!r}")
    day, hour, minute, end_hour, end_minute = match.groups()
    start = int(day + hour + minute + "0000")
    end = start if end_hour is None else int(day + end_hour + end_minute + "0000")
    return start, end


def timestamp_seconds(timestamp: int, fps: float) -> float:
    value = str(timestamp).zfill(9)
    day = int(value[0])
    hour = int(value[1:3])
    minute = int(value[3:5])
    second = int(value[5:7])
    frame = int(value[7:9])
    return (day - 1) * 86400 + hour * 3600 + minute * 60 + second + frame / fps
