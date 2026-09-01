CREATE TABLE videos (
    video_id VARCHAR PRIMARY KEY,
    scale VARCHAR NOT NULL CHECK (scale IN ('Day', 'Week', 'Month')),
    duration_s DOUBLE NOT NULL,
    source_count INTEGER NOT NULL
);

CREATE TABLE source_groups (
    source_group_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    name VARCHAR NOT NULL,
    timeline_start_s DOUBLE NOT NULL,
    timeline_end_s DOUBLE NOT NULL
);

CREATE TABLE source_videos (
    source_video_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    source_group_id VARCHAR NOT NULL REFERENCES source_groups(source_group_id),
    filename VARCHAR NOT NULL,
    duration_s DOUBLE NOT NULL,
    timeline_start_s DOUBLE NOT NULL,
    timeline_end_s DOUBLE NOT NULL
);

CREATE TABLE graph_nodes (
    row_id VARCHAR PRIMARY KEY,
    node_type VARCHAR NOT NULL CHECK (node_type IN ('entity', 'episode', 'semantic', 'visual_clip', 'object_state')),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE episodes (
    row_id VARCHAR PRIMARY KEY REFERENCES graph_nodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    source_video_id VARCHAR REFERENCES source_videos(source_video_id),
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    granularity VARCHAR NOT NULL CHECK (granularity IN ('30sec', '3min', '1h', '6h', 'video', 'scale')),
    text VARCHAR NOT NULL
);

CREATE TABLE visual_clips (
    row_id VARCHAR PRIMARY KEY REFERENCES graph_nodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    source_video_id VARCHAR NOT NULL REFERENCES source_videos(source_video_id),
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    source_start_s DOUBLE NOT NULL,
    source_end_s DOUBLE NOT NULL,
    filename VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE entities (
    row_id VARCHAR PRIMARY KEY REFERENCES graph_nodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    name VARCHAR NOT NULL,
    normalized_name VARCHAR NOT NULL,
    first_seen_s DOUBLE NOT NULL,
    last_seen_s DOUBLE NOT NULL,
    frequency INTEGER NOT NULL
);

CREATE TABLE semantic_triples (
    row_id VARCHAR PRIMARY KEY REFERENCES graph_nodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    caption_30sec_row_id VARCHAR NOT NULL REFERENCES episodes(row_id),
    snapshot_s DOUBLE NOT NULL,
    subject VARCHAR NOT NULL,
    predicate VARCHAR NOT NULL,
    object VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE object_states (
    row_id VARCHAR PRIMARY KEY REFERENCES graph_nodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    source_video_id VARCHAR NOT NULL REFERENCES source_videos(source_video_id),
    episode_3min_row_id VARCHAR NOT NULL REFERENCES episodes(row_id),
    caption_30sec_row_id VARCHAR NOT NULL REFERENCES episodes(row_id),
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    object VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE graph_edges (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    src_row_id VARCHAR NOT NULL REFERENCES graph_nodes(row_id),
    dst_row_id VARCHAR NOT NULL REFERENCES graph_nodes(row_id),
    edge_type VARCHAR NOT NULL CHECK (edge_type IN ('mentioned_in', 'appeared_in', 'has_property', 'state_in', 'temporal_next', 'contains', 'co_clip')),
    weight DOUBLE NOT NULL
);

CREATE TABLE topic_chains (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    entity VARCHAR NOT NULL,
    has_lifecycle BOOLEAN NOT NULL,
    level VARCHAR NOT NULL CHECK (level IN ('30min', '1h', '6h', 'video', 'scale')),
    scope_id VARCHAR NOT NULL,
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    summary VARCHAR NOT NULL
);

CREATE TABLE topic_facts (
    row_id VARCHAR PRIMARY KEY,
    chain_row_id VARCHAR NOT NULL REFERENCES topic_chains(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    caption_30sec_row_id VARCHAR NOT NULL REFERENCES episodes(row_id),
    time_s DOUBLE NOT NULL,
    fact VARCHAR NOT NULL
);

CREATE TABLE topic_fact_evidence (
    row_id VARCHAR PRIMARY KEY,
    fact_row_id VARCHAR NOT NULL REFERENCES topic_facts(row_id),
    caption_30sec_row_id VARCHAR NOT NULL REFERENCES episodes(row_id)
);

CREATE TABLE event_chains (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    name VARCHAR NOT NULL,
    chain_type VARCHAR NOT NULL,
    level VARCHAR NOT NULL CHECK (level IN ('30min', '1h', '6h', 'video', 'scale')),
    scope_id VARCHAR NOT NULL,
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    summary VARCHAR NOT NULL
);

CREATE TABLE event_chain_entities (
    row_id VARCHAR PRIMARY KEY,
    chain_row_id VARCHAR NOT NULL REFERENCES event_chains(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    entity VARCHAR NOT NULL
);

CREATE TABLE event_steps (
    row_id VARCHAR PRIMARY KEY,
    chain_row_id VARCHAR NOT NULL REFERENCES event_chains(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    caption_start_row_id VARCHAR NOT NULL REFERENCES episodes(row_id),
    caption_end_row_id VARCHAR NOT NULL REFERENCES episodes(row_id),
    step_index INTEGER NOT NULL,
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    description VARCHAR NOT NULL
);

CREATE TABLE event_step_evidence (
    row_id VARCHAR PRIMARY KEY,
    step_row_id VARCHAR NOT NULL REFERENCES event_steps(row_id),
    caption_30sec_row_id VARCHAR NOT NULL REFERENCES episodes(row_id)
);
