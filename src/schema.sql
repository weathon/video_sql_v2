CREATE TABLE videos (
    video_id VARCHAR PRIMARY KEY,
    filename VARCHAR NOT NULL,
    duration_s DOUBLE NOT NULL
);

CREATE TABLE graph_nodes (
    row_id VARCHAR PRIMARY KEY,
    node_type VARCHAR NOT NULL CHECK (node_type IN ('entity', 'episode', 'semantic', 'visual_clip')),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE episodes (
    row_id VARCHAR PRIMARY KEY REFERENCES graph_nodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    granularity VARCHAR NOT NULL CHECK (granularity IN ('30sec', '3min', '10min', '1h')),
    text VARCHAR NOT NULL
);

CREATE TABLE visual_clips (
    row_id VARCHAR PRIMARY KEY REFERENCES graph_nodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
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
    snapshot_s DOUBLE NOT NULL,
    subject VARCHAR NOT NULL,
    predicate VARCHAR NOT NULL,
    object VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE openie_triples (
    row_id VARCHAR PRIMARY KEY,
    episode_row_id VARCHAR NOT NULL REFERENCES episodes(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    subject VARCHAR NOT NULL,
    predicate VARCHAR NOT NULL,
    object VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE graph_edges (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    src_row_id VARCHAR NOT NULL REFERENCES graph_nodes(row_id),
    dst_row_id VARCHAR NOT NULL REFERENCES graph_nodes(row_id),
    edge_type VARCHAR NOT NULL CHECK (edge_type IN ('mentioned_in', 'appeared_in', 'has_property', 'temporal_next', 'contains', 'co_clip')),
    weight DOUBLE NOT NULL
);

CREATE TABLE topic_chains (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    entity VARCHAR NOT NULL,
    has_lifecycle BOOLEAN NOT NULL
);

CREATE TABLE topic_facts (
    row_id VARCHAR PRIMARY KEY,
    chain_row_id VARCHAR NOT NULL REFERENCES topic_chains(row_id),
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    time_s DOUBLE NOT NULL,
    fact VARCHAR NOT NULL
);

CREATE TABLE event_chains (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    name VARCHAR NOT NULL,
    chain_type VARCHAR NOT NULL
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
    step_index INTEGER NOT NULL,
    start_s DOUBLE NOT NULL,
    end_s DOUBLE NOT NULL,
    description VARCHAR NOT NULL
);
