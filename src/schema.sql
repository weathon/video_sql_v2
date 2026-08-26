CREATE TABLE videos (
    video_id VARCHAR PRIMARY KEY,
    sequence_index INTEGER NOT NULL,
    source_split VARCHAR NOT NULL,
    video_path VARCHAR NOT NULL,
    duration_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL
);

CREATE TABLE questions (
    question_id BIGINT PRIMARY KEY,
    source_split VARCHAR NOT NULL,
    question VARCHAR NOT NULL,
    answer VARCHAR NOT NULL,
    question_type VARCHAR NOT NULL,
    temporal_certificate VARCHAR NOT NULL
);

CREATE TABLE question_clues (
    question_id BIGINT NOT NULL,
    video_id VARCHAR NOT NULL,
    clue_order INTEGER NOT NULL,
    interval_order INTEGER NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL
);

CREATE TABLE asr_segments (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE captions (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    granularity_sec INTEGER NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE entities (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    chunk_id VARCHAR,
    entity VARCHAR NOT NULL
);

CREATE TABLE episodic_triples (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    chunk_id VARCHAR NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL,
    subject VARCHAR NOT NULL,
    predicate VARCHAR NOT NULL,
    object VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE semantic_triples (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    chunk_id VARCHAR,
    subject VARCHAR NOT NULL,
    predicate VARCHAR NOT NULL,
    object VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE activities (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL,
    name VARCHAR NOT NULL,
    location VARCHAR NOT NULL,
    summary VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE event_chains (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    chain_type VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE event_steps (
    row_id VARCHAR PRIMARY KEY,
    event_chain_row_id VARCHAR NOT NULL,
    video_id VARCHAR NOT NULL,
    step_index INTEGER NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL,
    description VARCHAR NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE topic_chains (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    entity VARCHAR NOT NULL,
    has_lifecycle BOOLEAN NOT NULL
);

CREATE TABLE topic_facts (
    row_id VARCHAR PRIMARY KEY,
    topic_chain_row_id VARCHAR NOT NULL,
    video_id VARCHAR NOT NULL,
    fact_index INTEGER NOT NULL,
    start_sec DOUBLE NOT NULL,
    global_time INTERVAL NOT NULL,
    text VARCHAR NOT NULL
);

CREATE TABLE video_clips (
    row_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    start_sec DOUBLE NOT NULL,
    end_sec DOUBLE NOT NULL,
    global_start INTERVAL NOT NULL,
    global_end INTERVAL NOT NULL
);

CREATE TABLE embeddings (
    source_table VARCHAR NOT NULL,
    row_id VARCHAR NOT NULL,
    modality VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    vector FLOAT[] NOT NULL,
    PRIMARY KEY (source_table, row_id)
);

CREATE INDEX question_clues_question_idx ON question_clues(question_id);
CREATE INDEX question_clues_video_idx ON question_clues(video_id, start_sec);
CREATE INDEX asr_video_time_idx ON asr_segments(video_id, start_sec);
CREATE INDEX captions_video_time_idx ON captions(video_id, start_sec);
CREATE INDEX episodic_video_time_idx ON episodic_triples(video_id, start_sec);
CREATE INDEX activities_video_time_idx ON activities(video_id, start_sec);
CREATE INDEX event_steps_video_time_idx ON event_steps(video_id, start_sec);
CREATE INDEX topic_facts_video_time_idx ON topic_facts(video_id, start_sec);
