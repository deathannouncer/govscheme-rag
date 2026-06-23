CREATE TABLE IF NOT EXISTS scheme_chunks (
    id SERIAL PRIMARY KEY,
    scheme_name TEXT NOT NULL,
    chunk_type TEXT NOT NULL,      -- description | eligibility | benefits | application_process
    content TEXT NOT NULL,
    official_link TEXT,
    milvus_id BIGINT,              -- mirrors id once the chunk has been embedded
    search_vector tsvector
);

CREATE INDEX IF NOT EXISTS idx_scheme_chunks_fts ON scheme_chunks USING GIN (search_vector);

CREATE OR REPLACE FUNCTION scheme_chunks_tsvector_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvectorupdate ON scheme_chunks;
CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
ON scheme_chunks FOR EACH ROW EXECUTE FUNCTION scheme_chunks_tsvector_trigger();
