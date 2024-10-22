CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    embedding VECTOR(1696),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
