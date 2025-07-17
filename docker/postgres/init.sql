-- FormFinder PostgreSQL Initialization Script
-- Creates necessary databases and users for development

-- Create databases if they don't exist
CREATE DATABASE formfinder_test WITH ENCODING 'UTF8' LC_COLLATE='C' LC_CTYPE='C' TEMPLATE=template0;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE formfinder TO formfinder;
GRANT ALL PRIVILEGES ON DATABASE formfinder_test TO formfinder;

-- Connect to formfinder database to create extensions
\c formfinder

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Connect to test database to create extensions
\c formfinder_test

-- Create extensions in test database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";