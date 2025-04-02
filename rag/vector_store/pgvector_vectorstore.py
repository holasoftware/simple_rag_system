import itertools
import os
import logging


import psycopg
from psycopg import sql
from psycopg.types.json import Json
from pgvector.psycopg import register_vector
import numpy as np

from .base import VectorDB
from ..document import Document


logger = logging.getLogger(__name__)


class PgVectorVectorDB(VectorDB):
    def __init__(
        self,
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="postgres",
        m=16,
        ef_construction=64,
        vector_dimension=1536,
        table_name="documents",
        create_extension=False,
    ):
        if not isinstance(vector_dimension, int) and vector_dimension <= 0:
            raise ValueError("Invalid vector dimention %s" % vector_dimension)

        logger.info(
            "Connecting to postgres host=%s port=%s dbname=%s user=%s password=%s",
            host,
            port,
            database,
            user,
            password,
        )

        self.conn = psycopg.connect(
            host=host, port=port, dbname=database, user=user, password=password
        )

        register_vector(self.conn)

        self.m = m
        self.ef_construction = ef_construction
        self.vector_dimension = vector_dimension
        self.table_name = table_name

        embedding_idx_name = table_name + "_embedding_idx"
        self.embedding_idx_name = embedding_idx_name

        self._init_db(
            m=m,
            ef_construction=ef_construction,
            table_name=table_name,
            embedding_idx_name=embedding_idx_name,
            vector_dimension=vector_dimension,
            create_extension=create_extension,
        )

    @classmethod
    def initialize_from_env_variables(cls, **kw):
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            database=os.getenv("POSTGRES_DB", "postgres"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            **kw
        )

    def _init_db(
        self,
        m,
        ef_construction,
        table_name,
        embedding_idx_name,
        vector_dimension,
        create_extension=False,
    ):
        with self.conn.cursor() as cur:
            if create_extension:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute(
                sql.SQL(
                    """
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({vector_dimension}),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """
                ).format(
                    table_name=sql.Identifier(table_name),
                    vector_dimension=sql.Literal(vector_dimension),
                )
            )
            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS {embedding_idx_name} 
                ON {table_name} USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction})
            """
                ).format(
                    embedding_idx_name=sql.Identifier(embedding_idx_name),
                    table_name=sql.Identifier(table_name),
                    m=sql.Literal(m),
                    ef_construction=sql.Literal(ef_construction),
                )
            )
            self.conn.commit()

    def _verify_index(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM pg_indexes
                WHERE tablename = %s AND indexdef LIKE '%%USING hnsw%%embedding%%' AND indexname = %s AND schemaname = 'public'
            """,
                self.table_name,
                self.table_name + "_embedding_idx",
            )

            return cur.fetchone()[0] == 1

    def store_document(self, content, embedding, metadata=None):
        logger.debug("Storing document: %s <%s>", content, embedding)

        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} (content, metadata, embedding) VALUES (%s, %s, %s)"
                ).format(sql.Identifier(self.table_name)),
                (
                    content,
                    Json(metadata) if metadata is not None else None,
                    np.array(embedding),
                ),
            )
            self.conn.commit()

    def store_documents_in_batch(self, content_list, embedding_list, metadata_list):
        with self.vector_db.conn.cursor() as cur:
            batch_size = min(len(content_list), len(embedding_list), len(metadata_list))
            metadata_list = [
                Json(metadata) if metadata is not None else None
                for metadata in metadata_list
            ]

            values_template = sql.SQL("({}, {}, {})").format(
                sql.Placeholder(), sql.Placeholder(), sql.Placeholder()
            )

            all_values_template = sql.SQL(",").join(
                [values_template for _ in range(batch_size)]
            )

            query = sql.SQL(
                "INSERT INTO {table_name} (content, embedding, metadata) VALUES {all_values}"
            ).format(
                table_name=sql.Identifier(self.table_name),
                all_values=all_values_template,
            )

            # flatten parameters
            params = list(
                itertools.chain(*zip(content_list, embedding_list, metadata_list))
            )

            cur.execute(query, params)
            self.conn.commit()

    def similarity_search(self, embedding, k=3, metadata_filter=None):
        with self.conn.cursor() as cur:
            if metadata_filter:
                cur.execute(
                    sql.SQL(
                        "SELECT id, content, metadata, created_at FROM {table_name} WHERE metadata @> %s ORDER BY embedding <=> %s LIMIT %s"
                    ).format(table_name=sql.Identifier(self.table_name)),
                    (metadata_filter, np.array(embedding), k),
                )
            else:
                cur.execute(
                    sql.SQL(
                        "SELECT id, content, metadata, created_at FROM {table_name} ORDER BY embedding <=> %s LIMIT %s"
                    ).format(table_name=sql.Identifier(self.table_name)),
                    (np.array(embedding), k),
                )

            result = [
                Document(id=row[0], content=row[1], metadata=row[2], created_at=row[3])
                for row in cur.fetchall()
            ]
            return result

    def close(self):
        logger.debug("Closing connection")
        self.conn.close()
