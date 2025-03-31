# Simple RAG system
Simple RAG system abstraction


## Installation
Create a virtualenv:
```
    python -m venv venv
```

Activate the virtualenv:
```
    . venv/bin/activate
```

Install dependencies:
```
    pip install -r requirements.txt
```

Pull the official docker image of `pgvector`:
```
    docker pull ankane/pgvector
```

Export these environment variables for configuring the pgvector docker container:
```
    export POSTGRES_USER='postgres'
    export POSTGRES_PASSWORD='postgres'
    export POSTGRES_DB='test'
    export POSTGRES_PORT=5432
```

Internally the class `rag.vector_store.pgvector_vectorstore.PgVectorVectorDB` is using the same environment variables by default in the method `initialize_from_env_variables`.


Run a docker container:
```
    docker run --name pgvector-db -p $POSTGRES_PORT:$POSTGRES_PORT ankane/pgvector
```

Create database and install `pgvector` extension:
```
    docker exec -it pgvector-db psql -U postgres -d test
    test=# CREATE EXTENSION IF NOT EXISTS vector;
```

For running the demo, configure the environment variable for your Openai API key `LLM_API_KEY`. 
```
    export LLM_API_KEY=<Openai API key>
```

Run the demo:
```
    python demo.py
```

There is also a demo server:
```
    python demo_with_server.py
```