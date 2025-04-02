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

Export these environment variables for configuring the pgvector docker container and the exposed port. For example:
```
    export POSTGRES_USER='postgres'
    export POSTGRES_PASSWORD='postgres'
    export POSTGRES_DB='test'
    export POSTGRES_PORT=5432
```

Internally in the class `rag.vector_store.pgvector_vectorstore.PgVectorVectorDB`, the class method `initialize_from_env_variables` is using the previous environment variables for initializing the class and returning an instance.


Run a docker container of `ankane/pgvector` in the backgound configured with the environment variables:
```
    docker run -d --name pgvector-db -p $POSTGRES_PORT:5432 -e POSTGRES_USER -e POSTGRES_PASSWORD -e POSTGRES_DB -e POSTGRES_PORT ankane/pgvector
```

Install `pgvector` extension in the database `$POSTGRES_DB`:
```
    docker exec -it pgvector-db psql -U $POSTGRES_USER -d $POSTGRES_DB
    test=# CREATE EXTENSION IF NOT EXISTS vector;
```

Installing the extension requires superuser privilegies.

Check that the extension is installed:
```
    SELECT vector '[1,2,3]'::vector;
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
    uvicorn demo_with_server:app
```