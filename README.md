# KFF Affordable Care Act — Scraper + RAG local

Pipeline para extraer articulos de la categoria "Affordable Care Act" de
[kff.org](https://www.kff.org/affordable-care-act/), guardar su contenido y
consultarlos por busqueda semantica usando una base vectorial local (ChromaDB),
con generacion de respuesta final via LLM (Gemini).

Todo el scraping usa la API REST publica de WordPress que expone kff.org
(`/wp-json/wp/v2/...`), en vez de raspar el HTML renderizado.

## Estructura

```
main.py                      CLI principal: compara retrievers y genera respuesta con LLM
query_chroma.py               CLI de inspeccion/mantenimiento de la coleccion Chroma

ingest/
  fetch_articles.py           listado de articulos (ex scrapper_url.py)
  fetch_content.py            contenido completo (ex scrapper_content.py)
  index_to_chroma.py          embeddings -> ChromaDB
  run_pipeline.py             corre los tres pasos anteriores en orden

retrieval/
  retrieval_methods.py        build_retrievers(): vectorial, BM25, hibrido, hibrido+rerank

llm/
  answer.py                   cliente Gemini, prompt template, generacion de respuesta

tools/
  meta_queries.py              lookups de metadata (conteo, mas recientes,
                                excerpt de un articulo) sin LLM ni busqueda
                                semantica -- la decision de cuando usarlos
                                la toma Gemini via function-calling

graph/
  pipeline.py                  grafo LangGraph: condensa pregunta -> Gemini
                                clasifica intent via function-calling (meta
                                vs contenido) -> [meta: responde directo con
                                tools/meta_queries.py] / [contenido: valida ->
                                retrieve -> grade -> generate -> valida respuesta]

data/
  kff_aca_articles.csv
  kff_aca_articles.json
  kff_aca_articles_content.json

chroma_db/                    base vectorial persistente (se genera en disco)
```

## Requisitos

```
pip install requests beautifulsoup4 chromadb sentence-transformers langchain-chroma langchain-classic langchain-community langchain-huggingface python-dotenv google-genai
```

## Pipeline de datos

Los scripts se ejecutan en orden desde la raiz del proyecto, cada uno alimenta al siguiente:

```
python ingest/fetch_articles.py    -> data/kff_aca_articles.json          (listado de articulos)
python ingest/fetch_content.py     -> data/kff_aca_articles_content.json  (contenido completo)
python ingest/index_to_chroma.py   -> chroma_db/                         (embeddings indexados)
```

O correr los tres de una vez con `ingest/run_pipeline.py` (los ejecuta como
subprocesos, en orden, y corta si alguno falla):

```
python ingest/run_pipeline.py                 # todo el catalogo, incremental
python ingest/run_pipeline.py --top-n 161      # solo los N articulos mas recientes
python ingest/run_pipeline.py --force          # reprocesa todo desde cero
```

`--top-n` se pasa a `fetch_content.py`; `--force` se pasa a `fetch_content.py`
e `index_to_chroma.py`.

### 1. `ingest/fetch_articles.py`

Recorre la categoria "Affordable Care Act" (`category_id=3962798`) via
`GET /wp-json/wp/v2/posts?categories=...`, paginando de 100 en 100 hasta
agotar `X-WP-TotalPages`. Incluye reintentos con backoff (`urllib3.Retry`)
y un `User-Agent` de navegador para evitar cortes de conexion.

Tambien resuelve `categories` (IDs) contra `GET /wp-json/wp/v2/categories`
para guardar nombres legibles, y calcula `year` desde la fecha de publicacion.

Genera:
- `data/kff_aca_articles.csv`
- `data/kff_aca_articles.json` — lista de
  `{id, title, url, date, year, category, primary_topic, excerpt}`

### 2. `ingest/fetch_content.py`

Toma los `TOP_N` articulos mas recientes (por `date`) de
`data/kff_aca_articles.json` y descarga el contenido completo de cada uno via
`GET /wp-json/wp/v2/posts/{id}?_fields=content`. Limpia el HTML con
BeautifulSoup (quita `script`/`style` y el bloque embebido de
"Related Content") y deja texto plano.

`TOP_N` es `None` por defecto (procesa todos los articulos listados);
se puede acotar puntualmente con `python ingest/fetch_content.py --top-n 20`.

Es incremental: si un articulo ya tiene contenido guardado en
`data/kff_aca_articles_content.json`, lo reusa tal cual en vez de
re-descargarlo. Usar `python ingest/fetch_content.py --force` para
re-descargar todo (por ejemplo, si KFF corrigio un articulo ya publicado).

Genera: `data/kff_aca_articles_content.json` —
`{id, title, url, date, year, category, primary_topic, excerpt, content}`

### 3. `ingest/index_to_chroma.py`

Trocea el texto de cada articulo en chunks de ~800 caracteres (100 de
solape), genera embeddings locales con `sentence-transformers`
(`all-MiniLM-L6-v2`, sin API key) y los guarda en una coleccion persistente
de ChromaDB.

Genera: `chroma_db/` — coleccion `kff_aca_articles`, un vector por chunk
con metadata `{article_id, title, url, date, chunk_index, year, category,
primary_topic, excerpt}`.

Tambien es incremental: si un `article_id` ya tiene chunks en la coleccion,
se saltea (no se vuelve a embeber). Usar
`python ingest/index_to_chroma.py --force` para re-embeber y reemplazar
todos los articulos (borra los chunks viejos de cada uno antes de
reinsertar, para no dejar chunks huerfanos si el conteo cambio).

## Consultas

### `main.py` — CLI principal (retrieval + LLM)

Requiere `.env` con `VERTEX_CREDENTIALS_JSON`, `VERTEX_PROJECT_ID`,
`VERTEX_LOCATION` para el cliente de Gemini via Vertex AI.

```
python main.py
```

Por cada pregunta: muestra la comparativa de los 4 metodos de retrieval
(vectorial, BM25, hibrido, hibrido+reranking) y genera la respuesta final
con Gemini usando el metodo hibrido+reranking como contexto (configurable
via `HYBRID_RERANK_KEY` en `main.py`).

### `retrieval/retrieval_methods.py` — solo comparativa de retrieval

```
python retrieval/retrieval_methods.py
```

Igual que `main.py` pero sin el paso de generacion con LLM.

### `query_chroma.py` — inspeccion y mantenimiento

```
python query_chroma.py count                       # total de chunks
python query_chroma.py peek 5                       # vistazo crudo, sin buscar
python query_chroma.py list_articles                # articulos unicos indexados
python query_chroma.py chunks <article_id>           # chunks de un articulo, en orden
python query_chroma.py full <article_id>             # texto completo reconstruido
python query_chroma.py search "pregunta libre" 3     # busqueda semantica (top N)
python query_chroma.py delete_collection si          # elimina la coleccion completa
```

`chunks`/`full` filtran por metadata exacta (`where={"article_id": ...}`).
`search` es la unica que hace busqueda semantica real (embedding + distancia
vectorial).

## Meta-preguntas sobre la coleccion (function-calling)

Ademas del RAG (busqueda semantica + generacion), el grafo de `graph/pipeline.py`
resuelve directo contra la coleccion, sin retrieval, tres tipos de pregunta
sobre la coleccion misma:

- **Conteo**: "How many articles do you have?" -> cuenta articulos unicos
  indexados en Chroma.
- **Mas recientes**: "What are the 5 most recent articles?" -> top N por
  `date` (N configurable en la pregunta, default 5).
- **Resumen de un articulo nombrado**: "Give me a summary of the article
  '<titulo>'" -> devuelve el `excerpt` editorial de KFF (ya viene en la
  metadata de cada chunk, no se genera con el LLM).

La decision de que tool usar (o ninguna, si es pregunta de contenido) la toma
**Gemini via function-calling nativo**, no reglas de keywords/regex: el nodo
`classify_intent` (`graph/pipeline.py`) le pasa la pregunta junto con 4
`FunctionDeclaration` (`count_articles`, `list_recent_articles`,
`get_article_excerpt`, y `answer_from_articles` para todo lo demas). Gemini
elige la funcion y, para `get_article_excerpt`, extrae el titulo mencionado
como parametro -- sin depender de que este entre comillas ni de que la
pregunta calce con un patron fijo. `tools/meta_queries.py` solo ejecuta el
lookup una vez decidida la intencion (`find_article_by_title` hace el match
final contra los titulos indexados, por contencion literal o similitud
aproximada con `difflib`, ya que el titulo que extrae Gemini no siempre es
identico caracter a caracter al indexado).

Se eligio este enfoque sobre un router por keywords porque `condense_question`
puede reformular la pregunta (inserta/cambia palabras), lo que rompia
sistematicamente el matching por substrings fijos; function-calling es
robusto a esa variacion de fraseo a cambio de un call a Gemini adicional por
pregunta (antes esta clasificacion no costaba nada).

`classify_intent` corre justo despues de `condense_question` y **antes** del
gatekeeper `validate_query`, a proposito: `validate_query` rechaza preguntas
que no sean sobre politica de salud/ACA, y una meta-pregunta como "how many
articles do you have?" es sobre el sistema mismo, no sobre el dominio --
pasarla por ese filtro la rechazaria por error. En el caso meta, el grafo
salta directo a `handle_meta_query` y evita `validate_query`,
`retrieve`/`grade_documents`/`generate`/`validate_answer` -- ninguno de esos
jueces LLM se ejecuta para estas tres preguntas.

## Notas y decisiones

- El listado (`ingest/fetch_articles.py`) y la descarga de contenido
  (`ingest/fetch_content.py`) estan separados a proposito: permite re-listar
  articulos sin tener que re-descargar contenido ya procesado, y viceversa.
- `TOP_N` en `ingest/fetch_content.py` (o `--top-n`) controla cuantos
  articulos recientes se procesan; `None`/sin flag procesa todos, un numero
  lo acota y reduce el tiempo de scraping y de indexado.
- Los embeddings son locales (no requieren `OPENAI_API_KEY` ni similar).
- `chroma_db/` es una base persistente en disco; volver a correr
  `ingest/index_to_chroma.py` es incremental (saltea articulos ya indexados)
  y usa `upsert` (no duplica) sobre los mismos ids.
- `category` guarda una sola categoria secundaria (distinta del ID fijo de
  "Affordable Care Act", redundante en todo el dataset), pensado para
  filtros simples en Chroma (`where={"category": ...}`).
