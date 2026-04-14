# Module 15: Database Systems for Tick Data

**Prerequisites:** Module 09 (Python for Quantitative Finance), Module 12 (Data Structures & Algorithms for Finance)
**Builds toward:** Module 32 (Backtesting Infrastructure), Module 34 (Data Engineering Pipelines)

---

## Table of Contents

1. [Motivation: The Tick Data Problem](#1-motivation-the-tick-data-problem)
2. [Time-Series Data Characteristics](#2-time-series-data-characteristics)
3. [kdb+/q: The Gold Standard](#3-kdbq-the-gold-standard)
4. [QuestDB: SQL for Time-Series](#4-questdb-sql-for-time-series)
5. [TimescaleDB: PostgreSQL Extension](#5-timescaledb-postgresql-extension)
6. [Apache Parquet & Arrow: Columnar Formats](#6-apache-parquet--arrow-columnar-formats)
7. [ClickHouse: OLAP at Scale](#7-clickhouse-olap-at-scale)
8. [Data Modeling for Tick Data](#8-data-modeling-for-tick-data)
9. [Ingestion Pipelines](#9-ingestion-pipelines)
10. [Query Patterns](#10-query-patterns)
11. [Archival Strategies](#11-archival-strategies)
12. [Exercises](#12-exercises)
13. [Summary](#13-summary)

---

## 1. Motivation: The Tick Data Problem

Quantitative trading generates and consumes enormous volumes of time-stamped data. A single US equity exchange produces ~1--3 billion messages per day. A firm monitoring 20 exchanges across equities, futures, and options faces a data rate of:

$$\text{Messages/day} \approx 20 \times 2 \times 10^9 = 4 \times 10^{10}$$

At ~40 bytes per message (compressed), this is ~1.6 TB/day of raw market data, or ~400 TB/year. Over a 10-year research horizon, the total dataset exceeds 4 PB.

The core challenge is not just storage volume but **query performance**. A researcher asks questions like:

- "What was the VWAP of AAPL between 10:30:00 and 10:35:00 on March 15, 2024?"
- "Compute 1-minute OHLCV bars for all S&P 500 constituents for the past 5 years."
- "Find all instances where the bid-ask spread exceeded 10 bps within 100 ms of an earnings announcement."

These queries must scan millions to billions of rows, aggregate, filter, and return results in seconds -- not hours.

| Requirement | Implication for Database Design |
|---|---|
| Append-heavy writes (millions/sec) | Write-optimized storage (LSM trees, append-only logs) |
| Time-ordered queries (99% of access) | Time-based partitioning and primary indexing |
| Columnar aggregation (VWAP, OHLCV) | Column-oriented storage for vectorized computation |
| High cardinality (thousands of symbols) | Efficient partition pruning and symbol indexing |
| Long retention (years to decades) | Tiered storage with compression |
| Reproducibility (exact replay) | Immutable data, deterministic query semantics |

```mermaid
graph LR
    A[Exchange Feeds<br/>~50M msgs/sec] --> B[Feed Handler<br/>Normalize + Timestamp]
    B --> C[Ingestion Pipeline<br/>Deduplicate + Validate]
    C --> D[("Time-Series Database<br/>kdb+ / QuestDB / TimescaleDB")]
    D --> E[Research Queries<br/>Jupyter / Python]
    D --> F[Production Signals<br/>Real-time Aggregation]
    D --> G[Backtesting<br/>Historical Replay]

    style D fill:#f96,stroke:#333
```

---

## 2. Time-Series Data Characteristics

### 2.1 The Append-Only Pattern

Market data is fundamentally **append-only**: once a trade or quote is recorded, it is never modified (barring corporate action adjustments, which are handled as separate correction records). This property enables:

- **WAL-free writes:** No write-ahead log needed for crash recovery of recent data, since data is naturally ordered and idempotent.
- **Immutable segments:** Completed time partitions (yesterday's data) can be compressed and sealed, never requiring write access again.
- **Trivial replication:** Append-only data can be replicated by simply streaming the tail of the log.

### 2.2 Time-Ordered Access

The overwhelming majority of queries filter by time first, then by symbol, then aggregate:

$$\text{Query pattern:} \quad \text{WHERE } t_{\text{start}} \leq \text{timestamp} \leq t_{\text{end}} \text{ AND symbol} = S$$

This suggests a **composite primary key** of (timestamp, symbol) or, in databases that support it, time-based partitioning with a secondary symbol index.

### 2.3 High Write Throughput, Read Diversity

The write path is simple and uniform: append timestamped records in arrival order. The read path is diverse:

| Query Type | Access Pattern | Example |
|---|---|---|
| Point lookup | Single record by (time, symbol) | Last trade price |
| Range scan | Sequential records in time window | All trades 10:30-10:35 |
| Aggregation | Reduce range to summary | VWAP, OHLCV bars |
| Cross-symbol join | Multiple symbols over same time window | Correlation estimation |
| Replay | Sequential scan of all data in order | Backtesting engine |

### 2.4 Cardinality and Partitioning

A "series" in time-series databases is typically defined by the (symbol, exchange, data_type) tuple. For US equities:

$$\text{Series count} \approx 8{,}000 \text{ symbols} \times 15 \text{ exchanges} \times 3 \text{ types (trade, quote, BBO)} = 360{,}000$$

This is the **cardinality** of the tag space. Databases handle this differently:
- **kdb+:** Partitions by date, each date directory contains one file per column per table. Symbol is an enumerated column (integer index into a symbol list).
- **QuestDB:** Partitions by time interval (day, hour), WAL-based ingestion supports out-of-order writes within a partition.
- **TimescaleDB:** Hypertables partition by time (and optionally by symbol hash). Each chunk is a PostgreSQL table.
- **ClickHouse:** Partitions by expression (e.g., `toYYYYMM(timestamp)`), parts within a partition are sorted by the primary key.

---

## 3. kdb+/q: The Gold Standard

### 3.1 Architecture

kdb+ is a column-oriented, in-memory database designed specifically for time-series data. Its query language, **q**, is a terse vector-oriented language descended from APL/K. kdb+ has dominated quantitative finance since the early 2000s due to:

- **In-memory speed:** Entire tables (or recent partitions) reside in memory. Columnar layout enables SIMD-friendly sequential scans.
- **Vector operations:** q processes entire columns as vectors, avoiding per-row interpretation overhead.
- **Memory-mapped files:** Historical data is memory-mapped, leveraging the OS page cache for transparent out-of-core access.
- **IPC:** Built-in inter-process communication for distributed queries across multiple kdb+ instances.

**Data model.** A kdb+ table is a dictionary of column names to vectors. Each column is a flat, typed array stored contiguously in memory (or on disk as a memory-mapped file).

### 3.2 q Language Fundamentals

```q
/ q is read right-to-left. Comments start with /

/ Define a trade table
trades: ([]
    time:   09:30:00.000 09:30:00.001 09:30:00.005 09:30:00.010 09:30:00.015;
    sym:    `AAPL`AAPL`MSFT`AAPL`MSFT;
    price:  175.50 175.51 420.10 175.52 420.12;
    size:   100 200 300 150 250;
    side:   `B`S`B`B`S
 )

/ Select: all AAPL trades
select from trades where sym=`AAPL

/ Aggregation: VWAP by symbol
select vwap: size wavg price by sym from trades
/ wavg is weighted average: (sum size*price) % sum size

/ Time-bucketed OHLCV bars (1-second bars)
select
    o: first price,
    h: max price,
    l: min price,
    c: last price,
    v: sum size
  by sym, 1 xbar time.second
  from trades
```

### 3.3 VWAP Computation in q

The Volume-Weighted Average Price is the most common aggregation in tick data analysis:

$$\text{VWAP}(t_1, t_2) = \frac{\sum_{i: t_1 \leq t_i \leq t_2} p_i \cdot q_i}{\sum_{i: t_1 \leq t_i \leq t_2} q_i}$$

```q
/ VWAP for a specific symbol and time window
vwap_aapl: exec size wavg price from trades
    where sym=`AAPL, time within 09:30:00.000 09:30:00.015

/ VWAP in rolling 5-minute windows for all symbols
/ Using the built-in `wj` (window join) for asof-style computation
windows: -[;] 300000000000j /- 5 minutes in nanoseconds
vwap_rolling: wj1[windows;
    select by sym from trades;
    trades;
    (size; (wavg; `size; `price))]
```

### 3.4 OHLCV Aggregation

```q
/ 1-minute OHLCV bars for all symbols
ohlcv: select
    open:   first price,
    high:   max price,
    low:    min price,
    close:  last price,
    volume: sum size,
    vwap:   size wavg price,
    trades: count i
  by sym, 1 xbar time.minute
  from trades

/ 5-minute bars with additional statistics
ohlcv5: select
    open:   first price,
    high:   max price,
    low:    min price,
    close:  last price,
    volume: sum size,
    vwap:   size wavg price,
    twap:   avg price,                    / time-weighted average
    spread: avg ask - avg bid,            / requires quote table joined
    trades: count i,
    buy_volume:  sum size where side=`B,
    sell_volume: sum size where side=`S
  by sym, 5 xbar time.minute
  from trades
```

### 3.5 On-Disk Partitioned Tables

For historical data, kdb+ stores tables in a partitioned directory structure:

```text
/hdb/
  2024.03.13/
    trades/
      time        <- memory-mapped vector (binary)
      sym         <- enumerated symbol vector
      price       <- float vector
      size        <- int vector
    quotes/
      time
      sym
      bid
      ask
      bidsize
      asksize
  2024.03.14/
    trades/
      ...
  sym              <- symbol enumeration file
```

Each column within a date partition is a flat binary file. A query like `select from trades where date=2024.03.14, sym=`AAPL` first prunes to the single date partition (directory), then performs a linear scan of the `sym` column (memory-mapped, page-cache friendly) to build an index mask, and applies it to all other columns.

**Query performance.** Scanning 500 million trade records (one day of consolidated US equity tape) for a single symbol takes ~200 ms on a modern server with the data in page cache. Aggregating VWAP across all symbols takes ~1.5 seconds.

### 3.6 kdb+ Limitations

- **Licensing cost:** kdb+ is proprietary and expensive (~$100K+/year per core for production use).
- **Single-threaded queries:** A single q query runs on one core (though multiple queries can run in parallel on separate slaves).
- **Memory model:** In-memory mode requires the entire active dataset to fit in RAM; on-disk mode relies on the OS page cache, which provides no query-aware caching.
- **Learning curve:** q's terse, right-to-left syntax is notoriously difficult to learn.

---

## 4. QuestDB: SQL for Time-Series

### 4.1 Architecture

QuestDB is an open-source time-series database written in Java and C++, designed to combine kdb+-level query performance with a standard SQL interface. Key architectural choices:

- **Column-based storage:** Each column is stored as a separate memory-mapped file, enabling vector processing.
- **Designated timestamp:** Each table has a single column designated as the timestamp, which serves as the implicit sort key and partition index.
- **Partitioned tables:** Data is partitioned by time interval (HOUR, DAY, WEEK, MONTH, YEAR). Each partition is a directory of column files.
- **WAL (Write-Ahead Log):** Supports out-of-order ingestion within configurable lag windows, critical for multi-exchange feeds.
- **SIMD-accelerated SQL:** Query execution uses AVX2/AVX-512 instructions for filtering and aggregation.

### 4.2 Table Creation and Ingestion

```sql
-- Create a trade table with designated timestamp and daily partitioning
CREATE TABLE trades (
    timestamp TIMESTAMP,
    symbol    SYMBOL CAPACITY 16384 CACHE,  -- enumerated string type
    price     DOUBLE,
    quantity  INT,
    side      CHAR,
    exchange  SYMBOL CAPACITY 64 CACHE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL
  DEDUP UPSERT KEYS(timestamp, symbol);

-- Insert sample data
INSERT INTO trades VALUES
    ('2024-03-15T10:30:00.000001Z', 'AAPL', 175.50, 100, 'B', 'XNAS'),
    ('2024-03-15T10:30:00.000002Z', 'AAPL', 175.51, 200, 'S', 'XNAS'),
    ('2024-03-15T10:30:00.000003Z', 'MSFT', 420.10, 300, 'B', 'XNYS'),
    ('2024-03-15T10:30:00.000010Z', 'AAPL', 175.52, 150, 'B', 'XNAS');

-- Bulk ingestion via ILP (InfluxDB Line Protocol) over TCP:
-- trades,symbol=AAPL,exchange=XNAS price=175.50,quantity=100i,side="B" 1710499800000001000
```

### 4.3 VWAP and OHLCV Queries

```sql
-- VWAP for AAPL in a specific time window
SELECT symbol,
       SUM(price * quantity) / SUM(quantity) AS vwap
FROM trades
WHERE symbol = 'AAPL'
  AND timestamp BETWEEN '2024-03-15T10:30:00Z' AND '2024-03-15T10:35:00Z';

-- 1-minute OHLCV bars using SAMPLE BY
SELECT timestamp, symbol,
       FIRST(price)               AS open,
       MAX(price)                 AS high,
       MIN(price)                 AS low,
       LAST(price)                AS close,
       SUM(quantity)              AS volume,
       SUM(price * quantity) / SUM(quantity) AS vwap,
       COUNT(*)                   AS trade_count
FROM trades
WHERE timestamp IN '2024-03-15'
SAMPLE BY 1m FILL(PREV)          -- 1-minute bars, forward-fill gaps
ALIGN TO CALENDAR;

-- LATEST ON: get the most recent trade per symbol (asof semantics)
SELECT * FROM trades
LATEST ON timestamp PARTITION BY symbol;
```

### 4.4 SAMPLE BY: Time-Bucketed Aggregation

QuestDB's `SAMPLE BY` is purpose-built for time-series aggregation and compiles to an efficient single-pass scan:

```sql
-- 5-minute bars with buy/sell volume breakdown
SELECT timestamp, symbol,
       FIRST(price)               AS open,
       MAX(price)                 AS high,
       MIN(price)                 AS low,
       LAST(price)                AS close,
       SUM(quantity)              AS volume,
       SUM(CASE WHEN side = 'B' THEN quantity ELSE 0 END) AS buy_volume,
       SUM(CASE WHEN side = 'S' THEN quantity ELSE 0 END) AS sell_volume,
       SUM(price * quantity) / SUM(quantity)               AS vwap
FROM trades
WHERE symbol IN ('AAPL', 'MSFT', 'GOOGL')
  AND timestamp IN '2024-03-15'
SAMPLE BY 5m
ALIGN TO CALENDAR;
```

**Performance.** On a 32-core server with NVMe storage, QuestDB processes ~1.5 billion rows/second for SAMPLE BY aggregation on a single table with 10 columns, thanks to columnar SIMD processing.

### 4.5 Python Client

```python
"""
QuestDB Python client for tick data analysis.
Uses the official questdb package for ingestion
and psycopg2 (PG wire protocol) for queries.
"""

import pandas as pd
from questdb.ingress import Sender, IngressError, TimestampNanos
import psycopg2
from datetime import datetime, timezone


def ingest_trades(trades_df: pd.DataFrame) -> None:
    """
    Ingest a DataFrame of trades into QuestDB via ILP.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Columns: timestamp (datetime64[ns]), symbol (str),
                 price (float), quantity (int), side (str), exchange (str)
    """
    with Sender("localhost", 9009) as sender:
        for _, row in trades_df.iterrows():
            sender.row(
                "trades",
                symbols={
                    "symbol": row["symbol"],
                    "exchange": row["exchange"],
                },
                columns={
                    "price": row["price"],
                    "quantity": row["quantity"],
                    "side": row["side"],
                },
                at=TimestampNanos(int(row["timestamp"].timestamp() * 1e9)),
            )
        sender.flush()


def query_vwap(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Query VWAP for a symbol in a time window.

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g., 'AAPL')
    start, end : str
        ISO 8601 timestamps

    Returns
    -------
    pd.DataFrame
        DataFrame with vwap result
    """
    conn = psycopg2.connect(
        host="localhost", port=8812,
        user="admin", password="quest",
        database="qdb"
    )
    query = f"""
        SELECT symbol,
               SUM(price * quantity) / SUM(quantity) AS vwap,
               SUM(quantity) AS total_volume,
               COUNT(*) AS trade_count
        FROM trades
        WHERE symbol = '{symbol}'
          AND timestamp BETWEEN '{start}' AND '{end}'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def query_ohlcv_bars(
    symbols: list[str],
    date: str,
    bar_size: str = "1m"
) -> pd.DataFrame:
    """
    Query OHLCV bars for multiple symbols on a given date.

    Parameters
    ----------
    symbols : list[str]
        List of instrument symbols
    date : str
        Date string (e.g., '2024-03-15')
    bar_size : str
        Bar duration (e.g., '1m', '5m', '1h')

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by (timestamp, symbol)
    """
    sym_list = ", ".join(f"'{s}'" for s in symbols)
    conn = psycopg2.connect(
        host="localhost", port=8812,
        user="admin", password="quest",
        database="qdb"
    )
    query = f"""
        SELECT timestamp, symbol,
               FIRST(price) AS open,
               MAX(price)   AS high,
               MIN(price)   AS low,
               LAST(price)  AS close,
               SUM(quantity) AS volume,
               SUM(price * quantity) / SUM(quantity) AS vwap,
               COUNT(*) AS trades
        FROM trades
        WHERE symbol IN ({sym_list})
          AND timestamp IN '{date}'
        SAMPLE BY {bar_size}
        ALIGN TO CALENDAR
    """
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    conn.close()
    return df.set_index(["timestamp", "symbol"])
```

---

## 5. TimescaleDB: PostgreSQL Extension

### 5.1 Architecture

TimescaleDB extends PostgreSQL with time-series-specific optimizations while retaining full SQL compatibility, indexing, joins, and the PostgreSQL ecosystem (PostGIS, pg_stat_statements, logical replication). Key features:

- **Hypertables:** A hypertable is a virtual table that automatically partitions data into **chunks** by time interval. Each chunk is a regular PostgreSQL table, but the hypertable abstraction handles partition creation, pruning, and management transparently.
- **Continuous aggregates:** Materialized views that automatically refresh as new data arrives, providing pre-computed OHLCV bars without query-time computation.
- **Compression:** Native columnar compression on older chunks, achieving 10--20x compression with minimal query overhead.
- **Data retention policies:** Automatic chunk dropping for expired data.

### 5.2 Schema and Hypertable Creation

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create the trades table
CREATE TABLE trades (
    timestamp   TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    quantity    INTEGER     NOT NULL,
    side        CHAR(1)     NOT NULL,  -- 'B' or 'S'
    exchange    TEXT        NOT NULL,
    sequence_no BIGINT      NOT NULL
);

-- Convert to a hypertable, partitioned by time (1-day chunks)
SELECT create_hypertable(
    'trades',
    by_range('timestamp', INTERVAL '1 day')
);

-- Add a composite index for symbol + time queries
CREATE INDEX idx_trades_sym_time ON trades (symbol, timestamp DESC);

-- Enable compression on chunks older than 7 days
ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Policy: auto-compress chunks older than 7 days
SELECT add_compression_policy('trades', INTERVAL '7 days');

-- Policy: drop chunks older than 2 years
SELECT add_retention_policy('trades', INTERVAL '2 years');
```

### 5.3 Continuous Aggregates

Continuous aggregates pre-compute OHLCV bars and incrementally refresh as new data arrives:

```sql
-- Create a continuous aggregate for 1-minute OHLCV bars
CREATE MATERIALIZED VIEW ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp)           AS bucket,
    symbol,
    FIRST(price, timestamp)                      AS open,
    MAX(price)                                   AS high,
    MIN(price)                                   AS low,
    LAST(price, timestamp)                       AS close,
    SUM(quantity)                                AS volume,
    SUM(price * quantity) / NULLIF(SUM(quantity), 0) AS vwap,
    COUNT(*)                                     AS trade_count
FROM trades
GROUP BY bucket, symbol
WITH NO DATA;  -- don't backfill immediately

-- Policy: refresh the continuous aggregate every minute
SELECT add_continuous_aggregate_policy('ohlcv_1m',
    start_offset  => INTERVAL '1 hour',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute'
);

-- Query the continuous aggregate directly (pre-computed, fast)
SELECT * FROM ohlcv_1m
WHERE symbol = 'AAPL'
  AND bucket >= '2024-03-15 09:30:00'
  AND bucket <  '2024-03-15 16:00:00'
ORDER BY bucket;

-- Hierarchical aggregation: 5-minute bars from 1-minute bars
CREATE MATERIALIZED VIEW ohlcv_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', bucket)             AS bucket_5m,
    symbol,
    FIRST(open, bucket)                          AS open,
    MAX(high)                                    AS high,
    MIN(low)                                     AS low,
    LAST(close, bucket)                          AS close,
    SUM(volume)                                  AS volume,
    SUM(vwap * volume) / NULLIF(SUM(volume), 0)  AS vwap,
    SUM(trade_count)                             AS trade_count
FROM ohlcv_1m
GROUP BY bucket_5m, symbol
WITH NO DATA;
```

### 5.4 Python Integration

```python
"""
TimescaleDB client for tick data analysis.
Uses psycopg2 with connection pooling for high-throughput queries.
"""

import psycopg2
import psycopg2.pool
import pandas as pd
import numpy as np
from typing import Optional


class TickStore:
    """
    Client for a TimescaleDB tick data store.
    Supports ingestion, OHLCV queries, and VWAP computation.
    """

    def __init__(self, dsn: str, min_conn: int = 2, max_conn: int = 10):
        """
        Parameters
        ----------
        dsn : str
            PostgreSQL connection string
            e.g., 'postgresql://user:pass@localhost:5432/tickdb'
        """
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn, dsn
        )

    def ingest_batch(self, trades_df: pd.DataFrame) -> int:
        """
        Batch-ingest trades using COPY protocol (fastest path).

        Parameters
        ----------
        trades_df : pd.DataFrame
            Required columns: timestamp, symbol, price,
                              quantity, side, exchange, sequence_no

        Returns
        -------
        int
            Number of rows ingested
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Use COPY for maximum throughput (~500K rows/sec)
                from io import StringIO
                buf = StringIO()
                trades_df.to_csv(buf, index=False, header=False)
                buf.seek(0)

                cur.copy_expert(
                    """COPY trades (timestamp, symbol, price, quantity,
                       side, exchange, sequence_no)
                       FROM STDIN WITH CSV""",
                    buf
                )
            conn.commit()
            return len(trades_df)
        finally:
            self.pool.putconn(conn)

    def query_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1 minute",
    ) -> pd.DataFrame:
        """
        Query OHLCV bars from the continuous aggregate.
        Falls back to raw computation if the aggregate does not cover
        the requested interval.

        Parameters
        ----------
        symbol : str
            Instrument symbol
        start, end : str
            ISO 8601 timestamps
        interval : str
            Bar interval (e.g., '1 minute', '5 minutes', '1 hour')

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame indexed by timestamp
        """
        conn = self.pool.getconn()
        try:
            query = """
                SELECT
                    time_bucket(%s::interval, timestamp) AS bucket,
                    FIRST(price, timestamp)              AS open,
                    MAX(price)                           AS high,
                    MIN(price)                           AS low,
                    LAST(price, timestamp)               AS close,
                    SUM(quantity)                        AS volume,
                    SUM(price * quantity) /
                        NULLIF(SUM(quantity), 0)         AS vwap,
                    COUNT(*)                             AS trades
                FROM trades
                WHERE symbol = %s
                  AND timestamp >= %s::timestamptz
                  AND timestamp <  %s::timestamptz
                GROUP BY bucket
                ORDER BY bucket
            """
            df = pd.read_sql(
                query, conn,
                params=(interval, symbol, start, end),
                parse_dates=["bucket"],
            )
            return df.set_index("bucket")
        finally:
            self.pool.putconn(conn)

    def query_vwap(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> float:
        """
        Compute VWAP for a symbol in a time window.

        Returns
        -------
        float
            Volume-weighted average price
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT SUM(price * quantity) / NULLIF(SUM(quantity), 0)
                    FROM trades
                    WHERE symbol = %s
                      AND timestamp >= %s::timestamptz
                      AND timestamp <  %s::timestamptz
                """, (symbol, start, end))
                result = cur.fetchone()
                return float(result[0]) if result[0] is not None else float("nan")
        finally:
            self.pool.putconn(conn)

    def close(self):
        self.pool.closeall()
```

---

## 6. Apache Parquet & Arrow: Columnar Formats

### 6.1 Parquet: Columnar Storage on Disk

Apache Parquet is a columnar file format designed for analytical workloads. It is the de facto standard for storing large-scale financial data in data lakes (S3, HDFS, GCS).

**Physical layout:**

```mermaid
graph TB
    subgraph "Parquet File"
        M[Magic: PAR1]
        RG1[Row Group 1<br/>~128 MB]
        RG2[Row Group 2<br/>~128 MB]
        RG3[Row Group N<br/>...]
        FT[File Footer<br/>Schema + Statistics]
        M2[Magic: PAR1]
    end

    subgraph "Row Group Detail"
        C1[Column Chunk: timestamp<br/>Pages → DELTA_BINARY_PACKED]
        C2[Column Chunk: symbol<br/>Pages → PLAIN_DICTIONARY]
        C3[Column Chunk: price<br/>Pages → PLAIN + SNAPPY]
        C4[Column Chunk: quantity<br/>Pages → RLE + SNAPPY]
    end

    RG1 --> C1
    RG1 --> C2
    RG1 --> C3
    RG1 --> C4
```

Each **row group** contains column chunks, each compressed independently. This enables:

- **Column pruning:** A query that only needs `price` and `quantity` reads only those column chunks, skipping `timestamp` and `symbol` data entirely.
- **Predicate pushdown:** Statistics in the file footer (min/max per column per row group) allow the reader to skip entire row groups that cannot match a filter.
- **Encoding per column:** Each column uses the optimal encoding (dictionary for low-cardinality strings, delta for timestamps, RLE for repeated values).

### 6.2 Arrow: In-Memory Columnar Format

Apache Arrow defines an in-memory columnar data format with **zero-copy** reads. Arrow is the bridge between Parquet (on disk) and computation (in CPU):

- **Zero-copy IPC:** Arrow buffers can be shared between processes via memory-mapped files without serialization.
- **Language-agnostic:** C++, Python (PyArrow), Java, Rust implementations share the same memory layout.
- **SIMD-friendly:** Arrays are aligned to 64-byte boundaries, enabling AVX-512 vectorized operations directly on Arrow buffers.

### 6.3 Python: Reading and Writing Tick Data with Parquet/Arrow

```python
"""
Tick data storage and retrieval using Apache Parquet + Arrow.
Optimized for analytical workloads: OHLCV computation, VWAP,
and cross-symbol analysis.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# ============ Schema Definition ============

TICK_SCHEMA = pa.schema([
    pa.field("timestamp",   pa.timestamp("ns", tz="UTC")),
    pa.field("symbol",      pa.dictionary(pa.int16(), pa.string())),
    pa.field("price",       pa.float64()),
    pa.field("quantity",    pa.int32()),
    pa.field("side",        pa.dictionary(pa.int8(), pa.string())),
    pa.field("exchange",    pa.dictionary(pa.int8(), pa.string())),
    pa.field("sequence_no", pa.int64()),
])


def write_daily_parquet(
    trades_df: pd.DataFrame,
    base_path: str,
    date: str,
) -> str:
    """
    Write a day's trades to a Parquet file with optimized encoding.

    Partitions by date, sorted by (symbol, timestamp) within
    each partition for optimal query performance.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade data for a single day
    base_path : str
        Root directory for Parquet files
    date : str
        Date string (e.g., '2024-03-15')

    Returns
    -------
    str
        Path to the written Parquet file
    """
    # Sort for optimal predicate pushdown
    trades_df = trades_df.sort_values(["symbol", "timestamp"])

    table = pa.Table.from_pandas(trades_df, schema=TICK_SCHEMA)

    output_path = Path(base_path) / f"date={date}" / "trades.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        table,
        str(output_path),
        compression="zstd",
        compression_level=3,
        row_group_size=1_000_000,     # ~1M rows per row group
        use_dictionary=["symbol", "side", "exchange"],
        write_statistics=True,         # min/max for predicate pushdown
    )

    return str(output_path)


def read_trades_arrow(
    base_path: str,
    symbols: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    columns: Optional[list[str]] = None,
) -> pa.Table:
    """
    Read trades from partitioned Parquet dataset with predicate pushdown.

    Parameters
    ----------
    base_path : str
        Root directory of Parquet dataset
    symbols : list[str], optional
        Filter by symbols
    start, end : str, optional
        ISO 8601 timestamp filters
    columns : list[str], optional
        Columns to read (column pruning)

    Returns
    -------
    pa.Table
        Arrow Table (zero-copy if memory-mapped)
    """
    dataset = ds.dataset(
        base_path,
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([pa.field("date", pa.string())]),
            flavor="hive",
        ),
    )

    # Build filter expression for predicate pushdown
    filter_expr = None
    if symbols:
        sym_filter = ds.field("symbol").isin(symbols)
        filter_expr = sym_filter if filter_expr is None \
            else filter_expr & sym_filter
    if start:
        ts_start = ds.field("timestamp") >= pa.scalar(
            pd.Timestamp(start, tz="UTC"), type=pa.timestamp("ns", tz="UTC")
        )
        filter_expr = ts_start if filter_expr is None \
            else filter_expr & ts_start
    if end:
        ts_end = ds.field("timestamp") < pa.scalar(
            pd.Timestamp(end, tz="UTC"), type=pa.timestamp("ns", tz="UTC")
        )
        filter_expr = ts_end if filter_expr is None \
            else filter_expr & ts_end

    return dataset.to_table(
        filter=filter_expr,
        columns=columns,
    )


def compute_vwap_arrow(table: pa.Table) -> dict[str, float]:
    """
    Compute VWAP per symbol using Arrow compute functions (zero-copy).

    Parameters
    ----------
    table : pa.Table
        Arrow Table with 'symbol', 'price', 'quantity' columns

    Returns
    -------
    dict[str, float]
        Symbol -> VWAP mapping
    """
    # Group by symbol
    grouped = table.group_by("symbol")

    # For VWAP, we need sum(price * quantity) / sum(quantity)
    # Arrow compute supports this via expression-based aggregation
    pq_col = pc.multiply(table.column("price"),
                         pc.cast(table.column("quantity"), pa.float64()))
    table = table.append_column("pq", pq_col)

    result = table.group_by("symbol").aggregate([
        ("pq", "sum"),
        ("quantity", "sum"),
    ])

    vwap_dict = {}
    for i in range(result.num_rows):
        sym = result.column("symbol")[i].as_py()
        pq_sum = result.column("pq_sum")[i].as_py()
        q_sum = result.column("quantity_sum")[i].as_py()
        vwap_dict[sym] = pq_sum / q_sum if q_sum > 0 else float("nan")

    return vwap_dict


def compute_ohlcv_bars(
    table: pa.Table,
    bar_seconds: int = 60,
) -> pd.DataFrame:
    """
    Compute OHLCV bars from an Arrow Table of trades.
    Converts to pandas for groupby (Arrow groupby lacks FIRST/LAST).

    Parameters
    ----------
    table : pa.Table
        Trade data with timestamp, symbol, price, quantity columns
    bar_seconds : int
        Bar duration in seconds (default: 60 for 1-minute bars)

    Returns
    -------
    pd.DataFrame
        OHLCV bars indexed by (bucket, symbol)
    """
    df = table.to_pandas()

    # Floor timestamp to bar boundary
    df["bucket"] = df["timestamp"].dt.floor(f"{bar_seconds}s")

    bars = df.groupby(["bucket", "symbol"]).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        trade_count=("price", "count"),
    )

    # Add VWAP
    pq = df.groupby(["bucket", "symbol"]).apply(
        lambda g: np.average(g["price"], weights=g["quantity"]),
        include_groups=False,
    )
    bars["vwap"] = pq

    return bars
```

### 6.4 Parquet Compression Benchmarks

For a representative tick data file (500M trades, 10 columns):

| Compression | File Size | Write Speed | Read Speed | Compression Ratio |
|---|---|---|---|---|
| None | 32 GB | 1.2 GB/s | 3.5 GB/s | 1.0x |
| Snappy | 8.1 GB | 950 MB/s | 3.2 GB/s | 3.9x |
| ZSTD (level 1) | 5.4 GB | 800 MB/s | 2.8 GB/s | 5.9x |
| ZSTD (level 3) | 4.2 GB | 600 MB/s | 2.8 GB/s | 7.6x |
| ZSTD (level 9) | 3.8 GB | 200 MB/s | 2.8 GB/s | 8.4x |
| Gzip (level 6) | 4.5 GB | 150 MB/s | 1.5 GB/s | 7.1x |

**Recommendation:** ZSTD level 3 provides the best balance of compression ratio and write throughput for tick data. Read speed is dominated by I/O, not decompression, since ZSTD decompresses faster than NVMe read bandwidth in most configurations.

---

## 7. ClickHouse: OLAP at Scale

### 7.1 Architecture

ClickHouse is an open-source column-oriented OLAP database designed for analytical queries on large datasets. Originally developed at Yandex, it excels at:

- **Vectorized execution:** Processes data in batches of ~8,192 rows at a time, exploiting SIMD instructions.
- **MergeTree engine family:** LSM-tree-inspired storage with background merges, sparse primary index, and data skipping indices.
- **Materialized views:** Real-time aggregation pipelines that update as data is inserted.
- **Approximate queries:** HyperLogLog, quantile sketches, and sampling for sub-second responses on billion-row tables.

### 7.2 Table Design

```sql
-- ClickHouse trade table with MergeTree engine
CREATE TABLE trades (
    timestamp   DateTime64(9, 'UTC'),
    symbol      LowCardinality(String),  -- dictionary-encoded
    price       Float64,
    quantity    UInt32,
    side        Enum8('B' = 1, 'S' = 2),
    exchange    LowCardinality(String),
    sequence_no UInt64
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)    -- monthly partitions
ORDER BY (symbol, timestamp)         -- primary key (sparse index)
TTL toDateTime(timestamp) + INTERVAL 5 YEAR DELETE
SETTINGS index_granularity = 8192;

-- Materialized view: auto-compute 1-minute bars on insert
CREATE MATERIALIZED VIEW ohlcv_1m_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(bucket)
ORDER BY (symbol, bucket)
AS SELECT
    toStartOfMinute(timestamp) AS bucket,
    symbol,
    argMinState(price, timestamp)   AS open_state,
    maxState(price)                 AS high_state,
    minState(price)                 AS low_state,
    argMaxState(price, timestamp)   AS close_state,
    sumState(quantity)              AS volume_state,
    sumState(toFloat64(price) * toFloat64(quantity)) AS pq_state,
    countState(*)                   AS count_state
FROM trades
GROUP BY bucket, symbol;

-- Query the materialized view
SELECT
    bucket,
    symbol,
    argMinMerge(open_state)                       AS open,
    maxMerge(high_state)                          AS high,
    minMerge(low_state)                           AS low,
    argMaxMerge(close_state)                      AS close,
    sumMerge(volume_state)                        AS volume,
    sumMerge(pq_state) / sumMerge(volume_state)   AS vwap,
    countMerge(count_state)                       AS trades
FROM ohlcv_1m_mv
WHERE symbol = 'AAPL'
  AND bucket >= '2024-03-15 09:30:00'
  AND bucket <  '2024-03-15 16:00:00'
GROUP BY bucket, symbol
ORDER BY bucket;
```

### 7.3 Approximate Queries

For exploratory analysis where exact results are unnecessary, ClickHouse provides sketch-based aggregations:

```sql
-- Approximate distinct count (HyperLogLog, ~2% error, 100x faster)
SELECT uniqHLL12(symbol) AS approx_symbols
FROM trades
WHERE timestamp >= '2024-03-15';

-- Approximate quantiles (t-digest sketch)
SELECT symbol,
       quantilesExactExclusive(0.5, 0.95, 0.99)(price) AS price_quantiles
FROM trades
WHERE symbol = 'AAPL' AND toDate(timestamp) = '2024-03-15'
GROUP BY symbol;

-- Sampling: read only 10% of data for fast estimation
SELECT symbol,
       avg(price)    AS avg_price,
       sum(quantity) AS est_volume
FROM trades
SAMPLE 0.1
WHERE toDate(timestamp) = '2024-03-15'
GROUP BY symbol;
```

### 7.4 ClickHouse vs. kdb+ vs. QuestDB

| Feature | kdb+ | QuestDB | TimescaleDB | ClickHouse |
|---|---|---|---|---|
| Query language | q (APL-derived) | SQL (extended) | SQL (PostgreSQL) | SQL (extended) |
| Storage model | Column files, mmap | Column files, mmap | PostgreSQL heap + compression | MergeTree (LSM-like) |
| Write throughput | ~5M rows/s | ~3M rows/s (ILP) | ~500K rows/s (COPY) | ~2M rows/s |
| Scan throughput | ~2B rows/s | ~1.5B rows/s | ~200M rows/s | ~1B rows/s |
| License | Commercial ($$$) | Apache 2.0 | Apache 2.0 (community) | Apache 2.0 |
| Best use case | Real-time + historical | Time-series SQL | Existing PG ecosystem | Large-scale OLAP |
| ASOF joins | Native (aj, wj) | ASOF JOIN | Manual (lateral) | asof JOIN |
| Streaming agg | Continuous via IPC | Limited | Continuous aggregates | Materialized views |

---

## 8. Data Modeling for Tick Data

### 8.1 Tick Schema Design

The canonical tick schema captures the essential fields of a market data event:

```sql
-- Universal trade schema
CREATE TABLE trades (
    recv_timestamp  TIMESTAMPTZ NOT NULL,   -- when our system received it
    exch_timestamp  TIMESTAMPTZ NOT NULL,   -- when the exchange generated it
    symbol          TEXT        NOT NULL,    -- normalized ticker
    price           BIGINT      NOT NULL,   -- price in smallest unit (avoid float)
    quantity        INTEGER     NOT NULL,
    side            CHAR(1),                -- 'B', 'S', or NULL (undisclosed)
    trade_condition TEXT,                    -- e.g., '@', 'F' (FormT)
    exchange        TEXT        NOT NULL,
    sequence_no     BIGINT      NOT NULL,
    feed_id         SMALLINT    NOT NULL     -- identifies the source feed
);

-- Universal quote (BBO) schema
CREATE TABLE quotes (
    recv_timestamp  TIMESTAMPTZ NOT NULL,
    exch_timestamp  TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    bid_price       BIGINT,                 -- NULL if no bid
    bid_size        INTEGER,
    ask_price       BIGINT,                 -- NULL if no ask
    ask_size        INTEGER,
    exchange        TEXT        NOT NULL,
    sequence_no     BIGINT      NOT NULL,
    nbbo_flag       BOOLEAN DEFAULT FALSE   -- is this the NBBO?
);
```

**Price representation.** Storing prices as `BIGINT` in the smallest currency unit (e.g., 100ths of a cent for USD equities, where \$175.50 = 17550000) avoids floating-point rounding errors. The conversion factor is stored as metadata:

$$P_{\text{display}} = P_{\text{stored}} \times 10^{-\text{exponent}}$$

### 8.2 Symbol Universe Management

A symbol universe table tracks the mapping between exchange-specific identifiers and normalized symbols:

```sql
CREATE TABLE symbol_universe (
    symbol          TEXT        PRIMARY KEY,
    exchange_symbol TEXT        NOT NULL,     -- exchange-native symbol
    exchange        TEXT        NOT NULL,
    asset_class     TEXT        NOT NULL,     -- 'equity', 'future', 'option'
    currency        CHAR(3)    NOT NULL,
    tick_size       BIGINT     NOT NULL,      -- minimum price increment
    lot_size        INTEGER    NOT NULL,      -- minimum quantity increment
    price_exponent  SMALLINT   NOT NULL,      -- decimal places in stored price
    active_from     DATE       NOT NULL,
    active_to       DATE,                     -- NULL if still active
    isin            CHAR(12),
    sedol           CHAR(7)
);
```

### 8.3 Corporate Action Adjustments

Stock splits, dividends, mergers, and symbol changes require retroactive adjustments to historical data. Rather than modifying immutable tick data, maintain an adjustment factor table:

```sql
CREATE TABLE adjustments (
    symbol          TEXT        NOT NULL,
    effective_date  DATE        NOT NULL,
    action_type     TEXT        NOT NULL,    -- 'SPLIT', 'DIVIDEND', 'RENAME'

    -- For splits: cumulative adjustment factor
    -- Adjusted_price = raw_price * cum_factor
    cum_factor      DOUBLE PRECISION NOT NULL DEFAULT 1.0,

    -- For renames
    old_symbol      TEXT,
    new_symbol      TEXT,

    PRIMARY KEY (symbol, effective_date, action_type)
);

-- Example: AAPL 4:1 split on 2020-08-28
INSERT INTO adjustments VALUES
    ('AAPL', '2020-08-28', 'SPLIT', 0.25, NULL, NULL);

-- Query adjusted prices
SELECT t.exch_timestamp,
       t.symbol,
       t.price * COALESCE(a.cum_factor, 1.0) AS adjusted_price,
       t.quantity / COALESCE(a.cum_factor, 1.0) AS adjusted_quantity
FROM trades t
LEFT JOIN adjustments a
  ON t.symbol = a.symbol
 AND a.effective_date = (
     SELECT MAX(effective_date) FROM adjustments a2
     WHERE a2.symbol = t.symbol
       AND a2.effective_date <= t.exch_timestamp::date
       AND a2.action_type = 'SPLIT'
 );
```

---

## 9. Ingestion Pipelines

### 9.1 Architecture

A production tick data ingestion pipeline handles normalization, deduplication, and routing:

```mermaid
graph LR
    A[Exchange Feed A<br/>ITCH] --> H1[Feed Handler A<br/>Parse + Normalize]
    B[Exchange Feed B<br/>PITCH] --> H2[Feed Handler B<br/>Parse + Normalize]
    C[Exchange Feed C<br/>OPRA] --> H3[Feed Handler C<br/>Parse + Normalize]

    H1 --> MQ[Message Queue<br/>Aeron / Kafka]
    H2 --> MQ
    H3 --> MQ

    MQ --> D1[Deduplication<br/>Sequence Check]
    D1 --> V1[Validation<br/>Schema + Ranges]
    V1 --> W1[Writer<br/>Batch + Flush]

    W1 --> DB[("Time-Series DB")]
    W1 --> PQ[("Parquet Archive")]

    style MQ fill:#f96,stroke:#333
```

### 9.2 Normalization

Each exchange uses different protocols (ITCH, PITCH, OPRA, CME MDP 3.0) with different field encodings. The feed handler normalizes messages to a common internal format:

```python
"""
Feed handler normalization layer.
Converts exchange-specific messages to a canonical tick format.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import struct


class Side(Enum):
    BUY = "B"
    SELL = "S"
    UNKNOWN = "U"


@dataclass(frozen=True, slots=True)
class NormalizedTrade:
    """Canonical trade format, independent of exchange protocol."""
    recv_timestamp_ns: int          # nanoseconds since epoch (our clock)
    exch_timestamp_ns: int          # nanoseconds since epoch (exchange clock)
    symbol: str                     # normalized symbol (e.g., 'AAPL')
    price: int                      # integer price in smallest unit
    quantity: int
    side: Side
    exchange: str                   # MIC code (e.g., 'XNAS', 'XNYS')
    sequence_no: int
    trade_condition: Optional[str]
    price_exponent: int             # number of decimal places


class ITCHNormalizer:
    """
    Normalize NASDAQ ITCH 5.0 trade messages.
    Trade message type 'P' (Non-Cross Trade).
    """

    # ITCH Trade message layout (type 'P'):
    # Offset 0:  message_type (1 byte)
    # Offset 1:  stock_locate (2 bytes)
    # Offset 3:  tracking_number (2 bytes)
    # Offset 5:  timestamp (6 bytes, ns since midnight)
    # Offset 11: order_ref (8 bytes)
    # Offset 19: buy_sell (1 byte, 'B' or 'S')
    # Offset 20: shares (4 bytes)
    # Offset 24: stock (8 bytes, right-padded with spaces)
    # Offset 32: price (4 bytes, fixed-point 4 decimal places)
    # Offset 36: match_number (8 bytes)

    TRADE_FORMAT = ">xHHq8sBcI8sI8s"  # simplified

    def __init__(self, symbol_map: dict[int, str], date: str):
        """
        Parameters
        ----------
        symbol_map : dict[int, str]
            Maps stock_locate ID to normalized symbol name
        date : str
            Trading date (YYYY-MM-DD) for converting midnight-offset timestamps
        """
        self.symbol_map = symbol_map
        self.midnight_ns = int(
            datetime.strptime(date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp() * 1_000_000_000
        )

    def normalize_trade(
        self,
        raw: bytes,
        recv_ns: int
    ) -> Optional[NormalizedTrade]:
        """
        Parse an ITCH trade message (type 'P') and normalize.

        Parameters
        ----------
        raw : bytes
            Raw ITCH message bytes (44 bytes for type 'P')
        recv_ns : int
            Receive timestamp in nanoseconds

        Returns
        -------
        NormalizedTrade or None if not a trade message
        """
        if raw[0:1] != b"P":
            return None

        stock_locate = int.from_bytes(raw[1:3], "big")

        # 6-byte timestamp: nanoseconds since midnight
        ts_bytes = b"\x00\x00" + raw[5:11]
        ts_ns = int.from_bytes(ts_bytes, "big")
        exch_ns = self.midnight_ns + ts_ns

        buy_sell = raw[19:20]
        shares = int.from_bytes(raw[20:24], "big")
        stock = raw[24:32].decode("ascii").rstrip()

        # Price: 4 bytes, 4 implied decimal places
        raw_price = int.from_bytes(raw[32:36], "big")

        symbol = self.symbol_map.get(stock_locate, stock)
        side = Side.BUY if buy_sell == b"B" else Side.SELL

        return NormalizedTrade(
            recv_timestamp_ns=recv_ns,
            exch_timestamp_ns=exch_ns,
            symbol=symbol,
            price=raw_price,          # already in 4-decimal fixed point
            quantity=shares,
            side=side,
            exchange="XNAS",
            sequence_no=stock_locate,  # simplified
            trade_condition=None,
            price_exponent=4,
        )
```

### 9.3 Deduplication

Market data feeds frequently deliver duplicate messages (retransmissions, overlapping multicast groups). Deduplication uses sequence numbers:

```python
class SequenceDeduplicator:
    """
    Detect and filter duplicate messages using per-symbol sequence tracking.
    Uses a ring buffer of recent sequence numbers for space efficiency.
    """

    def __init__(self, window_size: int = 1024):
        self.window_size = window_size
        # Per-symbol: highest seen sequence number
        self._high_water: dict[str, int] = {}
        # Per-symbol: bitmap of seen sequences in the window
        self._seen: dict[str, bytearray] = {}
        self._dup_count = 0
        self._total_count = 0

    def is_duplicate(self, symbol: str, sequence_no: int) -> bool:
        """
        Check if this (symbol, sequence_no) has been seen before.

        Returns True if duplicate (should be dropped).
        """
        self._total_count += 1

        hw = self._high_water.get(symbol, -1)

        if sequence_no <= hw - self.window_size:
            # Very old sequence: assume duplicate
            self._dup_count += 1
            return True

        if sequence_no > hw:
            # New high water: shift window
            self._high_water[symbol] = sequence_no
            if symbol not in self._seen:
                self._seen[symbol] = bytearray(self.window_size // 8 + 1)
            # Clear bits for new range
            return False

        # Within window: check bitmap
        offset = hw - sequence_no
        byte_idx = offset // 8
        bit_idx = offset % 8

        seen = self._seen[symbol]
        if byte_idx < len(seen) and (seen[byte_idx] >> bit_idx) & 1:
            self._dup_count += 1
            return True

        # Mark as seen
        if byte_idx < len(seen):
            seen[byte_idx] |= (1 << bit_idx)
        return False

    @property
    def duplicate_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._dup_count / self._total_count
```

---

## 10. Query Patterns

### 10.1 OHLCV Bar Aggregation

The most common query pattern. Already covered in detail per database (Sections 3--7). The general approach:

$$
\begin{aligned}
\text{Open}   &= P_{\text{first}} \quad \text{(first price in the window)} \\
\text{High}   &= \max(P_i) \\
\text{Low}    &= \min(P_i) \\
\text{Close}  &= P_{\text{last}} \quad \text{(last price in the window)} \\
\text{Volume} &= \sum Q_i
\end{aligned}
$$

### 10.2 VWAP

$$\text{VWAP}(t_1, t_2) = \frac{\sum_{i: t_1 \leq t_i \leq t_2} P_i \cdot Q_i}{\sum_{i: t_1 \leq t_i \leq t_2} Q_i}$$

VWAP is the benchmark for execution quality. An execution algorithm that achieves a price better than VWAP (buy below, sell above) has added alpha.

### 10.3 Order Flow Imbalance

Order flow imbalance measures the asymmetry between buy and sell pressure:

$$\text{OFI}(t_1, t_2) = \frac{\sum_{i \in \text{buys}} Q_i - \sum_{i \in \text{sells}} Q_i}{\sum_{i \in \text{buys}} Q_i + \sum_{i \in \text{sells}} Q_i}$$

```sql
-- Order flow imbalance in 1-minute buckets (QuestDB syntax)
SELECT timestamp, symbol,
       SUM(CASE WHEN side = 'B' THEN quantity ELSE 0 END) AS buy_vol,
       SUM(CASE WHEN side = 'S' THEN quantity ELSE 0 END) AS sell_vol,
       CAST(
           SUM(CASE WHEN side = 'B' THEN quantity ELSE -quantity END) AS DOUBLE
       ) / NULLIF(SUM(quantity), 0)  AS ofi
FROM trades
WHERE symbol = 'AAPL'
  AND timestamp IN '2024-03-15'
SAMPLE BY 1m
ALIGN TO CALENDAR;
```

```q
/ Order flow imbalance in q (kdb+)
ofi: select
    buy_vol:  sum size where side=`B,
    sell_vol: sum size where side=`S,
    ofi: {(sum x where y=`B) - (sum x where y=`S)} [size;side]
         % sum size
  by sym, 1 xbar time.minute
  from trades where sym=`AAPL
```

### 10.4 ASOF Joins: Aligning Quote and Trade Data

ASOF joins match each trade with the most recent quote at or before the trade timestamp. This is essential for computing effective spread, trade classification (Lee-Ready algorithm), and execution analysis.

```q
/ kdb+ asof join: attach prevailing quote to each trade
trade_with_quote: aj[`sym`time; trades; quotes]

/ Now compute effective spread
/ Effective spread = 2 * |trade_price - midpoint| * direction
/ where midpoint = (bid + ask) / 2
enriched: update
    mid: (bid + ask) % 2,
    eff_spread: 2 * abs[price - (bid + ask) % 2]
  from trade_with_quote
```

```sql
-- QuestDB ASOF JOIN
SELECT t.timestamp, t.symbol, t.price, t.quantity,
       q.bid_price, q.ask_price,
       (q.bid_price + q.ask_price) / 2.0 AS midpoint,
       2.0 * ABS(t.price - (q.bid_price + q.ask_price) / 2.0) AS eff_spread
FROM trades t
ASOF JOIN quotes q ON (t.symbol = q.symbol)
WHERE t.symbol = 'AAPL'
  AND t.timestamp IN '2024-03-15';
```

---

## 11. Archival Strategies

### 11.1 Hot/Warm/Cold Tiering

A tiered storage architecture balances query performance against storage cost:

```mermaid
graph TB
    subgraph "Hot Tier (Today + Yesterday)"
        H[In-Memory / NVMe SSD<br/>Uncompressed columns<br/>Full indexes<br/>~$5/GB/month]
    end

    subgraph "Warm Tier (Last 30 days)"
        W[NVMe SSD<br/>Compressed (ZSTD)<br/>Sparse indexes<br/>~$0.50/GB/month]
    end

    subgraph "Cold Tier (30 days - 5 years)"
        C[Object Storage S3/GCS<br/>Parquet + ZSTD<br/>Predicate pushdown only<br/>~$0.02/GB/month]
    end

    subgraph "Archive (5+ years)"
        A[Glacier / Coldline<br/>Parquet + ZSTD level 19<br/>Batch retrieval only<br/>~$0.004/GB/month]
    end

    H -->|"Daily roll"| W
    W -->|"Monthly roll"| C
    C -->|"Annual roll"| A
```

### 11.2 Compression Ratios by Tier

| Tier | Format | Compression | Ratio | Query Latency |
|---|---|---|---|---|
| Hot | Native DB (column files) | None or LZ4 | 1--2x | < 100 ms |
| Warm | Native DB (compressed chunks) | ZSTD level 3 | 5--8x | 200 ms -- 1 s |
| Cold | Parquet | ZSTD level 9 | 8--12x | 2 -- 30 s |
| Archive | Parquet | ZSTD level 19 | 10--15x | Minutes (retrieval) |

### 11.3 Retention Policies

```sql
-- TimescaleDB: auto-compress after 7 days, drop after 2 years
SELECT add_compression_policy('trades', INTERVAL '7 days');
SELECT add_retention_policy('trades', INTERVAL '2 years');

-- For cold/archive, export to Parquet on a schedule:
-- (Run nightly via cron or Airflow)
-- 1. Export yesterday's data to Parquet
-- 2. Upload to S3 with lifecycle rules
-- 3. Verify checksums
-- 4. Drop from hot tier
```

```python
"""
Daily archival job: export completed partitions to Parquet on S3.
"""

import pyarrow.parquet as pq
import pyarrow as pa
import boto3
from datetime import date, timedelta


def archive_day(
    tick_store,  # TickStore from Section 5.4
    archive_date: date,
    s3_bucket: str,
    s3_prefix: str = "tick-archive",
) -> dict:
    """
    Archive one day of tick data to Parquet on S3.

    Returns
    -------
    dict
        Summary: row count, compressed size, S3 key
    """
    start = archive_date.isoformat()
    end = (archive_date + timedelta(days=1)).isoformat()

    # Read full day from database
    conn = tick_store.pool.getconn()
    try:
        query = """
            SELECT recv_timestamp, exch_timestamp, symbol,
                   price, quantity, side, exchange, sequence_no
            FROM trades
            WHERE recv_timestamp >= %s::timestamptz
              AND recv_timestamp <  %s::timestamptz
            ORDER BY symbol, recv_timestamp
        """
        table = pa.Table.from_pandas(
            pd.read_sql(query, conn, params=(start, end))
        )
    finally:
        tick_store.pool.putconn(conn)

    # Write to local Parquet (ZSTD level 9 for archive)
    local_path = f"/tmp/trades_{archive_date}.parquet"
    pq.write_table(
        table, local_path,
        compression="zstd",
        compression_level=9,
        row_group_size=1_000_000,
    )

    # Upload to S3
    s3_key = f"{s3_prefix}/date={archive_date}/trades.parquet"
    s3 = boto3.client("s3")
    s3.upload_file(local_path, s3_bucket, s3_key)

    import os
    compressed_size = os.path.getsize(local_path)
    os.remove(local_path)

    return {
        "date": str(archive_date),
        "rows": table.num_rows,
        "compressed_bytes": compressed_size,
        "s3_key": s3_key,
    }
```

---

## 12. Exercises

**Exercise 1: kdb+ TWAP Implementation.**
Implement Time-Weighted Average Price (TWAP) in q. Unlike VWAP which weights by volume, TWAP weights by time duration:

$$\text{TWAP}(t_1, t_2) = \frac{1}{t_2 - t_1} \int_{t_1}^{t_2} P(t) \, dt \approx \sum_{i=1}^{n-1} P_i \cdot \frac{t_{i+1} - t_i}{t_n - t_1}$$

where $P(t)$ is the last-traded price at time $t$. Compute TWAP for all symbols in 5-minute windows and compare with VWAP. Under what market conditions do TWAP and VWAP diverge significantly?

**Exercise 2: TimescaleDB Continuous Aggregate Hierarchy.**
Create a three-level continuous aggregate hierarchy: 1-second bars -> 1-minute bars -> 1-hour bars. Measure the refresh latency at each level when 1 million new trades are inserted. Compare the query latency of reading hourly bars from the continuous aggregate vs. computing them from raw trades.

**Exercise 3: Parquet Predicate Pushdown Benchmark.**
Write a benchmark that measures query performance with and without predicate pushdown on a Parquet dataset of 100 million trades across 1,000 symbols. Queries:
- Single-symbol lookup (1 symbol out of 1,000)
- Time-range scan (1 hour out of 24 hours)
- Combined symbol + time filter
Report the number of row groups read, bytes scanned, and wall-clock time for each query, with and without predicate pushdown enabled.

**Exercise 4: Schema Evolution.**
Your tick data schema needs to add a new field (`lit_dark` flag indicating lit vs. dark pool execution). Design a migration strategy that:
- Does not require rewriting existing Parquet files.
- Handles the new field being NULL for historical data.
- Works across all three databases (kdb+, QuestDB, TimescaleDB).
Implement the migration and verify that existing queries continue to work.

**Exercise 5: Ingestion Pipeline Load Test.**
Build an ingestion pipeline that can sustain 5 million messages/second into QuestDB:
- Generate synthetic ITCH-like messages with realistic symbol distribution (Zipf, $\alpha = 1.1$).
- Measure end-to-end latency (message generation to queryable in database).
- Identify the bottleneck: is it serialization, network, or database write path?
- Optimize until you achieve the 5M msg/s target (hint: batch ILP writes, use multiple sender threads).

**Exercise 6: Corporate Action Backtest.**
Download historical AAPL price data spanning the 2020 4:1 stock split (2020-08-28). Implement split-adjusted price queries that:
- Correctly adjust pre-split prices by the cumulative split factor.
- Handle the case where a backtest window straddles the split date.
- Verify that a VWAP computed over a window crossing the split date is correct using adjusted prices.

**Exercise 7: Cross-Database Query.**
Design an architecture where hot data (today) lives in QuestDB for real-time queries, warm data (last 30 days) lives in TimescaleDB for analytical queries, and cold data (30+ days) lives in Parquet on S3. Write a query federation layer in Python that:
- Accepts a (symbol, start_time, end_time) query.
- Determines which tier(s) to query based on the time range.
- Queries each tier in parallel and merges results.
- Handles overlap between tiers gracefully (deduplication by sequence number).

**Exercise 8: Compression Deep Dive.**
For a 1 GB tick data file (trades for a single day, single exchange), measure:
- Dictionary encoding compression ratio for the symbol column.
- Delta encoding ratio for the timestamp column.
- ZSTD compression at levels 1, 3, 9, 19 for each column independently.
- Total file size with all techniques combined.
Plot compression ratio vs. decompression throughput for each configuration. What is the Pareto-optimal set of configurations?

---

## 13. Summary

This module surveyed the database systems and storage technologies used to manage tick data in quantitative finance. The central challenge is the tension between **write throughput** (millions of messages per second from exchange feeds), **query diversity** (from point lookups to multi-year aggregations), and **storage efficiency** (petabytes of historical data).

**Technology selection guide:**

| Scenario | Recommended System | Rationale |
|---|---|---|
| Greenfield, budget unconstrained | kdb+ | Fastest vectorized queries, battle-tested |
| SQL-first, time-series focused | QuestDB | kdb+-like performance with SQL, open-source |
| Existing PostgreSQL ecosystem | TimescaleDB | Familiar SQL, rich extensions, continuous aggregates |
| Large-scale analytics, approximation OK | ClickHouse | Massive scan throughput, materialized views |
| Data lake / archival | Apache Parquet + Arrow | Universal format, S3-native, language-agnostic |
| Real-time + research hybrid | QuestDB (hot) + Parquet (cold) | Fast ingestion with cheap archival |

**Concept map:**

```mermaid
graph TB
    subgraph "Storage Engines"
        KDB[kdb+/q<br/>Column-store<br/>Vector-oriented]
        QDB[QuestDB<br/>Column-store<br/>SQL + SIMD]
        TSDB[TimescaleDB<br/>PostgreSQL extension<br/>Hypertables]
        CH[ClickHouse<br/>MergeTree<br/>OLAP]
        PQ[Parquet + Arrow<br/>Columnar files<br/>Zero-copy]
    end

    subgraph "Data Pipeline"
        FH[Feed Handlers<br/>Parse + Normalize]
        DEDUP[Deduplication<br/>Sequence Tracking]
        INGEST[Batch Ingestion<br/>COPY / ILP]
    end

    subgraph "Query Patterns"
        OHLCV[OHLCV Bars]
        VWAP[VWAP]
        OFI[Order Flow Imbalance]
        ASOF[ASOF Joins]
    end

    subgraph "Data Management"
        SCHEMA[Schema Design<br/>Fixed-point prices]
        CA[Corporate Actions<br/>Split adjustment]
        TIER[Hot/Warm/Cold<br/>Tiered Storage]
    end

    FH --> DEDUP --> INGEST --> KDB & QDB & TSDB & CH
    KDB & QDB & TSDB & CH --> OHLCV & VWAP & OFI & ASOF
    CH & QDB --> PQ
    SCHEMA --> KDB & QDB & TSDB & CH
    CA --> SCHEMA
    TIER --> PQ

    style KDB fill:#f96,stroke:#333
    style PQ fill:#69b,stroke:#333,color:#fff
```

The database layer is the foundation of both research and production. A well-designed tick data infrastructure enables reproducible backtesting (exact historical replay), real-time signal computation (sub-second OHLCV and VWAP), and efficient storage (petabyte-scale at manageable cost).

---

*Next: [Module 16 — Distributed Systems & Message Queues](../computation/16_distributed_systems.md)*
