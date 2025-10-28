from sqlalchemy import text

def knn_cosine(db, qvec, k=5, model="mobilefacenet-192d"):
    qv = "[" + ",".join(str(float(x)) for x in qvec) + "]"
    sql = text("""
        SELECT emb.emp_id, e.full_name,
               (1 - (emb.embedding <#> CAST(:qv AS vector(192)))) AS score
        FROM embeddings emb
        JOIN employees e ON e.emp_id = emb.emp_id
        WHERE emb.model = :m
        ORDER BY score DESC
        LIMIT :k
    """)
    return db.execute(sql, {"qv": qv, "m": model, "k": k}).fetchall()
