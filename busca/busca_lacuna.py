"""Busca bibliográfica reprodutível que sustenta a lacuna do §2 (backdoor/envenenamento em AST e em sons ambientais).

Bases: arXiv (API), OpenAlex (API), Semantic Scholar (API bulk search). Período: 2017–2026.
Uso: python3 busca/busca_lacuna.py  -> grava busca/resultados_AAAA-MM-DD.json e imprime o resumo.
"""
import json, time, datetime, urllib.parse, urllib.request, xml.etree.ElementTree as ET

UA = {"User-Agent": "tcc-literature-search/1.0"}
ATK = ['backdoor', 'trojan', '"data poisoning"', '"poisoning attack"']
GROUPS = {
    "G1_AST": (ATK, ['"audio spectrogram transformer"']),
    "G2_som_ambiental": (ATK, ['"environmental sound"', '"sound event"', '"audio event"', '"acoustic scene"',
                               '"sound classification"', '"audio classification"']),
    "G3_conjuntos": (ATK, ['UrbanSound8K', '"ESC-50"', 'AudioSet', 'DCASE']),
    "G4_disparos": (ATK + ['adversarial'], ['gunshot', '"gun shot"']),
    "G5_mascaramento": (['backdoor', 'trojan'], ['psychoacoustic', '"auditory masking"']),
}


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            time.sleep(5 * (k + 1)); err = e
    raise err


def arxiv(a, b):
    f = lambda t: f'abs:{t}' if not t.startswith('"') else f'abs:{t}'
    q = f"({' OR '.join(f(t) for t in a)}) AND ({' OR '.join(f(t) for t in b)})"
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 200})
    root = ET.fromstring(get(url)); ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        y = int(e.find("a:published", ns).text[:4])
        if 2017 <= y <= 2026:
            out.append({"title": " ".join(e.find("a:title", ns).text.split()), "year": y,
                        "id": e.find("a:id", ns).text, "abstract": " ".join(e.find("a:summary", ns).text.split())})
    return q, out


def openalex(a, b):
    q = f"({' OR '.join(a)}) AND ({' OR '.join(b)})"
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "filter": f"title_and_abstract.search:{q},publication_year:2017-2026", "per-page": 200,
        "select": "id,doi,title,publication_year,primary_location,abstract_inverted_index"})
    d = json.loads(get(url)); out = []
    for w in d.get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((p, t) for t, ps in inv.items() for p in ps)
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        out.append({"title": w.get("title"), "year": w.get("publication_year"), "id": w.get("doi") or w["id"],
                    "venue": src, "abstract": " ".join(t for _, t in pos)})
    return q + f"  [total={d['meta']['count']}]", out


def s2(a, b):
    j = lambda xs: "(" + " | ".join(xs) + ")"
    q = f"{j(a)} + {j(b)}"
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode({
        "query": q, "fields": "title,year,venue,externalIds,abstract", "year": "2017-2026"})
    d = json.loads(get(url)); out = []
    for p in d.get("data", []):
        ext = p.get("externalIds") or {}
        out.append({"title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"),
                    "id": ext.get("DOI") or ext.get("ArXiv") or p.get("paperId"), "abstract": p.get("abstract") or ""})
    return q + f"  [total={d.get('total')}]", out


if __name__ == "__main__":
    today = datetime.date.today().isoformat(); res = {"data": today, "periodo": "2017-2026", "grupos": {}}
    for g, (a, b) in GROUPS.items():
        res["grupos"][g] = {}
        for name, fn in (("arXiv", arxiv), ("OpenAlex", openalex), ("SemanticScholar", s2)):
            try:
                q, items = fn(a, b)
            except Exception as e:
                q, items = f"ERRO: {e}", []
            res["grupos"][g][name] = {"consulta": q, "n": len(items), "itens": items}
            print(f"{g:18} {name:16} n={len(items):4}  {q[:110]}")
            time.sleep(4)
    json.dump(res, open(f"busca/resultados_{today}.json", "w"), ensure_ascii=False, indent=1)
