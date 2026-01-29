# Unofficial API for xAI's Grokipedia (not affiliated)
# Fork by utrumsit - adds Markdown output support
from api_analytics.fastapi import Analytics
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from bs4 import BeautifulSoup, NavigableString, Tag
import requests
import re
import urllib.parse
from datetime import datetime, timedelta
from urllib.parse import urljoin  # For absolute URLs
from pathlib import Path
import sys
from collections import defaultdict
import time
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Grokipedia API v0.4",
    description="Unofficial API for xAI's Grokipedia (not affiliated). Fork with Markdown support.",
    version="0.4.0-beta",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add FastAPI Analytics middleware with API key
app.add_middleware(Analytics, api_key=os.getenv("ANALYTICS_KEY") )

# Robust INDEX_PATH: Start from cwd, fallback to __file__ if needed
base_path = Path.cwd()
INDEX_PATH = base_path / "public" / "static" / "index.html"

# Fallback if cwd fails (rare in Vercel)
if not INDEX_PATH.exists():
    INDEX_PATH = Path(__file__).parent / "public" / "static" / "index.html"

# Conditional mount
static_dir = base_path / "public" / "static"
if static_dir.exists() and os.getenv("VERCEL") is None:
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory="public/static", html=True), name="static")

BASE_URL = "https://grokipedia.com"
_cache = {}
MAX_CACHE_SIZE = 1000  # Adjust as needed; keeps cache small (~50MB assuming avg 50KB/page)
CACHE_TTL = timedelta(days=2)

# Rate limiting setup
request_times = defaultdict(list)
RATE_LIMIT = 100  # Generous: 100 requests per window
RATE_WINDOW = 60  # 60 seconds (1 minute)

def rate_limit_dependency(request: Request):
    client_ip = request.client.host
    now = time.time()
    # Clean expired timestamps
    request_times[client_ip] = [t for t in request_times[client_ip] if now - t < RATE_WINDOW]
    if len(request_times[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT} requests per {RATE_WINDOW} seconds). Try again later."
        )
    request_times[client_ip].append(now)

class Reference(BaseModel):
    number: int
    url: str = ""

class Page(BaseModel):
    title: str
    slug: str
    url: str
    content_text: str  # Plain text (legacy)
    content_markdown: Optional[str] = None  # Markdown formatted
    char_count: int
    word_count: int
    references_count: int
    references: Optional[List[Reference]] = None


# HTML to Markdown conversion
BLOCK_TAGS = {'p', 'div', 'section', 'article', 'blockquote', 'li', 'tr', 'br', 'hr'}
HEADER_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
INLINE_EMPHASIS_TAGS = {'i', 'em'}
INLINE_STRONG_TAGS = {'b', 'strong'}
INLINE_CODE_TAGS = {'code', 'tt'}
SKIP_TAGS = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'sup', 'sub'}
# Tags whose content we want but that shouldn't add formatting (like spans, anchors to internal links)
TRANSPARENT_TAGS = {'span', 'a', 'font', 'u'}


def html_to_markdown(element: Tag, skip_metadata: bool = True) -> str:
    """
    Convert HTML element to clean Markdown, preserving inline formatting.

    - Headers become ## Header
    - <i>/<em> become *italic*
    - <b>/<strong> become **bold**
    - <a> links to other pages become *italic* (internal links are usually terms)
    - <a> links to external URLs become [text](url)
    - Block elements get paragraph breaks
    - Inline elements flow naturally with surrounding text
    """
    seen_title = False
    metadata_phrases = {'fact-checked by grok', 'weeks ago', 'days ago', 'hours ago', 'yesterday', 'month ago', 'months ago'}

    def process_node(node, in_paragraph=False):
        nonlocal seen_title

        if isinstance(node, NavigableString):
            text = str(node)
            # Collapse whitespace but preserve single spaces
            text = re.sub(r'\s+', ' ', text)
            # Skip metadata text nodes
            if skip_metadata and any(phrase in text.lower() for phrase in metadata_phrases):
                return ''
            return text

        if not isinstance(node, Tag):
            return ''

        tag_name = node.name.lower() if node.name else ''

        # Skip unwanted tags entirely
        if tag_name in SKIP_TAGS:
            return ''

        # Handle headers
        if tag_name in HEADER_TAGS:
            level = int(tag_name[1])
            header_text = node.get_text(strip=True)

            # Skip the first h1 (it's the title, we handle that separately)
            if tag_name == 'h1' and not seen_title:
                seen_title = True
                return ''

            # Skip metadata-like headers
            if any(phrase in header_text.lower() for phrase in metadata_phrases):
                return ''

            prefix = '#' * min(level, 6)
            return f'\n\n{prefix} {header_text}\n\n'

        # Handle block elements
        if tag_name in BLOCK_TAGS:
            inner = ''.join(process_node(child, in_paragraph=True) for child in node.children)
            inner = inner.strip()

            # Skip metadata lines
            if skip_metadata and any(phrase in inner.lower() for phrase in metadata_phrases):
                return ''

            if not inner:
                return ''

            if tag_name == 'li':
                return f'- {inner}\n'
            elif tag_name == 'blockquote':
                # Prefix each line with >
                quoted = '\n'.join(f'> {line}' for line in inner.split('\n'))
                return f'\n\n{quoted}\n\n'
            elif tag_name == 'br':
                return '\n'
            elif tag_name == 'hr':
                return '\n\n---\n\n'
            else:
                return f'\n\n{inner}\n\n'

        # Handle lists
        if tag_name in ('ul', 'ol'):
            inner = ''.join(process_node(child) for child in node.children)
            return f'\n\n{inner.strip()}\n\n'

        # Handle inline emphasis
        if tag_name in INLINE_EMPHASIS_TAGS:
            inner = ''.join(process_node(child, in_paragraph=True) for child in node.children)
            inner = inner.strip()
            if inner:
                return f'*{inner}*'
            return ''

        # Handle inline strong
        if tag_name in INLINE_STRONG_TAGS:
            inner = ''.join(process_node(child, in_paragraph=True) for child in node.children)
            inner = inner.strip()
            if inner:
                return f'**{inner}**'
            return ''

        # Handle inline code
        if tag_name in INLINE_CODE_TAGS:
            inner = ''.join(process_node(child, in_paragraph=True) for child in node.children)
            inner = inner.strip()
            if inner:
                return f'`{inner}`'
            return ''

        # Handle links
        if tag_name == 'a':
            href = node.get('href', '')
            inner = ''.join(process_node(child, in_paragraph=True) for child in node.children)
            inner = inner.strip()

            if not inner:
                return ''

            # Internal Grokipedia links - just use the text
            if href.startswith('/page/') or 'grokipedia.com' in href:
                return f'*{inner}*'  # Italicize internal links (they're usually terms)

            # External links - full markdown link
            if href.startswith(('http://', 'https://', '//')):
                return f'[{inner}]({href})'

            # No href or relative - just text
            return inner

        # Handle images
        if tag_name == 'img':
            alt = node.get('alt', '')
            src = node.get('src', '')
            if src:
                return f'![{alt}]({src})'
            return ''

        # Transparent tags and anything else - process children
        return ''.join(process_node(child, in_paragraph) for child in node.children)

    result = process_node(element)

    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)

    # Clean up spaces around newlines
    result = re.sub(r' +\n', '\n', result)
    result = re.sub(r'\n +', '\n', result)

    # Remove leading/trailing whitespace from lines while preserving paragraph structure
    lines = result.split('\n')
    lines = [line.strip() for line in lines]
    result = '\n'.join(lines)

    # Final cleanup of excessive newlines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()

def normalize_slug(input_str: str) -> str:
    # FastAPI and query params automatically decode %26 to &
    # Handle potential double-encoding from browser address bar (e.g., typing "at%26t" sends "at%2526t", decoded to "at%26t")
    input_str = urllib.parse.unquote(input_str)
    # Replace space with underscore
    normalized = input_str.replace(' ', '_')
    # Check if first letter is capitalized
    if not normalized[0].isupper():
        normalized = normalized.capitalize()
    return normalized

def find_content_div(soup: BeautifulSoup) -> Tag:
    """Find the main content div, handling Grokipedia's nested structure."""
    # Try article.prose first (older structure)
    div = soup.select_one('article.prose')
    if div:
        return div

    # Current structure: article > div (non-hidden with content)
    article = soup.find('article')
    if article:
        for child in article.children:
            if isinstance(child, Tag) and child.name == 'div':
                classes = child.get('class', [])
                if 'hidden' not in classes:
                    # Check it has substantial content
                    if len(child.get_text(strip=True)) > 100:
                        return child

    # Fallback selectors
    selectors = ['div.content', 'main', 'article']
    for sel in selectors:
        div = soup.select_one(sel)
        if div:
            return div

    return soup.body

def extract_references(soup: BeautifulSoup) -> tuple[List[Reference], int]:
    # First, try to find <div id="references">
    refs_div = soup.find('div', id='references')
    if refs_div:
        # Look for ol/ul inside refs_div
        ol = refs_div.find('ol') or refs_div.find('ul')
        if ol:
            list_items = ol.find_all('li', recursive=True)
        else:
            # Fallback: direct li children of div
            list_items = refs_div.find_all('li', recursive=True)
    else:
        # Fallback: search entire soup for ol with references/citations class
        ol = soup.find('ol', class_=re.compile(r'references?|citations?', re.I))
        if ol:
            list_items = ol.find_all('li', recursive=True)
        else:
            return [], 0
    
    references = []
    for i, li in enumerate(list_items, 1):
        # Extract hrefs that are http or //, take first as singular URL
        urls = []
        for a in li.find_all('a', href=True):
            href = a.get('href')
            if href and (href.startswith(('http', '//'))):
                urls.append(urljoin(BASE_URL, href))
        url = urls[0] if urls else ""
        references.append(Reference(number=i, url=url))
    
    return references, len(references)

def get_size(obj, seen=None):
    """Recursively find size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(k, seen) + get_size(v, seen) for k, v in obj.items()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, tuple)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def get_cache_size_bytes():
    """Calculate total memory size of the cache in bytes"""
    total = 0
    for key, value in _cache.items():
        page, ts = value
        total += get_size(key) + get_size(value)  # key is str, value is tuple (Page, datetime)
    return total

def get_cached_slugs():
    """Extract unique slugs from cache keys"""
    slugs = set()
    for key in _cache:
        slug = key.split(':')[0]  # First part is slug
        slugs.add(slug)
    return sorted(list(slugs))  # Sort alphabetically for consistency

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        # Fallback if file doesn't exist
        return HTMLResponse(
            content="<h1>File not found!</h1><p>Please try again.</p>",
            status_code=404
        )

@app.get("/page/{slug:path}", response_model=Page, dependencies=[Depends(rate_limit_dependency)])
async def get_page(
    slug: str,
    extract_refs: bool = Query(True),
    truncate: Optional[int] = Query(None),
    citations: bool = Query(False),
    format: str = Query("markdown", description="Output format: 'markdown' (default) or 'text'")
):
    slug = normalize_slug(slug)

    cache_key = f"{slug}:{extract_refs}:{truncate or 'full'}:{citations}:{format}"
    now = datetime.now()

    if cache_key in _cache:
        page, ts = _cache[cache_key]
        if now - ts > CACHE_TTL:
            _cache.pop(cache_key)
        else:
            return page

    url = f"{BASE_URL}/page/{urllib.parse.quote(slug)}"

    resp = requests.get(url, headers={"User-Agent": "Grokipedia-API/0.4"})
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Not found: {slug}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Clean up: remove unwanted tags, but preserve references div
    unwanted_tags = ["script", "style", "nav", "header", "footer", "aside"]
    if not citations:
        unwanted_tags.append("sup")
    for tag in soup(unwanted_tags):
        tag.decompose()

    content_div = find_content_div(soup)

    h1 = content_div.find("h1")
    page_title = h1.get_text(strip=True) if h1 else slug.replace("_", " ")

    # Generate both plain text and markdown
    content_markdown = html_to_markdown(content_div, skip_metadata=True)
    content_text = re.sub(r'\n{3,}', '\n\n', content_div.get_text(separator="\n\n", strip=True))

    # Apply truncation
    if truncate:
        content_text = content_text[:truncate]
        content_markdown = content_markdown[:truncate]

    words = len(re.split(r'\s+', content_markdown.strip()))

    references, refs_count = extract_references(soup) if extract_refs else ([], 0)

    page_dict = {
        "title": page_title,
        "slug": slug,
        "url": url,
        "content_text": content_text,
        "content_markdown": content_markdown,
        "char_count": len(content_markdown),
        "word_count": words,
        "references_count": refs_count,
        "references": references,
    }
    page = Page(**page_dict)

    # Cache the new page (evict oldest if at max size)
    if len(_cache) >= MAX_CACHE_SIZE:
        _cache.popitem(last=False)  # Evict oldest (FIFO)
    _cache[cache_key] = (page, now)

    return page

@app.get("/health", include_in_schema=False)
async def health(key: str = Query(..., description="Secret key for health endpoint access")):
    health_secret = os.getenv("HEALTH_SECRET")  # Load from .env
    if not health_secret:
        raise HTTPException(status_code=500, detail="Health secret not configured in .env")
    if key != health_secret:
        raise HTTPException(status_code=401, detail="Unauthorized access to health endpoint")
    cache_size_bytes = get_cache_size_bytes()
    cache_size_mb = round(cache_size_bytes / (1024 * 1024), 2)
    cached_slugs = get_cached_slugs()
    return {
        "status": "Live",
        "cached_items": len(_cache),
        "cache_size_bytes": cache_size_bytes,
        "cache_size_mb": cache_size_mb,
        "cached_slugs": cached_slugs,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)