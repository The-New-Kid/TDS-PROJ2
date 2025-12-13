import os
import json
import requests
import pandas as pd
from flask import Flask, request, jsonify
from openai import OpenAI
from playwright.sync_api import sync_playwright
import io
import sys
import threading
import pdfplumber
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup 
import traceback
if os.getenv("HF_SPACE") != "true":
    from dotenv import load_dotenv
    load_dotenv()

app = Flask(__name__)


# --- CONFIGURATION ---
STUDENT_EMAIL = "23f2005606@ds.study.iitm.ac.in" 
MY_SECRET = "your_chosen_secret"
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY") 


client = OpenAI(
    api_key=AIPIPE_API_KEY,
    base_url="https://aipipe.org/openrouter/v1"
)


# ==============================================================================
# EXPERT SYSTEM PROMPTS
# ==============================================================================

# ------------------------------------------------------------------------------
# CLASSIFIER PROMPT - Routes to appropriate expert
# ------------------------------------------------------------------------------
CLASSIFIER_PROMPT = """You are a task classifier for a quiz solver system.

Analyze the task and classify it into ONE of these categories:

1. **entry_page** - Instructions/intro page with no specific answer required
   Examples: "How to play:", "Start by POSTing", "After each submission", pages that just explain the rules
   KEY: If the page doesn't ask for a SPECIFIC piece of information, it's an entry page

2. **simple_answer** - Task asks for a literal string/number answer
   Examples: "answer anything you want", "type 'hello'", "respond with 42"

3. **scrape_current_page** - Answer is on the current page HTML
   Examples: "find the secret on this page", "what number is displayed", "extract the code from this page"

4. **scrape_external_page** - Need to fetch another HTML page and extract data
   Examples: "scrape URL X", "visit /data and get the code", "follow the link and extract"

5. **download_csv** - Download and process a CSV file
   Examples: "download data.csv and sum column X", "process the CSV file", "calculate average from CSV"

6. **download_pdf** - Download and process a PDF file
   Examples: "read document.pdf", "extract text from PDF", "find answer in PDF"

7. **download_excel** - Download and process an Excel file
   Examples: "process spreadsheet.xlsx", "read Excel file", "sum values in Excel"

8. **download_json** - Download and process a JSON file
   Examples: "fetch data.json", "parse JSON data", "extract from JSON file"

9. **api_call** - Make an API request
   Examples: "call the API at...", "GET request to...", "fetch from REST endpoint"

CRITICAL RULES:
- If page says "How to play", "Instructions", "Start by POSTing" → it's **entry_page**
- Only classify as scraping/download if there's a SPECIFIC data extraction task
- Look for action words: "find", "extract", "calculate", "download", "scrape"

Return ONLY a JSON object:
{
  "task_type": "one_of_the_above",
  "confidence": "high/medium/low",
  "reasoning": "brief explanation",
  "extracted_url": "URL if any, else null",
  "submission_url": "where to submit, default /submit"
}

TASK TEXT:
"""


# ------------------------------------------------------------------------------
# EXPERT: Entry Page (No answer required)
# ------------------------------------------------------------------------------
ENTRY_PAGE_EXPERT = """You are an expert at handling entry/instruction pages.

This page provides instructions but doesn't ask for a specific answer.
For entry pages, submit a placeholder value to proceed to the first real task.

Return ONLY a JSON object:
{
  "answer": "start"
}

Common placeholder values: "start", "begin", "ready", "ok"

TASK TEXT:
"""


# ------------------------------------------------------------------------------
# EXPERT: Simple Answer
# ------------------------------------------------------------------------------
SIMPLE_ANSWER_EXPERT = """You are an expert at extracting literal answers from task descriptions.

The task asks for a simple, direct answer. Extract exactly what's requested.

Return ONLY a JSON object:
{
  "answer": "the literal answer as a string"
}

Examples:
- "answer anything you want" → {"answer": "anything you want"}
- "type the number 42" → {"answer": "42"}
- "respond with 'hello world'" → {"answer": "hello world"}

TASK TEXT:
"""


# ------------------------------------------------------------------------------
# EXPERT: Scrape Current Page
# ------------------------------------------------------------------------------
SCRAPE_CURRENT_PAGE_EXPERT = """You are an expert at extracting data from HTML pages.

The answer is on the CURRENT page. Generate Python code to extract it from `page_html`.

Available variables:
- page_html: Full HTML of current page
- task_text: Visible text
- BeautifulSoup, re, sys

Return ONLY a JSON object:
{
  "python_code": "code that extracts and prints the answer"
}

RULES:
- Use BeautifulSoup to parse page_html
- Print ONLY the answer value (no JSON, no labels)
- Use try-except, print empty string on failure
- Add debug output to sys.stderr

Example:
{
  "python_code": "from bs4 import BeautifulSoup\\nimport sys\\ntry:\\n    soup = BeautifulSoup(page_html, 'html.parser')\\n    text = soup.get_text()\\n    print(f'DEBUG: Page text: {text[:100]}', file=sys.stderr)\\n    match = re.search(r'secret[:\\\\s]+(\\\\w+)', text, re.I)\\n    result = match.group(1) if match else ''\\n    print(f'DEBUG: Found: {result}', file=sys.stderr)\\nexcept Exception as e:\\n    print(f'DEBUG: Error: {e}', file=sys.stderr)\\n    result = ''\\nprint(result)"
}

TASK TEXT:
"""


# ------------------------------------------------------------------------------
# EXPERT: Scrape External Page
# ------------------------------------------------------------------------------
SCRAPE_EXTERNAL_PAGE_EXPERT = """You are an expert at identifying URLs to scrape.

The task requires fetching an external page. Your job is to identify the target URL.

Return ONLY a JSON object:
{
  "target_url": "the URL mentioned in the task (relative or absolute)"
}

Examples:
- "Scrape /demo-scrape-data?email=..." → {"target_url": "/demo-scrape-data?email=..."}
- "Visit https://example.com/api" → {"target_url": "https://example.com/api"}
- "Get data from /data/info.html" → {"target_url": "/data/info.html"}

Extract the EXACT URL from the task description.

TASK TEXT:
"""


# ------------------------------------------------------------------------------
# EXPERT: Download CSV
# ------------------------------------------------------------------------------
DOWNLOAD_CSV_EXPERT = """You are an expert at processing CSV files with pandas.

The task requires downloading and analyzing a CSV file.

Available variables:
- DATA_FILE_PATH: Local path to downloaded CSV
- pd (pandas)

Return ONLY a JSON object:
{
  "file_url": "URL of CSV file to download",
  "python_code": "code that processes CSV and prints answer"
}

RULES:
- Use pd.read_csv(DATA_FILE_PATH)
- Handle missing columns gracefully
- Print ONLY the final answer
- Add debug output to sys.stderr

Example:
{
  "file_url": "/data/sales.csv",
  "python_code": "import pandas as pd\\nimport sys\\ntry:\\n    print(f'DEBUG: Reading {DATA_FILE_PATH}', file=sys.stderr)\\n    df = pd.read_csv(DATA_FILE_PATH)\\n    print(f'DEBUG: Columns: {df.columns.tolist()}', file=sys.stderr)\\n    print(f'DEBUG: Shape: {df.shape}', file=sys.stderr)\\n    result = str(df['amount'].sum())\\n    print(f'DEBUG: Result: {result}', file=sys.stderr)\\nexcept Exception as e:\\n    print(f'DEBUG: Error: {e}', file=sys.stderr)\\n    result = ''\\nprint(result)"
}

TASK TEXT:
"""


# ------------------------------------------------------------------------------
# EXPERT: Download PDF
# ------------------------------------------------------------------------------
DOWNLOAD_PDF_EXPERT = """You are an expert at extracting text from PDFs with pdfplumber.

Available variables:
- DATA_FILE_PATH: Local path to downloaded PDF
- pdfplumber

Return ONLY a JSON object:
{
  "file_url": "URL of PDF to download",
  "python_code": "code that extracts text and prints answer"
}

Example:
{
  "file_url": "/docs/report.pdf",
  "python_code": "import pdfplumber\\nimport sys\\ntry:\\n    print(f'DEBUG: Opening {DATA_FILE_PATH}', file=sys.stderr)\\n    with pdfplumber.open(DATA_FILE_PATH) as pdf:\\n        text = ''.join([p.extract_text() or '' for p in pdf.pages])\\n        print(f'DEBUG: Extracted {len(text)} chars', file=sys.stderr)\\n        match = re.search(r'answer[:\\\\s]+(\\\\w+)', text, re.I)\\n        result = match.group(1) if match else ''\\n        print(f'DEBUG: Found: {result}', file=sys.stderr)\\nexcept Exception as e:\\n    print(f'DEBUG: Error: {e}', file=sys.stderr)\\n    result = ''\\nprint(result)"
}

TASK TEXT:
"""


# ==============================================================================
# INTELLIGENT SCRAPER - Uses Playwright for JS-heavy pages
# ==============================================================================
def scrape_with_playwright(url):
    """
    Scrape a URL using Playwright (handles JavaScript).
    Returns the rendered HTML and text.
    """
    print(f"[SCRAPER] Fetching with Playwright: {url}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"])

            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(2000)  # Wait for JS to execute
            
            html = page.content()
            text = page.locator("body").inner_text()
            
            browser.close()
            
            print(f"[SCRAPER] ✓ Got {len(html)} chars HTML, {len(text)} chars text")
            return html, text
            
    except Exception as e:
        print(f"[SCRAPER] ✗ Error: {e}")
        return None, None


def extract_answer_from_content(html, text):
    """
    Multi-method extraction from HTML and text.
    """
    soup = BeautifulSoup(html, 'html.parser')
    result = ""
    
    print(f"[EXTRACTOR] Starting extraction...")
    print(f"[EXTRACTOR] HTML preview: {html[:300]}")
    print(f"[EXTRACTOR] Text preview: {text[:200]}")
    
    # Method 1: Check common HTML tags
    print(f"[EXTRACTOR] Method 1: Checking HTML tags...")
    for tag in ['strong', 'b', 'em', 'span', 'code', 'mark']:
        elem = soup.find(tag)
        if elem and elem.get_text().strip():
            result = elem.get_text().strip()
            print(f"[EXTRACTOR] ✓ Found in <{tag}>: {result}")
            return result
    
    # Method 2: Check divs with common IDs
    print(f"[EXTRACTOR] Method 2: Checking div IDs...")
    for div_id in ['question', 'answer', 'secret', 'code', 'result']:
        elem = soup.find(id=div_id)
        if elem:
            elem_text = elem.get_text().strip()
            print(f"[EXTRACTOR] Found div#{div_id}: {elem_text[:100]}")
            
            if elem_text:
                # Try to extract numbers
                match = re.search(r'(\d{4,})', elem_text)
                if match:
                    result = match.group(1)
                    print(f"[EXTRACTOR] ✓ Extracted number: {result}")
                    return result
                # Or just return the text if it's short
                if len(elem_text) < 50:
                    result = elem_text
                    print(f"[EXTRACTOR] ✓ Extracted text: {result}")
                    return result
    
    # Method 3: Regex patterns on plain text
    print(f"[EXTRACTOR] Method 3: Trying regex patterns...")
    patterns = [
        (r'[Ss]ecret code is\s*(\d+)', 'secret code is'),
        (r'[Ss]ecret code:\s*(\d+)', 'secret code:'),
        (r'[Cc]ode is\s*(\d+)', 'code is'),
        (r'[Aa]nswer is\s*(\d+)', 'answer is'),
        (r'[Aa]nswer:\s*(\d+)', 'answer:'),
        (r'[Ss]ecret:\s*(\d+)', 'secret:'),
        (r'(\d{5})', 'any 5-digit number'),
        (r'(\d{4})', 'any 4-digit number')
    ]
    
    for pattern, desc in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            print(f"[EXTRACTOR] ✓ Matched '{desc}': {result}")
            return result
    
    # Method 4: Look for any standalone numbers
    print(f"[EXTRACTOR] Method 4: Looking for standalone numbers...")
    numbers = re.findall(r'\b\d{4,}\b', text)
    if numbers:
        result = numbers[0]
        print(f"[EXTRACTOR] ✓ Found standalone number: {result}")
        return result
    
    print(f"[EXTRACTOR] ✗ No extraction method succeeded")
    print(f"[EXTRACTOR] Full text: {text}")
    return ""


# ==============================================================================
# PLAYWRIGHT EXTRACTION
# ==============================================================================
def extract_task_with_playwright(url):
    print(f"[PLAYWRIGHT] Loading {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
            )

            page = browser.new_page()
            page.goto(url, wait_until="networkidle") 
            page.wait_for_timeout(1000) 
            
            rendered_text = page.locator("body").inner_text()
            page_html = page.content()
            links = page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => a.href);
            }""")
            
            browser.close()
            print(f"[PLAYWRIGHT] ✓ Extracted {len(rendered_text)} chars, {len(links)} links")
            return rendered_text.strip(), page_html, links
    except Exception as e:
        print(f"[PLAYWRIGHT] ✗ Error: {e}")
        return None, None, []


# ==============================================================================
# FILE DOWNLOADER
# ==============================================================================
def download_data_file(data_url, base_url=None):
    try:
        if not data_url:
            return None
        
        if base_url and not data_url.startswith('http'):
            data_url = urljoin(base_url, data_url)
        
        path = urlparse(data_url).path
        filename = os.path.basename(path) or "temp_data_file"
        
        print(f"[DOWNLOAD] Fetching: {data_url}")
        response = requests.get(data_url, timeout=10)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"[DOWNLOAD] ✓ Saved: {filename} ({len(response.content)} bytes)")
        return filename
    except Exception as e:
        print(f"[DOWNLOAD] ✗ Error: {e}")
        return None


# ==============================================================================
# LLM CALL HELPER
# ==============================================================================
def call_llm(prompt, return_json=True):
    """Generic LLM caller"""
    try:
        messages = [
            {"role": "system", "content": "Return ONLY valid JSON. No markdown, no explanations."},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0
        }
        
        if return_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        completion = client.chat.completions.create(**kwargs)
        response_text = completion.choices[0].message.content
        
        if return_json:
            return json.loads(response_text)
        return response_text
        
    except Exception as e:
        print(f"[LLM] ✗ Error: {e}")
        return None


# ==============================================================================
# EXPERT SYSTEM - Task Classification & Routing
# ==============================================================================
def classify_task(task_text, links):
    """Step 1: Classify the task type"""
    print(f"[CLASSIFIER] Analyzing task...")
    
    prompt = CLASSIFIER_PROMPT + task_text + "\n\nLINKS DETECTED:\n" + str(links)
    classification = call_llm(prompt)
    
    if not classification:
        return None
    
    print(f"[CLASSIFIER] ✓ Type: {classification.get('task_type')}")
    print(f"[CLASSIFIER] ✓ Confidence: {classification.get('confidence')}")
    print(f"[CLASSIFIER] ✓ Reasoning: {classification.get('reasoning')}")
    
    return classification


def route_to_expert(classification, task_text, links):
    """Step 2: Route to appropriate expert and get solution"""
    task_type = classification.get('task_type')
    print(f"[EXPERT] Routing to: {task_type}")
    
    # Route to appropriate expert
    if task_type == 'entry_page':
        prompt = ENTRY_PAGE_EXPERT + task_text
        expert_response = call_llm(prompt)
        return {
            'method': 'direct_answer',
            'answer': expert_response.get('answer', ''),
            'data_url': None,
            'submission_url': classification.get('submission_url', '/submit')
        }
    
    elif task_type == 'simple_answer':
        prompt = SIMPLE_ANSWER_EXPERT + task_text
        expert_response = call_llm(prompt)
        return {
            'method': 'direct_answer',
            'answer': expert_response.get('answer'),
            'data_url': None,
            'submission_url': classification.get('submission_url', '/submit')
        }
    
    elif task_type == 'scrape_current_page':
        prompt = SCRAPE_CURRENT_PAGE_EXPERT + task_text
        expert_response = call_llm(prompt)
        return {
            'method': 'execute_code',
            'python_code': expert_response.get('python_code'),
            'data_url': None,
            'submission_url': classification.get('submission_url', '/submit')
        }
    
    elif task_type == 'scrape_external_page':
        prompt = SCRAPE_EXTERNAL_PAGE_EXPERT + task_text + "\n\nLINKS:\n" + str(links)
        expert_response = call_llm(prompt)
        return {
            'method': 'scrape_url',
            'target_url': expert_response.get('target_url'),
            'data_url': None,
            'submission_url': classification.get('submission_url', '/submit')
        }
    
    elif task_type == 'download_csv':
        prompt = DOWNLOAD_CSV_EXPERT + task_text + "\n\nLINKS:\n" + str(links)
        expert_response = call_llm(prompt)
        return {
            'method': 'execute_code',
            'python_code': expert_response.get('python_code'),
            'data_url': expert_response.get('file_url'),
            'submission_url': classification.get('submission_url', '/submit')
        }
    
    elif task_type == 'download_pdf':
        prompt = DOWNLOAD_PDF_EXPERT + task_text + "\n\nLINKS:\n" + str(links)
        expert_response = call_llm(prompt)
        return {
            'method': 'execute_code',
            'python_code': expert_response.get('python_code'),
            'data_url': expert_response.get('file_url'),
            'submission_url': classification.get('submission_url', '/submit')
        }
    
    else:
        print(f"[EXPERT] ✗ Unknown task type: {task_type}")
        return None


# ==============================================================================
# CODE EXECUTION ENGINE
# ==============================================================================
def execute_solution(solution, base_url, page_html_content, task_text_content):
    """Execute the solution from the expert system"""
    
    # Method 1: Direct answer (no code execution needed)
    if solution.get('method') == 'direct_answer':
        answer = solution.get('answer', '')
        print(f"[EXECUTE] Direct answer: '{answer}'")
        return answer
    
    # Method 2: Scrape a URL with Playwright
    if solution.get('method') == 'scrape_url':
        target_url = solution.get('target_url')
        if not target_url:
            print(f"[EXECUTE] ✗ No target URL provided")
            return ""
        
        # Make URL absolute
        if not target_url.startswith('http'):
            target_url = urljoin(base_url, target_url)
        
        # Scrape with Playwright
        html, text = scrape_with_playwright(target_url)
        
        if not html:
            print(f"[EXECUTE] ✗ Failed to scrape URL")
            return ""
        
        # Extract answer using intelligent extraction
        answer = extract_answer_from_content(html, text)
        print(f"[EXECUTE] ✓ Extracted: '{answer}'")
        return answer
    
    # Method 3: Execute Python code
    if solution.get('method') == 'execute_code':
        # Download file if needed
        data_url = solution.get('data_url')
        local_filename = None
        
        if data_url:
            if not data_url.startswith('http'):
                data_url = urljoin(base_url, data_url)
            local_filename = download_data_file(data_url, base_url)
        
        # Prepare execution environment
        local_scope = {
            'pd': pd,
            'pdfplumber': pdfplumber,
            'json': json,
            're': re,
            'math': __import__('math'),
            'BeautifulSoup': BeautifulSoup,
            'requests': requests,
            'urljoin': urljoin,
            'sys': sys,
            'base_url': base_url,
            'DATA_FILE_PATH': local_filename,
            'page_html': page_html_content,
            'task_text': task_text_content
        }
        
        # Capture output
        captured_output = io.StringIO()
        captured_errors = io.StringIO()
        sys_stdout_backup = sys.stdout
        sys_stderr_backup = sys.stderr
        sys.stdout = captured_output
        sys.stderr = captured_errors
        
        result = ""
        try:
            code = solution.get('python_code', '')
            print(f"[EXECUTE] Running code ({len(code)} chars)...", file=sys_stderr_backup)
            
            exec(code, local_scope, local_scope)
            result = captured_output.getvalue().strip()
            
            print(f"[EXECUTE] ✓ Result: '{result[:100]}'", file=sys_stderr_backup)
            
        except Exception as e:
            print(f"[EXECUTE] ✗ Error: {e}", file=sys_stderr_backup)
            print(f"[EXECUTE] Traceback:\n{traceback.format_exc()}", file=sys_stderr_backup)
            result = ""
            
        finally:
            sys.stdout = sys_stdout_backup
            sys.stderr = sys_stderr_backup
            
            # Show debug output from code
            stderr_content = captured_errors.getvalue()
            if stderr_content:
                print(f"[DEBUG OUTPUT]\n{stderr_content}")
            
            # Cleanup downloaded file
            if local_filename and os.path.exists(local_filename):
                try:
                    os.remove(local_filename)
                except:
                    pass
        
        return result
    
    return ""


# ==============================================================================
# MAIN SOLVER LOOP
# ==============================================================================
def background_solver(start_url):
    current_url = start_url
    attempt_count = 0
    max_attempts = 20
    
    while current_url and attempt_count < max_attempts:
        attempt_count += 1
        print(f"\n{'='*80}")
        print(f"🔄 ATTEMPT {attempt_count}/{max_attempts}")
        print(f"🌐 URL: {current_url}")
        print(f"{'='*80}\n")
        
        # Step 1: Extract page content
        task_text, page_html, links = extract_task_with_playwright(current_url)
        if not task_text:
            print("❌ [ERROR] Failed to extract page content")
            break
        
        print(f"[TASK] {task_text[:200]}...")
        
        # Step 2: Classify task type
        classification = classify_task(task_text, links)
        if not classification:
            print("❌ [ERROR] Failed to classify task")
            break
        
        # Step 3: Route to expert and get solution
        solution = route_to_expert(classification, task_text, links)
        if not solution:
            print("❌ [ERROR] Failed to get solution from expert")
            break
        
        # Step 4: Execute solution
        answer = execute_solution(solution, current_url, page_html, task_text)
        print(f"\n[ANSWER] '{answer}'")
        
        # Step 5: Submit answer
        sub_url = solution.get('submission_url', '/submit')
        if not sub_url.startswith('http'):
            sub_url = urljoin(current_url, sub_url)
        
        payload = {
            "email": STUDENT_EMAIL,
            "secret": MY_SECRET,
            "url": current_url,
            "answer": answer
        }
        
        print(f"\n[SUBMIT] → {sub_url}")
        
        try:
            resp = requests.post(sub_url, json=payload, timeout=10)
            print(f"[RESPONSE] Status: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                
                if data.get("correct"):
                    print("✅ [SUCCESS] Answer correct!")
                else:
                    print(f"❌ [FAILURE] Wrong answer")
                    if data.get('reason'):
                        print(f"    Reason: {data.get('reason')}")
                
                if data.get("url"):
                    current_url = data["url"]
                    print(f"➡️  [NEXT] Moving to next question")
                else:
                    print("🎉 [COMPLETE] Quiz finished!")
                    break
            else:
                print(f"❌ [ERROR] HTTP {resp.status_code}: {resp.text}")
                break
                
        except Exception as e:
            print(f"❌ [ERROR] Submit failed: {e}")
            traceback.print_exc()
            break
    
    if attempt_count >= max_attempts:
        print(f"⚠️  [LIMIT] Reached max attempts ({max_attempts})")


# ==============================================================================
# FLASK ENDPOINT
# ==============================================================================
@app.route('/', methods=['POST'])
def handle_quiz():
    try:
        data = request.get_json(force=True)
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    if not data or data.get('secret') != MY_SECRET:
        return jsonify({"error": "Access Denied"}), 403
    
    if not data.get('url'):
        return jsonify({"error": "No URL provided"}), 400
    
    thread = threading.Thread(target=background_solver, args=(data['url'],))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "Processing started"}), 200


if __name__ == '__main__':
    print("🚀 Expert System Quiz Solver Starting...")
    print(f"📧 Email: {STUDENT_EMAIL}")
    print(f"🔒 Secret: {MY_SECRET[:4]}...")
    print(f"🎯 Ready to solve quizzes!\n")
    app.run(host='0.0.0.0', port=7860)