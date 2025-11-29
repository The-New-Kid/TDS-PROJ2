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
import ast
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup 
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)


# --- CONFIGURATION ---
STUDENT_EMAIL = "student@example.com" 
MY_SECRET = "your_chosen_secret"
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY") 


client = OpenAI(
    api_key=AIPIPE_API_KEY,
    base_url="https://aipipe.org/openrouter/v1"
)


# ------------------------------------------------------------------------------
# 1. Playwright (Extraction)
# ------------------------------------------------------------------------------
def extract_task_with_playwright(url):
    print(f"Playwright: Loading {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle") 
            page.wait_for_timeout(1000) 
            
            rendered_text = page.locator("body").inner_text()
            page_html = page.content()
            links = page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => a.href);
            }""")
            
            browser.close()
            return rendered_text.strip(), page_html, links
    except Exception as e:
        print(f"Playwright Error: {e}")
        return None, None, []


# ------------------------------------------------------------------------------
# 2. Downloader
# ------------------------------------------------------------------------------
def download_data_file(data_url, base_url=None):
    try:
        if not data_url or data_url == "NO_DATA_REQUIRED":
            return None
        
        # Make URL absolute if needed
        if base_url and not data_url.startswith('http'):
            data_url = urljoin(base_url, data_url)
        
        path = urlparse(data_url).path
        filename = os.path.basename(path)
        if not filename or len(filename) < 2: 
            filename = "temp_data_file.txt"
        
        print(f"Downloading file: {filename} from {data_url}")
        response = requests.get(data_url, timeout=10)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    except Exception as e:
        print(f"Download Error: {e}")
        return None


# ------------------------------------------------------------------------------
# 3. LLM Agent with FIXED Prompt (Don't wrap answer in JSON!)
# ------------------------------------------------------------------------------
def solve_with_llm(task_text, captured_links):
    prompt = """You are an LLM planning agent for a quiz solver.
Your only job is to analyze the task on the current page and return a small JSON plan describing how to solve it with Python code.

Return **only** a single JSON object with exactly these 3 keys:

{
  "data_url": "string",
  "submission_url": "string",
  "python_code": "string"
}

No extra keys. No comments. No markdown. No natural language. Only JSON.

---

## CRITICAL RULE FOR PYTHON CODE

**YOUR GENERATED PYTHON CODE MUST PRINT ONLY THE ANSWER VALUE, NOT JSON!**

✓ CORRECT:
print("secret123")
print(42)
print("https://example.com")

✗ WRONG - DO NOT DO THIS:
print(json.dumps({"answer": "secret123"}))
print(json.dumps({"email": "...", "secret": "...", "answer": "..."}))

The answer will be submitted as-is. Do not wrap it.

---

## 1. General behavior

- Read the task text carefully and literally.
- Use only information explicitly present in the task text and the detected links.
- Never guess column names, keys, URLs, or file types.
- Never invent links or endpoints.
- If something is ambiguous, choose the safest, most literal interpretation.

---

## 2. `data_url` decision

Set `"data_url"` as follows:

1. If the task explicitly asks to download a file (CSV, JSON, Excel, PDF, etc.) or scrape a specific link:
    - Set `"data_url"` to that link exactly as written.
    - If the link is relative, keep it relative.

2. If the task can be solved from the current page text:
    - Set `"data_url"` to `"NO_DATA_REQUIRED"`.

3. If multiple links are mentioned and you're unsure:
    - Set `"data_url"` to `"NO_DATA_REQUIRED"` and use page_html + task_text instead.

---

## 3. `submission_url` decision

- Look in `task_text` for: "POST your answer to …", "Submit to …", "Send to /submit"
- Extract the exact URL/path and put it in `"submission_url"`.
- If not specified, set to "" (empty string).

---

## 4. Python code requirements

Available variables:
- `DATA_FILE_PATH`: Local path to downloaded file (if data_url != "NO_DATA_REQUIRED"), else None
- `page_html`: Full HTML of current page
- `task_text`: Visible text of current page
- `base_url`: Current page's base URL
- `requests`: HTTP library (use for following referred links with timeout=10)
- `urljoin`: URL utility

Pre-imported libraries:
- pandas as pd
- pdfplumber
- json
- re
- math
- BeautifulSoup

**RULES:**
- Print ONLY the final answer (no JSON wrapper, no "Answer: " prefix, just the value)
- If extraction fails, print empty string "" (not an error message)
- Use try-except blocks to handle failures gracefully
- Last line MUST be: print(result)

---

## 4.1. When answer is on a referred link

If the page mentions a link to another page that contains the answer:

```python
from bs4 import BeautifulSoup
import requests

soup = BeautifulSoup(page_html, 'html.parser')

# Find the link
secret_link = soup.find('a', text=lambda x: 'secret' in x.lower() if x else False)

if secret_link:
    href = secret_link.get('href')
    full_url = urljoin(base_url, href)
    
    try:
        resp = requests.get(full_url, timeout=10)
        linked_soup = BeautifulSoup(resp.text, 'html.parser')
        secret_code = linked_soup.find(text=lambda x: 'secret code' in x.lower() if x else False)
        
        if secret_code:
            result = secret_code.split(':')[-1].strip()
        else:
            result = ""
    except:
        result = ""
else:
    result = ""

print(result)
```

**IMPORTANT:** Print the raw value, not JSON!

---

## 4.2. When answer is on current page

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(page_html, 'html.parser')

# Find the answer
answer_elem = soup.find(text=lambda x: 'answer' in x.lower() if x else False)

if answer_elem:
    result = answer_elem.strip()
else:
    result = ""

print(result)
```

---

## 5. Output requirements

- Last line: `print(result)` where result is the answer (string/number, NOT JSON)
- Do not print debug messages, labels, or multiple lines
- If extraction fails, print "" (empty string)

---

## 6. Examples

**Example 1: Answer on current page**
```json
{
  "data_url": "NO_DATA_REQUIRED",
  "submission_url": "/submit",
  "python_code": "soup = BeautifulSoup(page_html, 'html.parser')\nresult = soup.find(text=lambda x: 'answer' in x.lower() if x else False)\nif result:\n    result = result.strip().split(':')[-1].strip()\nelse:\n    result = ''\nprint(result)"
}
```

**Example 2: Answer on referred page**
```json
{
  "data_url": "NO_DATA_REQUIRED",
  "submission_url": "/submit",
  "python_code": "soup = BeautifulSoup(page_html, 'html.parser')\nlink = soup.find('a', text=lambda x: 'secret' in x.lower() if x else False)\nif link:\n    url = urljoin(base_url, link.get('href'))\n    try:\n        r = requests.get(url, timeout=10)\n        s = BeautifulSoup(r.text, 'html.parser')\n        code = s.find(text=lambda x: 'secret code' in x.lower() if x else False)\n        result = code.split(':')[-1].strip() if code else ''\n    except:\n        result = ''\nelse:\n    result = ''\nprint(result)"
}
```
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No explanations. Remember: print the answer VALUE, not JSON!"},
                {"role": "user", "content": prompt},
                {"role": "user", "content": "TASK TEXT:\n" + task_text},
                {"role": "user", "content": "LINKS DETECTED:\n" + str(captured_links)}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        print(f"LLM Error: {e}")
        return None


# ------------------------------------------------------------------------------
# 4. Execution Engine (Enhanced with debugging)
# ------------------------------------------------------------------------------
def execute_solution(plan, base_url, page_html_content, task_text_content):
    data_url = plan.get('data_url')
    
    if data_url and data_url != "NO_DATA_REQUIRED" and not data_url.startswith('http'):
        data_url = urljoin(base_url, data_url)

    local_filename = download_data_file(data_url, base_url)
    
    # Enhanced scope: add requests and urljoin for link-following
    local_scope = {
        'pd': pd,
        'pdfplumber': pdfplumber,
        'json': json,
        're': re,
        'math': sys.modules['math'],
        'BeautifulSoup': BeautifulSoup,
        'requests': requests,
        'urljoin': urljoin,
        'base_url': base_url,
        'DATA_FILE_PATH': local_filename,
        'page_html': page_html_content,
        'task_text': task_text_content  
    }

    captured_output = io.StringIO()
    captured_errors = io.StringIO()
    sys_stdout_backup = sys.stdout
    sys_stderr_backup = sys.stderr
    sys.stdout = captured_output
    sys.stderr = captured_errors
    
    result = ""
    try:
        code = plan.get('python_code', '')
        if (not code or code.strip() == ""):
            if plan.get('answer') is not None:
                result = str(plan.get('answer')).strip()
            else:
                result = ""
        else:
            exec(code, local_scope, local_scope)
            result = captured_output.getvalue().strip()
            
            # Fallback: if nothing printed, check for answer in plan
            if (not result) and plan.get('answer') is not None:
                result = str(plan.get('answer')).strip()
    except Exception as e:
        error_msg = captured_errors.getvalue() + str(e)
        print(f"[EXEC ERROR] {error_msg}", file=sys_stderr_backup)
        result = ""
    finally:
        sys.stdout = sys_stdout_backup
        sys.stderr = sys_stderr_backup
        if local_filename and os.path.exists(local_filename):
            try: 
                os.remove(local_filename)
            except: 
                pass
            
    return result


# ------------------------------------------------------------------------------
# 5. Background Solver
# ------------------------------------------------------------------------------
def background_solver(start_url):
    current_url = start_url
    attempt_count = 0
    max_attempts = 20
    
    while current_url and attempt_count < max_attempts:
        attempt_count += 1
        print(f"\n>>> PROCESSING (Attempt {attempt_count}): {current_url} <<<")
        
        # 1. Extract
        task_text, page_html, links = extract_task_with_playwright(current_url)
        if not task_text: 
            print("[ERROR] Failed to extract page content")
            break
        
        # 2. Plan
        plan = solve_with_llm(task_text, links)
        if not plan: 
            print("[ERROR] LLM failed to generate plan")
            break
        
        print(f"[PLAN] data_url: {plan.get('data_url')}")
        print(f"[PLAN] submission_url: {plan.get('submission_url')}")
        print(f"[CODE] {plan.get('python_code')[:200]}...")
        
        sub_url = plan.get('submission_url')
        if sub_url and not sub_url.startswith('http'):
            sub_url = urljoin(current_url, sub_url)
            
        # 3. Execute
        answer = execute_solution(plan, current_url, page_html, task_text)
        print(f"[ANSWER] {answer}")
        
        # Check if answer is empty
        if not answer or answer.strip() == "":
            print("[WARNING] Answer is empty, but submitting anyway...")
        
        # 4. Submit
        payload = {
            "email": STUDENT_EMAIL,
            "secret": MY_SECRET,
            "url": current_url,
            "answer": answer 
        }
        
        print(f"[SUBMIT] Posting to {sub_url}...")
        try:
            resp = requests.post(sub_url, json=payload, timeout=10)
            print(f"[RESPONSE] {resp.status_code} - {resp.text}")
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("url"):
                    current_url = data["url"]
                    print(f"[NEXT] Following to: {current_url}")
                else:
                    print("[SUCCESS] No next URL. Quiz complete!")
                    break
            else:
                print(f"[ERROR] Submission failed with status {resp.status_code}")
                break
        except Exception as e:
            print(f"[ERROR] Network error: {e}")
            break
    
    if attempt_count >= max_attempts:
        print(f"[LIMIT] Reached max attempts ({max_attempts})")


# ------------------------------------------------------------------------------
# 6. Flask Endpoint
# ------------------------------------------------------------------------------
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
    app.run(host='0.0.0.0', port=5000)
