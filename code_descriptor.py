import os
import ast
import shutil
import requests
import zipfile
import markdown2
from collections import defaultdict, deque
from graphviz import Digraph
from openai import OpenAI

# --- CONFIGURATION ---
PROJECT_TITLE = "Foundation Models Language Project"
GITHUB_LINK = "https://github.com/vimalthomas/foundationmodels"
OPENAI_API_KEY = "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"  # Replace your OpenAI API key here
client = OpenAI(api_key=OPENAI_API_KEY)

DOWNLOAD_PATH = './downloaded_repo'
OUTPUT_DOCS_PATH = './docs'
SUMMARY_PATH = os.path.join(OUTPUT_DOCS_PATH, 'project_summary.md')
FILE_DOCS_PATH = os.path.join(OUTPUT_DOCS_PATH, 'file_documentation.md')
DEPENDENCY_GRAPH_PATH = os.path.join(OUTPUT_DOCS_PATH, 'ordered_full_repo_structure.png')
FINAL_HTML_PATH = os.path.join(OUTPUT_DOCS_PATH, 'final_report.html')

# --- STATIC TEXTS ---
CODE_REVIEW_SECTION = """
## Code Review and Future Work

The project demonstrates strong modular design, clear organization of models, utilities, and training scripts. Code quality is solid with clear naming conventions and modularity principles.

Future improvements could include:
- Adding unit tests for critical modules and functions
- Increasing flexibility by externalizing model parameters via configuration files
- Implementing better exception handling and logging across modules
- Introducing continuous integration (CI) tools for automated testing
- Exploring further model enhancements or optimization techniques

Overall, the project is well-structured and easy for new developers to understand and contribute to.
"""

# --- UTILITY FUNCTIONS ---

def clean_up():
    if os.path.exists('repo.zip'):
        os.remove('repo.zip')
    if os.path.exists(DOWNLOAD_PATH):
        shutil.rmtree(DOWNLOAD_PATH)

def safe_summarize(prompt, critical=False):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert software engineer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Summarization failed: {e}")
        if critical:
            print("Critical summarization failure. Exiting...")
            clean_up()
            exit(1)
        return "Summary generation failed."

# --- PARSING AND DOCUMENTATION FUNCTIONS ---

def download_github_repo_zip(github_url, download_path):
    if github_url.endswith('/'):
        github_url = github_url[:-1]
    zip_url = github_url + '/archive/refs/heads/main.zip'
    response = requests.get(zip_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download GitHub repository from {zip_url}")
    with open('repo.zip', 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile('repo.zip', 'r') as zip_ref:
        zip_ref.extractall(download_path)
    extracted_folders = os.listdir(download_path)
    if extracted_folders:
        return os.path.join(download_path, extracted_folders[0])
    else:
        raise Exception("Extraction failed.")

def find_python_files(repo_path):
    py_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def get_module_name(filepath, repo_root):
    relative_path = os.path.relpath(filepath, repo_root)
    return relative_path.replace('/', '.').replace('\\', '.').replace('.py', '')

def parse_python_file(filepath):
    imports, classes, functions = [], {}, {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)

        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
            elif isinstance(node, ast.FunctionDef) and isinstance(getattr(node, 'parent', None), ast.Module):
                functions[node.name] = (node.lineno - 1, node.end_lineno)
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods[item.name] = (item.lineno - 1, item.end_lineno)
                classes[node.name] = {"range": (node.lineno - 1, node.end_lineno), "methods": methods}

    except Exception as e:
        print(f"Parsing error in {filepath}: {e}")
        code = ""
    return imports, classes, functions, code

def build_repo_structure(py_files, repo_root):
    repo_structure = {}
    module_to_file = {get_module_name(filepath, repo_root): filepath for filepath in py_files}

    for filepath in py_files:
        module_name = get_module_name(filepath, repo_root)
        imports, classes, functions, code = parse_python_file(filepath)
        local_imports = [imp for imp in imports if imp in module_to_file]
        repo_structure[module_name] = {
            "filepath": filepath,
            "imports": local_imports,
            "classes": classes,
            "functions": functions,
            "code": code
        }

    return repo_structure

def topological_sort(repo_structure):
    graph = defaultdict(list)
    indegree = defaultdict(int)

    for file, info in repo_structure.items():
        for dep in info['imports']:
            graph[dep].append(file)
            indegree[file] += 1

    queue = deque([node for node in repo_structure.keys() if indegree[node] == 0])
    sorted_files = []
    while queue:
        node = queue.popleft()
        sorted_files.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_files

def generate_ordered_dependency_graph(repo_structure, sorted_files, output_path):
    dot = Digraph(comment='Ordered Repo Structure')
    dot.attr(rankdir='LR')

    for file in sorted_files:
        dot.node(file, shape='box')
        
        classes = repo_structure[file]['classes']
        functions = repo_structure[file]['functions']

        for cls, info in classes.items():
            cls_node = f"{file}.{cls}"
            dot.node(cls_node, shape='ellipse')
            dot.edge(file, cls_node)
            for method in info["methods"].keys():
                method_node = f"{cls_node}.{method}"
                dot.node(method_node, shape='plaintext')
                dot.edge(cls_node, method_node)
        
        for func in functions.keys():
            func_node = f"{file}.{func}"
            dot.node(func_node, shape='plaintext')
            dot.edge(file, func_node)
 
    for file, contents in repo_structure.items():
        for imp in contents['imports']:
            if imp in repo_structure:
                dot.edge(file, imp, style='dashed', color='gray')

    dot.render(output_path.replace('.png', ''), format='png')
    print("Dependency graph generated.")

def generate_project_summary(repo_structure, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Project Overview\n\n")
        file_list = list(repo_structure.keys())

        full_prompt = "Summarize the overall purpose of this project based on these files:\n" + "\n".join(file_list)
        overall_summary = safe_summarize(full_prompt, critical=True)
        f.write(f"## Project Summary\n\n{overall_summary}\n\n")

        f.write("## File Contributions\n\n")
        for module, info in repo_structure.items():
            short_prompt = f"Summarize briefly what this file does:\n\n{info['code'][:4000]}"
            brief_summary = safe_summarize(short_prompt)
            f.write(f"- {module}.py: {brief_summary}\n")

def generate_file_documentation(repo_structure, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Detailed File-Level Documentation\n\n")
        for module, contents in repo_structure.items():
            code = contents['code']
            classes = contents['classes']
            functions = contents['functions']

            f.write(f"# {module}.py\n\n")

            file_prompt = f"Summarize this Python file:\n\n{code[:4000]}"
            file_summary = safe_summarize(file_prompt)
            f.write("## File Summary\n")
            f.write(file_summary + "\n\n")

            if classes:
                f.write("## Classes\n")
                for cls, info in classes.items():
                    start, end = info["range"]
                    class_code = "\n".join(code.splitlines()[start:end])
                    cls_prompt = f"Summarize this class:\n\n{class_code}"
                    cls_summary = safe_summarize(cls_prompt)
                    f.write(f"- {cls}: {cls_summary}\n\n")

            if functions:
                f.write("## Top-level Functions\n")
                for func, (start, end) in functions.items():
                    func_code = "\n".join(code.splitlines()[start:end])
                    func_prompt = f"Summarize this function:\n\n{func_code}"
                    func_summary = safe_summarize(func_prompt)
                    f.write(f"- {func}: {func_summary}\n\n")

def read_markdown_file(path):
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def build_final_report():
    project_summary = read_markdown_file(SUMMARY_PATH)
    file_documentation = read_markdown_file(FILE_DOCS_PATH)
    code_review_html = markdown2.markdown(CODE_REVIEW_SECTION)

    project_summary_html = markdown2.markdown(project_summary)
    file_docs_html = markdown2.markdown(file_documentation)

    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{PROJECT_TITLE}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #222;
            margin-top: 40px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }}
        a {{
            color: #1a0dab;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>

<h1>{PROJECT_TITLE}</h1>
<p><strong>GitHub Repository:</strong> <a href="{GITHUB_LINK}">{GITHUB_LINK}</a></p>

<hr>

<h2>Project Overview</h2>
{project_summary_html}

<hr>

<h2>Dependency Diagram</h2>
<img src="{os.path.basename(DEPENDENCY_GRAPH_PATH)}" alt="Dependency Graph">

<hr>

<h2>Detailed File Documentation</h2>
{file_docs_html}

<hr>

{code_review_html}

</body>
</html>
    """

    with open(FINAL_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"HTML report generated at {FINAL_HTML_PATH}")
    print("To create a PDF, open the HTML file in a browser and 'Save As PDF' manually.")

# --- MAIN DRIVER ---

def main():
    try:
        print("Starting project parsing and documentation generation...")
        repo_path = download_github_repo_zip(GITHUB_LINK, DOWNLOAD_PATH)
        print(f"Repository downloaded to: {repo_path}")

        py_files = find_python_files(repo_path)
        print(f"Found {len(py_files)} Python files.")

        repo_structure = build_repo_structure(py_files, repo_path)
        sorted_files = topological_sort(repo_structure)

        print("Generating dependency graph...")
        generate_ordered_dependency_graph(repo_structure, sorted_files, DEPENDENCY_GRAPH_PATH)

        print("Generating project overview documentation...")
        generate_project_summary(repo_structure, SUMMARY_PATH)

        print("Generating detailed file documentation...")
        generate_file_documentation(repo_structure, FILE_DOCS_PATH)

        print("Building final HTML report...")
        build_final_report()

        print("Project documentation completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up temporary files...")
        clean_up()

if __name__ == "__main__":
    main()
